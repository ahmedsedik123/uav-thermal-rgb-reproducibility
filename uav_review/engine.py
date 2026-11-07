"""Fresh training, fixed-checkpoint inference, and measurements on the actual host."""
from __future__ import annotations

import json
import random
import re
import shutil
import time
from pathlib import Path

from PIL import Image

from .core import (digest_object, environment, read_csv, read_json, require_empty,
                   sha256, write_csv, write_json)
from .metrics import (coco_metrics, error_analysis, nms, point_metrics,
                      select_confidence, truths)
from .paths import local_path, validate_manifest_paths


MODELS = {
    "yolov5s": {"loader": "CLASSIC_YOLOV5", "weights": "yolov5s.pt", "family": "yolov5", "scale": "s", "nms": None, "head": "anchor_based_external_NMS"},
    "yolov5su": {"loader": "YOLO", "weights": "yolov5su.pt", "family": "yolov5", "scale": "s", "nms": None, "head": "anchor_free_YOLOv5u_external_NMS"},
    "yolov8n": {"loader": "YOLO", "weights": "yolov8n.pt", "family": "yolov8", "nms": None, "head": "one_to_many_external_NMS"},
    "yolov8m": {"loader": "YOLO", "weights": "yolov8m.pt", "family": "yolov8", "scale": "m", "nms": None, "head": "one_to_many_external_NMS"},
    "yolov8l": {"loader": "YOLO", "weights": "yolov8l.pt", "family": "yolov8", "scale": "l", "nms": None, "head": "one_to_many_external_NMS"},
    "yolov10n": {"loader": "YOLO", "weights": "yolov10n.pt", "family": "yolov10", "nms": False, "head": "native_end_to_end"},
    "yolo11n": {"loader": "YOLO", "weights": "yolo11n.pt", "family": "yolo11", "nms": None, "head": "one_to_many_external_NMS"},
    "yolo12n": {"loader": "YOLO", "weights": "yolo12n.pt", "family": "yolo12", "nms": None, "head": "one_to_many_external_NMS"},
    "yolo26n": {"loader": "YOLO", "weights": "yolo26n.pt", "family": "yolo26", "nms": False, "head": "native_end_to_end"},
    "rtdetr_l": {"loader": "RTDETR", "weights": "rtdetr-l.pt", "family": "rtdetr", "scale": "l", "nms": False, "head": "native_end_to_end"},
}


def validate_model_identity(key, model):
    spec = MODELS[key]
    architecture = getattr(model.model, "yaml", {}) or {}
    yaml_file = str(architecture.get("yaml_file", "")).lower().replace("\\", "/")
    # Official older YOLO12 checkpoints store yolov12n.yaml; current names omit v.
    canonical_yaml = re.sub(r"yolov(?=\d)", "yolo", yaml_file).replace("rt-detr", "rtdetr")
    canonical_family = re.sub(r"yolov(?=\d)", "yolo", spec["family"]).replace("rt-detr", "rtdetr")
    if yaml_file and canonical_family not in canonical_yaml:
        raise ValueError(f"Architecture/model label mismatch: {key} resolves to {yaml_file}")
    expected_scale = spec.get("scale") or (key[-1] if key.startswith("yolo") and key[-1] in "nsmlx" else None)
    actual_scale = architecture.get("scale")
    if not actual_scale and yaml_file:
        match = re.search(re.escape(canonical_family) + r"-?([nsmlx])(?:u)?\.ya?ml$", canonical_yaml)
        actual_scale = match.group(1) if match else None
    if expected_scale and actual_scale and expected_scale != actual_scale:
        raise ValueError(f"Model scale mismatch: {key} resolves to scale {actual_scale}")
    return architecture


def load_model(key, checkpoint=None, weights_dir=None):
    if key not in MODELS:
        raise ValueError(f"Unregistered model: {key}")
    spec = MODELS[key]
    initial = Path(weights_dir) / spec["weights"] if weights_dir else spec["weights"]
    if weights_dir and not Path(initial).is_file():
        raise FileNotFoundError(initial)
    if spec["loader"] == "CLASSIC_YOLOV5":
        from .classic_yolov5 import ClassicYolov5
        model = ClassicYolov5(weights=initial, checkpoint=checkpoint)
    else:
        from ultralytics import YOLO, RTDETR
        model = (RTDETR if spec["loader"] == "RTDETR" else YOLO)(str(checkpoint or initial))
    validate_model_identity(key, model)
    model.overrides["nms"] = spec["nms"]
    return model


def synchronize(device):
    import torch
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device(device))


def cuda_device(device):
    import torch
    return torch.device(f"cuda:{device}" if str(device).isdigit() else device)


def validate_device(device):
    import torch
    if str(device) != "cpu":
        normalized = cuda_device(device)
        if normalized.type != "cuda" or normalized.index is None:
            raise ValueError("Choose one explicit CUDA device, such as 0 or cuda:0.")
    if str(device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable; choose cpu explicitly for a CPU experiment.")


def check_protocol(fold):
    from .protocol import assert_no_leakage
    fold = local_path(fold)
    protocol = read_json(fold / "protocol.json")
    rows = read_csv(fold / "assignments.csv")
    if protocol["assignments_sha"] != digest_object(rows):
        # CSV round-trips turn numeric values into strings; normalizing at split creation avoids ambiguity.
        normalized = [{k: str(v) for k, v in r.items()} for r in rows]
        if protocol["assignments_sha"] != digest_object(normalized):
            raise ValueError("Protocol assignments changed after splitting. Regenerate the protocol.")
    exploratory = protocol["mode"] == "exploratory"
    grouping_field = "source_group" if exploratory else "session_id"
    if protocol.get("grouping_field", grouping_field) != grouping_field:
        raise ValueError("Protocol grouping field disagrees with its evaluation mode.")
    if exploratory:
        if (protocol.get("source_sessions_verified") is not False
                or any(r.get("session_id") or r.get("provenance_verified", "").lower() != "false" for r in rows)):
            raise ValueError("Exploratory partitions must not declare verified flight/session identities.")
    assert_no_leakage({split: [r for r in rows if r["split"] == split] for split in ["train", "val", "test"]},
                      group_field=grouping_field)
    validate_manifest_paths(rows)
    # Verify the actual file lists consumed by the backend, not just the manifest.
    dataset_yaml = read_json(fold / "data.yaml")
    if "path" in dataset_yaml and local_path(dataset_yaml["path"]) != fold:
        raise ValueError("data.yaml points to a different local folder. Rerun audit/split on this machine.")
    if dataset_yaml.get("names") != ["Human"] or dataset_yaml.get("nc") != 1:
        raise ValueError("data.yaml class mapping changed. Regenerate the protocol.")
    for split in ["train", "val", "test"]:
        expected_list = fold / f"{split}.txt"
        if not dataset_yaml.get(split) or not Path(dataset_yaml[split]).is_absolute() or local_path(dataset_yaml[split]) != expected_list:
            raise ValueError(f"data.yaml has a stale or changed {split} path. Regenerate the protocol locally.")
        actual_paths = expected_list.read_text(encoding="utf-8").splitlines()
        expected_paths = [r["image_path"] for r in rows if r["split"] == split]
        if actual_paths != expected_paths:
            raise ValueError(f"{split}.txt differs from the frozen assignments. Regenerate the protocol.")
    for row in rows:
        if sha256(row["image_path"]) != row["sha256"]:
            raise ValueError(f"Image changed since audit: {row['image_id']}")
        if sha256(row["label_path"]) != row.get("label_sha256"):
            raise ValueError(f"Annotation changed since audit: {row['image_id']}")
    return fold, protocol, rows


def train(config_path, fold_path, output, model_keys=None, seeds=None):
    from .protocol import protocol_scope
    config = read_json(config_path)
    fold, protocol, rows = check_protocol(fold_path)
    config["models"] = model_keys or config["models"]
    config["seeds"] = seeds or config["seeds"]
    validate_device(config["device"])
    output = require_empty(output)
    write_json(output / "frozen_config.json", config)
    write_json(output / "environment.json", environment())
    for key in config["models"]:
        for seed in config["seeds"]:
            print(f"Starting {key}, seed {seed}, device {config['device']}, workers {config['training'].get('workers', 0)}", flush=True)
            # A fresh pretrained model is instantiated for EVERY seed and variant.
            run_dir = output / f"{key}__seed_{seed}"
            run_dir.mkdir()
            metadata = {"status": "INITIALIZING", "model_key": key, "seed": seed,
                        "dataset": protocol["dataset"], "modality": protocol["modality"],
                        "protocol_sha": protocol["protocol_sha"], "fold": str(fold),
                        "config_sha": digest_object(config), "device": config["device"],
                        "config": config, "environment": environment(), **protocol_scope(protocol)}
            write_json(run_dir / "run.json", metadata)
            try:
                model = load_model(key, weights_dir=config.get("weights_dir"))
                original_checkpoint = Path(str(getattr(model, "ckpt_path", MODELS[key]["weights"])))
                metadata["initial_checkpoint"] = str(original_checkpoint)
                if not original_checkpoint.is_file():
                    raise FileNotFoundError(f"Cannot fingerprint the resolved initialization checkpoint: {original_checkpoint}")
                metadata["initial_sha256"] = sha256(original_checkpoint)
                metadata["head_mode"] = MODELS[key]["head"]
                metadata["architecture"] = json.loads(json.dumps(getattr(model.model, "yaml", {}), default=str))
                metadata["parameters"] = sum(p.numel() for p in model.model.parameters())
                metadata["initial_parameters"] = metadata["parameters"]
                try:
                    from ultralytics.utils.torch_utils import get_flops
                    flops = get_flops(model.model, imgsz=int(config["training"]["imgsz"]))
                    metadata["profiled_GFLOPs"] = float(flops) if flops and flops > 0 else None
                except Exception:
                    metadata["profiled_GFLOPs"] = None
                metadata["initial_profiled_GFLOPs"] = metadata["profiled_GFLOPs"]
                best = {"AP50_95": -1.}
                selected = run_dir / "best_mAP50_95.pt"

                def save_best(trainer):
                    metric = trainer.metrics.get("metrics/mAP50-95(B)")
                    if metric is None:
                        raise RuntimeError("Backend did not report mAP50-95; inspect the backend version.")
                    if float(metric) > best["AP50_95"]:
                        best.update(AP50_95=float(metric), epoch=int(trainer.epoch)+1)
                        shutil.copy2(trainer.last, selected)
                        write_json(run_dir / "checkpoint_selection.json", {
                            **best, "criterion": "maximum validation mAP50-95", "split": "val"})

                model.add_callback("on_model_save", save_best)
                args = {**config["training"], "data": str(fold / "data.yaml"),
                        "device": config["device"], "seed": int(seed),
                        "project": str(run_dir.resolve()), "name": "fit", "exist_ok": False,
                        "save": True, "save_period": -1, "val": True,
                        "nms": MODELS[key]["nms"],
                        "deterministic": key != "rtdetr_l"}
                # RT-DETR CUDA grid-sample backward is not strictly deterministic.
                metadata["strict_determinism_requested"] = args["deterministic"]
                metadata["training_args"] = args
                start = time.perf_counter()
                from .classic_yolov5 import blocked_optional_imports
                # Optional installed Albumentations must not add undeclared transforms.
                with blocked_optional_imports(["albumentations"]):
                    model.train(**args)
                synchronize(model.trainer.device)
                metadata["training_wall_seconds"] = time.perf_counter()-start
                metadata["actual_device"] = str(model.trainer.device)
                metadata["resolved_training_args"] = json.loads(json.dumps(vars(model.trainer.args), default=str))
                metadata["actual_batch_size"] = int(model.trainer.batch_size)
                if int(model.trainer.batch_size) != int(config["training"]["batch"]):
                    raise RuntimeError("Backend changed the batch size. Choose one feasible common batch and rerun the primary comparison.")
                if not selected.is_file():
                    raise RuntimeError("No validation-selected checkpoint was saved.")
                history = Path(getattr(model.trainer, "history_path", Path(model.trainer.save_dir) / "results.csv"))
                shutil.copy2(history, run_dir / "history.csv")
                logged = read_csv(history)
                epoch_values = [float(next(v for k, v in r.items() if k.strip() == "epoch")) for r in logged]
                if any(b <= a for a, b in zip(epoch_values, epoch_values[1:])):
                    raise ValueError("Non-monotonic history: multiple runs must not be concatenated.")
                if len(logged) != int(config["training"]["epochs"]):
                    raise RuntimeError("Training did not complete the fixed epoch budget. Preserve this attempt and rerun before comparing it.")
                metadata.update(status="TRAINED", checkpoint=str(selected.resolve()),
                                checkpoint_sha256=sha256(selected), selection=best,
                                actual_logged_epochs=len(logged))
                metadata["parameters"] = sum(p.numel() for p in model.model.parameters())
                metadata["parameter_scope"] = "trained single-class model"
                try:
                    flops = get_flops(model.model, imgsz=int(config["training"]["imgsz"]))
                    metadata["profiled_GFLOPs"] = float(flops) if flops and flops > 0 else None
                except Exception:
                    metadata["profiled_GFLOPs"] = None
                metadata["backend_details"] = getattr(model, "backend_details", {"backend": "ultralytics_8.4.183"})
                metadata["optional_albumentations_disabled"] = True
            except Exception as exc:
                metadata.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                write_json(run_dir / "run.json", metadata)
                raise
            write_json(run_dir / "run.json", metadata)
            print(f"Completed {key}, seed {seed}: {run_dir}", flush=True)
    return output


def tile_origins(length, size, overlap):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid tile size/overlap")
    if length <= size:
        return [0]
    step = max(1, round(size*(1-overlap)))
    return sorted(set(list(range(0, length-size+1, step)) + [length-size]))


def infer_image(model, path, device, imgsz, variant="full", tile_size=320, overlap=0.2, half=False):
    """Outputs original-image xyxy boxes. Tiled timing includes crop, inference, and merge."""
    def predict(source):
        result = model.predict(source=source, imgsz=imgsz, device=device, batch=1,
                               conf=0.001, iou=0.7, max_det=300, quantize=16 if half else 32,
                               nms=model.overrides.get("nms"),
                               save=False, verbose=False)[0]
        items = []
        if result.boxes is not None:
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                if int(cls) != 0:
                    raise ValueError("The fine-tuned detector must contain the single Human class 0.")
                items.append({"xyxy": [float(v) for v in box], "score": float(score), "class_id": 0})
        return items, result.speed
    if variant == "full":
        return predict(str(path))
    if variant != "tiled":
        raise ValueError("variant must be full or tiled")
    predictions = []
    with Image.open(path) as image:
        image = image.convert("RGB")
        for y in tile_origins(image.height, tile_size, overlap):
            for x in tile_origins(image.width, tile_size, overlap):
                tile = image.crop((x, y, min(x+tile_size, image.width), min(y+tile_size, image.height)))
                items, _ = predict(tile)
                for item in items:
                    left, top, right, bottom = item["xyxy"]
                    item["xyxy"] = [left+x, top+y, right+x, bottom+y]
                    predictions.append(item)
    return nms(predictions, 0.55), {}


def get_trained_run(path):
    run_dir = Path(path).resolve()
    run = read_json(run_dir / "run.json")
    if run["status"] != "TRAINED":
        raise ValueError("Only successfully completed training runs can be evaluated.")
    if sha256(run["checkpoint"]) != run["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after training.")
    fold, protocol, rows = check_protocol(run["fold"])
    if run["protocol_sha"] != protocol["protocol_sha"]:
        raise ValueError("Training/evaluation protocols differ.")
    return run_dir, run, rows


def evaluate(run_path, variant="full", tile_size=320, overlap=0.2, precision="fp32", instance_conditions=None):
    from .protocol import protocol_scope
    run_dir, run, rows = get_trained_run(run_path)
    scope = protocol_scope(read_json(Path(run["fold"]) / "protocol.json"))
    config = run["config"]
    validate_device(config["device"])
    if precision not in {"fp32", "fp16"} or (precision == "fp16" and str(config["device"]) == "cpu"):
        raise ValueError("FP16 evaluation requires CUDA.")
    output = require_empty(run_dir / f"evaluation_{variant}_{precision}")
    imgsz = int(config["training"]["imgsz"])
    model = load_model(run["model_key"], run["checkpoint"])
    if len(model.names) != 1:
        raise ValueError("Checkpoint is not a fine-tuned single-class detector.")
    # Freeze inference settings before the test predictions are generated.
    settings = {"variant": variant, "precision": precision,
                "tile_size": tile_size, "overlap": overlap, "imgsz": imgsz,
                "prediction_confidence_floor": 0.001, "NMS_IoU": 0.7,
                "head_mode": MODELS[run["model_key"]]["head"],
                "global_tiled_merge_IoU": 0.55, "COCO_max_detections": 100,
                "checkpoint_sha256": run["checkpoint_sha256"], "protocol_sha": run["protocol_sha"],
                "instance_conditions_sha256": sha256(instance_conditions) if instance_conditions else None,
                **scope}
    write_json(output / "frozen_evaluation_settings.json", settings)
    prediction_sets = {}
    for split in ["val", "test"]:
        selected = [r for r in rows if r["split"] == split]
        predictions = {}
        for row in selected:
            predictions[row["image_id"]], _ = infer_image(
                model, row["image_path"], config["device"], imgsz, variant, tile_size, overlap, precision == "fp16")
        prediction_sets[split] = predictions
        write_json(output / f"{split}_predictions.json", predictions)
        if split == "val":
            confidence, curve = select_confidence(selected, predictions)
            write_json(output / "operating_point.json", {
                "confidence": confidence, "selection_split": "val",
                "selection_criterion": "maximum F1 at IoU 0.50 on fixed grid 0.05:0.05:0.95"})
            write_csv(output / "validation_threshold_sweep.csv", curve)
    test = [r for r in rows if r["split"] == "test"]
    ap = coco_metrics(test, prediction_sets["test"], return_curve=True)
    point = point_metrics(test, prediction_sets["test"], confidence)
    images, instances, sizes, conditions, instance_strata = error_analysis(
        test, prediction_sets["test"], confidence, read_csv(instance_conditions) if instance_conditions else None)
    for name, values in [("image_errors", images), ("instance_errors", instances),
                         ("size_analysis", sizes), ("condition_analysis", conditions),
                         ("occlusion_truncation_analysis", instance_strata)]:
        write_csv(output / f"{name}.csv", values)
    write_json(output / "test_metrics.json", {
        **ap, **point, "model": run["model_key"], "seed": run["seed"],
        "dataset": run["dataset"], "modality": run["modality"],
        "protocol_sha": run["protocol_sha"], "variant": variant, "numerical_precision": precision,
        "training_config_sha": run["config_sha"], **scope})
    return output


def benchmark(run_path, seconds=60, warmup=50, precision="fp32", platform_type="workstation",
              variant="full", tile_size=320, overlap=0.2):
    import numpy as np
    import torch
    import psutil
    run_dir, run, rows = get_trained_run(run_path)
    if seconds <= 0 or warmup < 0:
        raise ValueError("Invalid timing duration/warm-up count")
    if platform_type not in {"workstation", "embedded"}:
        raise ValueError("Specify the actual platform type")
    device = run["config"]["device"]
    validate_device(device)
    if precision not in {"fp32", "fp16"} or (precision == "fp16" and str(device) == "cpu"):
        raise ValueError("FP16 measurements require a CUDA device; FP32 is the CPU baseline.")
    half = precision == "fp16"
    output = require_empty(run_dir / f"timing_{variant}_{precision}_{platform_type}")
    samples = [r for r in rows if r["split"] == "val"]
    model = load_model(run["model_key"], run["checkpoint"])
    imgsz = int(run["config"]["training"]["imgsz"])
    for i in range(warmup):
        infer_image(model, samples[i % len(samples)]["image_path"], device, imgsz, variant, tile_size, overlap, half)
    if warmup == 0:
        infer_image(model, samples[0]["image_path"], device, imgsz, variant, tile_size, overlap, half)
    runtime_device = model.predictor.device
    synchronize(runtime_device)
    if str(device) != "cpu":
        torch.cuda.reset_peak_memory_stats(runtime_device)
    measurements = []
    power_samples = []
    # Sandboxed hosts may not expose this process through /proc. Missing
    # optional RSS telemetry must not discard otherwise valid timing samples.
    rss_status = "AVAILABLE"
    try:
        process = psutil.Process()
        peak_rss = process.memory_info().rss
    except (psutil.Error, OSError) as exc:
        process, peak_rss = None, None
        rss_status = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    nvml = handle = None
    try:
        import pynvml
        pynvml.nvmlInit()
        properties = torch.cuda.get_device_properties(runtime_device)
        uuid = getattr(properties, "uuid", None)
        if uuid:
            handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
        else:
            import os
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise RuntimeError("Cannot reliably map CUDA to NVML with the available device metadata.")
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(str(runtime_device).split(":")[-1]))
        nvml = pynvml
    except Exception:
        pass
    begin = time.perf_counter()
    i = 0
    try:
        while time.perf_counter()-begin < seconds:
            sample = samples[i % len(samples)]
            synchronize(runtime_device)
            start = time.perf_counter()
            _, components = infer_image(model, sample["image_path"], device, imgsz, variant, tile_size, overlap, half)
            synchronize(runtime_device)
            elapsed = (time.perf_counter()-start)*1000
            measurements.append({"image_id": sample["image_id"], "wall_latency_ms": elapsed,
                                 **{f"library_reported_{k}_ms": float(v) for k, v in components.items()}})
            if process is not None:
                try:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                except (psutil.Error, OSError) as exc:
                    process = None
                    rss_status = f"PARTIAL_SAMPLES: {type(exc).__name__}: {exc}"
            if nvml and i % 10 == 0:
                try:
                    power_samples.append({"elapsed_s": time.perf_counter()-begin,
                                          "gpu_power_W": nvml.nvmlDeviceGetPowerUsage(handle)/1000,
                                          "gpu_temperature_C": nvml.nvmlDeviceGetTemperature(handle, 0)})
                except Exception:
                    pass
            i += 1
        duration = time.perf_counter()-begin
    finally:
        if nvml:
            nvml.nvmlShutdown()
    latencies = [r["wall_latency_ms"] for r in measurements]
    if not latencies:
        raise RuntimeError("No timing samples recorded")
    write_csv(output / "latency_samples.csv", measurements)
    write_csv(output / "GPU_sensor_samples.csv", power_samples,
              ["elapsed_s", "gpu_power_W", "gpu_temperature_C"])
    write_json(output / "timing_summary.json", {
        "checkpoint_sha256": run["checkpoint_sha256"], "protocol_sha": run["protocol_sha"],
        "platform_type": platform_type, "environment": environment(),
        "requested_device": device, "device": str(runtime_device),
        "device_name": torch.cuda.get_device_name(runtime_device) if str(device) != "cpu" else "CPU",
        "precision": precision, "variant": variant, "imgsz": imgsz, "batch_size": 1,
        "tile_size": tile_size, "overlap": overlap,
        "requested_seconds": seconds, "requested_warmup_iterations": warmup,
        "head_mode": MODELS[run["model_key"]]["head"],
        "warmup_iterations": max(1, warmup), "measured_images": len(latencies), "sustained_seconds": duration,
        "measured_loop_FPS": len(latencies)/duration,
        "reciprocal_mean_latency_FPS": 1000/float(np.mean(latencies)),
        "latency_mean_ms": float(np.mean(latencies)), "latency_std_ms": float(np.std(latencies)),
        "latency_median_ms": float(np.median(latencies)), "latency_P95_ms": float(np.percentile(latencies, 95)),
        "max_sampled_process_RSS_MiB": peak_rss/1024**2 if peak_rss is not None else None,
        "process_RSS_measurement_status": rss_status,
        "peak_torch_allocated_MiB": torch.cuda.max_memory_allocated(runtime_device)/1024**2 if str(device) != "cpu" else None,
        "mean_GPU_power_W": float(np.mean([p["gpu_power_W"] for p in power_samples])) if power_samples else None,
        "power_scope": "GPU sensor only; not total platform power",
        "includes": ["image_file_decode", "preprocessing", "detector", "postprocessing", "tile_merge_if_enabled"],
        "excludes": ["camera_capture", "UAV_communication", "tracking", "result_transmission"],
        "claim_scope": "offline-image processing on the named host; not a complete UAV deployment test",
    })
    return output


def export_onnx(run_path, output):
    run_dir, run, _ = get_trained_run(run_path)
    output = require_empty(output)
    model = load_model(run["model_key"], run["checkpoint"])
    exported = Path(model.export(
        format="onnx", imgsz=int(run["config"]["training"]["imgsz"]),
        opset=17, dynamic=False, simplify=False, quantize=32, device=run["config"]["device"]))
    target = output / exported.name
    shutil.copy2(exported, target)
    write_json(output / "export_manifest.json", {
        "model": run["model_key"], "source_checkpoint_sha256": run["checkpoint_sha256"],
        "ONNX_sha256": sha256(target), "format": "onnx", "precision": "fp32", "opset": 17,
        "accuracy_equivalence_status": "NOT_YET_TESTED",
        "deployment_status": "EXPORT_ONLY_NOT_HARDWARE_VALIDATION",
        "next_requirement": "Validate accuracy/output equivalence and measure the actual target-device pipeline."})
    return output
