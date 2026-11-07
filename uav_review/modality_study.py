"""The same model roster and recipe for RGB, thermal and verified paired fusion.

Two unmatched collections remain separate exploratory cohorts. If a verified
aligned pair manifest is supplied, all three modalities share cases, labels and
partitions. No pairing, flight metadata or model result is manufactured.
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import re
import shutil

from .audit import audit_dataset
from .core import digest_object, read_csv, read_json, sha256, write_csv, write_json
from .fusion import inspect_pairs, materialize_paired_sources
from .notebook import prepare_notebook_protocol
from .protocol import assert_no_leakage


MODALITIES = ("thermal", "rgb", "fusion")
DATASET_KEYS = {"thermal": "thermal_v4", "rgb": "rgb_frames"}
MODEL_NAMES = {"yolov5s": "YOLOv5s", "yolov5su": "YOLOv5su", "yolov8n": "YOLOv8n",
               "yolov8m": "YOLOv8m", "yolov8l": "YOLOv8l", "rtdetr_l": "RT-DETR-L",
               "yolov10n": "YOLOv10n", "yolo11n": "YOLOv11n", "yolo12n": "YOLOv12n", "yolo26n": "YOLO26n"}


def _path(root, value):
    value = Path(value).expanduser()
    return (root / value).resolve() if not value.is_absolute() else value.resolve()


def load_study_config(root, overrides=None):
    root = Path(root).expanduser().resolve()
    config = read_json(root / "multimodal_config.json")
    for key, value in (overrides or {}).items():
        if value is not None:
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(copy.deepcopy(value))
            else: config[key] = copy.deepcopy(value)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["study_name"])
            or config["study_name"].rstrip(".") != config["study_name"]):
        raise ValueError("Use a simple portable study_name, such as all_modalities_v1.")
    if not config["modalities"] or not set(config["modalities"]) <= set(MODALITIES):
        raise ValueError("Choose modalities from thermal, rgb and fusion.")
    if len(set(config["modalities"])) != len(config["modalities"]):
        raise ValueError("Duplicate modality names are not allowed.")
    if not config["models"] or not set(config["models"]) <= set(MODEL_NAMES):
        raise ValueError("Choose registered, unambiguous model identities.")
    if len(config["models"]) != len(set(config["models"])):
        raise ValueError("Duplicate model aliases are not allowed.")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Specify distinct training seeds.")
    if config["evaluation_mode"] not in {"exploratory", "verified_sessions", "grouped_cv"}:
        raise ValueError("evaluation_mode must be exploratory, verified_sessions or grouped_cv.")
    if config["design"] not in {"auto", "paired", "independent"}:
        raise ValueError("design must be auto, paired or independent.")
    return root, config


def _protocol_options(config):
    return {"mode": {"exploratory": "exploratory", "verified_sessions": "holdout",
                     "grouped_cv": "grouped_cv"}[config["evaluation_mode"]],
            "folds": int(config["folds"]), "seed": int(config["split_seed"])}


def _inventory(root, source):
    if not source.is_dir():
        raise FileNotFoundError(f"Modality input folder is missing: {source}")
    rows = []
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".txt"}:
            rows.append({"path": p.relative_to(source).as_posix(), "sha256": sha256(p)})
    return rows


def _shared_partition_id(rows):
    return digest_object(sorted((r["image_id"], r["split"], r["leakage_group"]) for r in rows))


def _write_protocol_copy(reference, manifest, destination, paired_partition_id):
    destination.mkdir(parents=True, exist_ok=True)
    old = read_json(reference / "protocol.json")
    ref_rows = read_csv(reference / "assignments.csv")
    by_id = {r["image_id"]: r for r in manifest}
    rows = []
    for r in ref_rows:
        m = by_id[r["image_id"]]
        merged = {**r, **m, "split": r["split"], "leakage_group": r["leakage_group"]}
        if old["mode"] == "exploratory":
            merged.update(session_id="", provenance_verified="false", source_video="",
                          source_group=r["source_group"])
        # Hash the exact CSV representation consumed by every backend.
        rows.append({k: str(v) for k, v in merged.items()})
    parts = {s: [r for r in rows if r["split"] == s] for s in ["train", "val", "test"]}
    assert_no_leakage(parts, old["grouping_field"])
    assert _shared_partition_id(rows) == paired_partition_id
    write_csv(destination / "assignments.csv", rows)
    for split, selected in parts.items():
        (destination / f"{split}.txt").write_text("\n".join(r["image_path"] for r in selected) + "\n", encoding="utf-8")
    write_json(destination / "data.yaml", {
        "path": destination.resolve().as_posix(), "nc": 1, "names": ["Human"],
        **{s: (destination / f"{s}.txt").resolve().as_posix() for s in parts}})
    protocol = {**old, "dataset": rows[0]["dataset"], "modality": rows[0]["declared_modality"],
                "assignments_sha": digest_object(rows), "source_manifest_sha": digest_object(manifest),
                "paired_sensors_verified": True, "paired_case_partition_id": paired_partition_id,
                "counts": {s: {"images": len(items), "humans": sum(int(r["n_instances"]) for r in items),
                               "sessions": len({r["session_id"] for r in items if r["session_id"]}),
                               "groups": len({r["leakage_group"] for r in items})} for s, items in parts.items()},
                "note": "Matched RGB/thermal/fused cases share verified registration, labels and the same split. "
                        + ("Source-flight identities remain unverified; image-group uncertainty is not flight uncertainty."
                           if old["mode"] == "exploratory" else "Source sessions were supplied and verified separately.")}
    protocol.pop("protocol_sha", None)
    protocol["protocol_sha"] = digest_object(protocol)
    write_json(destination / "protocol.json", protocol)
    return destination


def prepare_study(package_root, overrides=None):
    """Prepare local paths and frozen partitions, without importing Torch or training."""
    root, config = load_study_config(package_root, overrides)
    study = _path(root, config["output_root"]) / config["study_name"]
    pairs_csv = _path(root, config["fusion"]["pairs_csv"])
    entries = inspect_pairs(root, pairs_csv, config["fusion"]["rgb_weight"]) if config["design"] != "independent" else []
    if config["design"] == "paired" and not entries:
        raise ValueError("A paired three-modality study needs actual registered pairs and a completed pairs_VERIFIED.csv.")
    paired = bool(entries)
    design = "paired_modality_comparison" if paired else "independent_unpaired_cohorts"
    signature_inputs = {"configuration": config, "design": design}
    if paired:
        signature_inputs["pairs"] = [{k: e[k] for k in ["pair_id", "rgb_sha256", "thermal_sha256",
            "label_sha256", "shared_group", "session_id", "provenance_verified"]} for e in entries]
    else:
        signature_inputs["sources"] = {
            m: _inventory(root, _path(root, config["source_roots"][m]))
            for m in config["modalities"] if m != "fusion"}
        if config["evaluation_mode"] != "exploratory":
            signature_inputs["session_maps"] = {
                m: sha256(_path(root, config["sessions"][m])) for m in signature_inputs["sources"]}
    signature = digest_object(signature_inputs)
    marker = study / "study_inputs.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["input_signature"] != signature or saved["local_package_root"] != root.as_posix():
            raise ValueError("Data, study settings or package location changed. Choose a new study_name; existing results are preserved.")
        plan = read_json(study / "study_plan.json")
        # Verify cached data files and frozen lists before any result is reused.
        from .engine import check_protocol
        for cohort in plan["cohorts"].values():
            for fold in cohort["folds"]: check_protocol(fold)
        print(f"Reusing verified unchanged local data plan: {study}", flush=True)
        return plan
    study.mkdir(parents=True, exist_ok=True)
    options = _protocol_options(config)
    cohorts, statuses = {}, []
    if paired:
        manifests = materialize_paired_sources(entries, study / "paired_inputs", config["fusion"]["rgb_weight"])
        # Build the reference split once, then apply its exact case assignments.
        reference = study / "shared_reference_protocol"
        reference_manifest = study / "paired_inputs/rgb/image_manifest.csv"
        mapping = study / "paired_inputs/paired_sessions.csv"
        write_csv(mapping, [{"image_id": "paired/" + e["pair_id"], "session_id": e["session_id"],
                             "provenance_verified": e["provenance_verified"]} for e in entries])
        prepare_notebook_protocol(reference_manifest, mapping if options["mode"] != "exploratory" else None,
                                  reference, **options)
        for m in config["modalities"]:
            folds = []
            for ref in sorted(reference.glob("fold_*")):
                shared = _shared_partition_id(read_csv(ref / "assignments.csv"))
                dest = study / m / "protocol" / ref.name
                folds.append(str(_write_protocol_copy(ref, manifests[m], dest, shared)))
            cohorts[m] = {"modality": m, "dataset": "paired_" + m, "folds": folds,
                           "images": len(manifests[m]), "paired_cases": True}
            statuses.append({"modality": m, "status": "READY", "source_images": len(entries),
                             "design": design, "reason": "Same matched cases, common-coordinate labels and shared split."})
    else:
        for m in config["modalities"]:
            if m == "fusion":
                statuses.append({"modality": m, "status": "AWAITING_VERIFIED_PAIRS", "source_images": 0,
                    "design": design, "reason": "No synchronized, registered RGB/thermal pairs with common labels were supplied."})
                continue
            source = _path(root, config["source_roots"][m])
            audit = study / m / "audit"
            manifest = audit_dataset(None, DATASET_KEYS[m], audit, dataset_root=source)
            sessions = _path(root, config["sessions"][m]) if options["mode"] != "exploratory" else None
            protocol = study / m / "protocol"
            prepare_notebook_protocol(audit / "image_manifest.csv", sessions, protocol, **options)
            cohorts[m] = {"modality": m, "dataset": DATASET_KEYS[m],
                           "folds": [str(p) for p in sorted(protocol.glob("fold_*"))],
                           "images": len(manifest), "paired_cases": False}
            statuses.append({"modality": m, "status": "READY", "source_images": len(manifest),
                             "design": design, "reason": "Separate cohort; differences from another modality are confounded by different scenes."})
    scheduled = sum(len(c["folds"]) for c in cohorts.values()) * len(config["models"]) * len(config["seeds"])
    plan = {"package_root": str(root), "study_dir": str(study), "design": design, "config": config,
            "cohorts": cohorts, "modality_status": statuses, "scheduled_training_jobs": scheduled,
            "requested_modalities": config["modalities"], "input_signature": signature,
            "fusion_method": "registered fixed-weight three-channel input blend; not learned feature fusion",
            "source_sessions_verified": config["evaluation_mode"] != "exploratory"}
    write_json(study / "study_inputs.json", {"input_signature": signature, "local_package_root": root.as_posix(),
                                             "inputs": signature_inputs})
    write_json(study / "study_plan.json", plan)
    write_csv(study / "modality_status.csv", statuses)
    print(f"Prepared {design}: {scheduled} fresh model/seed/fold jobs.", flush=True)
    for row in statuses: print(f"  {row['modality']}: {row['status']} ({row['source_images']} images)", flush=True)
    return plan


def freeze_training_configs(plan, device=None, training_overrides=None):
    """One common model list, seed list and training recipe for every modality."""
    from .notebook import resolve_device
    root, study = Path(plan["package_root"]), Path(plan["study_dir"])
    base = read_json(root / "configs/primary.json")
    base.update(models=plan["config"]["models"], seeds=plan["config"]["seeds"],
                device=resolve_device(device or plan["config"]["device"]), weights_dir=str(root / "weights"))
    base["training"].update(training_overrides or {})
    base["training"]["workers"] = 0
    if base["device"] == "cpu": base["training"]["amp"] = False
    for name in ["epochs", "imgsz", "batch"]:
        if int(base["training"][name]) <= 0: raise ValueError(f"training {name} must be positive.")
    for m in plan["cohorts"]:
        path = study / m / "primary_config.json"
        if path.exists() and read_json(path) != base:
            raise ValueError("A frozen training recipe changed. Choose a new study_name for a different experiment.")
        write_json(path, base)
    return base


def execute_study(plan, do_train=True, do_evaluate=True, do_timing=True,
                  device=None, training_overrides=None):
    """Run all available modalities; preserve failures and continue other jobs."""
    from .notebook import train_notebook_jobs, run_notebook_measurement
    root, study = Path(plan["package_root"]), Path(plan["study_dir"])
    recipe = freeze_training_configs(plan, device, training_overrides)
    timings = plan["config"]["timing"]
    records = []
    for modality, cohort in plan["cohorts"].items():
        config = study / modality / "primary_config.json"
        for fold_string in cohort["folds"]:
            fold = Path(fold_string)
            training = study / modality / "training" / fold.name
            for model in recipe["models"]:
                for seed in recipe["seeds"]:
                    record = {"modality": modality, "model": model, "seed": seed, "fold": fold.name,
                              "status": "NOT_STARTED", "stage": "training", "run_dir": "", "error": ""}
                    try:
                        if not do_train:
                            record.update(status="SKIPPED", error="Training disabled; no completed results are claimed.")
                            records.append(record); continue
                        runs = train_notebook_jobs(root, config, fold, training, model_keys=[model], seed_values=[seed])
                        run = runs[0]; record.update(status="TRAINED", run_dir=str(run))
                        if do_evaluate:
                            record["stage"] = "evaluation"
                            run_notebook_measurement(root, "evaluate", run, precision="fp32")
                            record["status"] = "EVALUATED"
                        if do_timing and seed == recipe["seeds"][0]:
                            record["stage"] = "timing"
                            run_notebook_measurement(root, "benchmark", run, precision=timings["precision"],
                                seconds=float(timings["seconds"]), warmup=int(timings["warmup"]),
                                platform_type=timings["platform_type"])
                            record["status"] = "EVALUATED_AND_TIMED" if do_evaluate else "TRAINED_AND_TIMED"
                        record["stage"] = "complete"
                    except KeyboardInterrupt:
                        record.update(status="INTERRUPTED", error="User interrupted the child process.")
                        records.append(record); write_csv(study / "job_status.csv", records); raise
                    except Exception as exc:
                        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                        print(f"FAILED {modality}/{model}/{seed}/{fold.name}: {exc}", flush=True)
                    records.append(record)
                    write_csv(study / "job_status.csv", records)
    write_csv(study / "job_status.csv", records,
              ["modality", "model", "seed", "fold", "status", "stage", "run_dir", "error"])
    return records


def report_study(plan):
    """Summarize saved evaluations, with explicit expected/completed run counts."""
    from .notebook import run_notebook_command
    root, study = Path(plan["package_root"]), Path(plan["study_dir"])
    report_root = study / "reports" / datetime.now().strftime("report_%Y%m%d_%H%M%S_%f")
    report_root.mkdir(parents=True, exist_ok=False)
    rows, statuses = [], []
    for modality, cohort in plan["cohorts"].items():
        training = study / modality / "training"
        expected = len(plan["config"]["models"]) * len(plan["config"]["seeds"]) * len(cohort["folds"])
        completed = list(training.rglob("evaluation_full_fp32/test_metrics.json")) if training.exists() else []
        recipe_path = study / modality / "primary_config.json"
        recipe = read_json(recipe_path) if recipe_path.is_file() else {}
        settings = recipe.get("training", {})
        primary_protocol = (set(plan["config"]["models"]) == set(MODEL_NAMES)
                            and plan["config"]["seeds"] == [17, 42, 93]
                            and settings.get("epochs") == 100 and settings.get("imgsz") == 640
                            and settings.get("batch") == 8)
        timed = list(training.rglob("timing_full_fp32_workstation/timing_summary.json")) if training.exists() else []
        expected_timing = len(plan["config"]["models"]) * len(cohort["folds"])
        status = {"modality": modality, "expected_runs": expected, "evaluated_runs": len(completed),
                  "complete_declared_evaluation": len(completed) == expected,
                  "matches_primary_100epoch_3seed_protocol": primary_protocol,
                  "complete_primary_comparison": len(completed) == expected and primary_protocol,
                  "expected_timing_runs": expected_timing, "completed_timing_runs": len(timed),
                  "complete_declared_timing": len(timed) == expected_timing,
                  "design": plan["design"], "paired_cases": cohort["paired_cases"]}
        statuses.append(status)
        if not completed: continue
        if plan["config"]["evaluation_mode"] == "grouped_cv" and len(completed) != expected:
            status["reason"] = "Complete all required outer folds before aggregating this grouped-CV comparison."
            continue
        out = report_root / modality
        run_notebook_command(root, ["report", "--results", training, "--out", out, "--bootstrap", "0"],
                             log_file=study / modality / "report.log")
        for file in out.rglob("model_comparison_mean_SD.csv"):
            for row in read_csv(file):
                rows.append({"modality": modality, "model_display": MODEL_NAMES[row["model"]],
                             **row, "study_design": plan["design"], "cohort_complete": status["complete_primary_comparison"]})
    for s in plan["modality_status"]:
        if s["status"] != "READY":
            statuses.append({"modality": s["modality"], "expected_runs": 0, "evaluated_runs": 0,
                             "complete_primary_comparison": False, "design": plan["design"], "reason": s["reason"]})
    write_csv(report_root / "completion_status.csv", statuses)
    if rows: write_csv(report_root / "all_modalities_summary.csv", rows)
    write_json(report_root / "report_scope.json", {
        "design": plan["design"], "paired_modality_comparison": plan["design"] == "paired_modality_comparison",
        "source_sessions_verified": plan["source_sessions_verified"], "statuses": statuses,
        "cross_modality_causal_comparison_permitted": plan["design"] == "paired_modality_comparison",
        "note": "Same models and budgets do not remove scene/dataset confounding in unmatched cohorts. "
                "Seed SD is not confidence across independent flights."})
    (report_root / "README.md").write_text(
        "# Modality study results\n\n"
        "See completion_status.csv before interpreting a comparison as complete. "
        "all_modalities_summary.csv is written only when real evaluated runs exist. "
        "Each modality folder contains per-seed/mean-SD CSVs, learning curves, PR/confusion figures, "
        "matched qualitative examples and runtime records when measured.\n\n"
        + ("All modality inputs share verified aligned cases, labels and partitions.\n"
           if plan["design"] == "paired_modality_comparison" else
           "RGB and thermal use DIFFERENT collections. Their scores are descriptive separate studies, not a controlled modality/fusion comparison. Fusion has no fabricated images or results.\n"),
        encoding="utf-8")
    print(f"Saved numerical and visual results: {report_root}", flush=True)
    return report_root


def preview_sources(plan, per_modality=2):
    """Show actual input images; no generated empirical examples."""
    from IPython.display import display
    from PIL import Image
    for modality, cohort in plan["cohorts"].items():
        rows = read_csv(Path(cohort["folds"][0]) / "assignments.csv")
        for row in rows[:per_modality]:
            print(f"{modality}: {row['image_id']} | split={row['split']} | people={row['n_instances']}")
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB"); image.thumbnail((640, 420)); display(image)
