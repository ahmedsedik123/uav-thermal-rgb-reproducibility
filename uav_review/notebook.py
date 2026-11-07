"""Anaconda/Jupyter setup and isolated commands; imports here use only the stdlib."""
from __future__ import annotations

import copy
import importlib
import importlib.metadata
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from urllib.parse import urlparse


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_notebook_dependencies(package_root, install=True, torch_index_url="", reinstall_torch=False):
    """Use the running kernel's interpreter, preserving an existing Torch install by default."""
    if sys.version_info < (3, 10) or struct.calcsize("P") * 8 != 64:
        raise RuntimeError("Choose a 64-bit Python 3.10+ kernel. The supplied Anaconda environment uses Python 3.11.")
    package_root = Path(package_root).expanduser().resolve()
    print(f"Notebook Python: {sys.executable}", flush=True)
    print(f"Python prefix: {sys.prefix}", flush=True)
    if not install:
        print("Package installation skipped; the next cell checks the actual runtime.", flush=True)
        return {"installed": False, "python_executable": sys.executable}
    tracked = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
               "PIL": "Pillow", "matplotlib": "matplotlib", "yaml": "PyYAML",
               "cv2": "opencv-python", "pycocotools": "pycocotools", "ultralytics": "ultralytics",
               "pandas": "pandas", "seaborn": "seaborn", "scipy": "scipy", "requests": "requests",
               "thop": "ultralytics-thop"}
    loaded = {dist: _version(dist) for module, dist in tracked.items() if module in sys.modules}
    if reinstall_torch and not torch_index_url:
        raise ValueError("REINSTALL_TORCH requires TORCH_INDEX_URL from the official PyTorch installation selector.")
    if torch_index_url:
        parsed = urlparse(torch_index_url)
        if parsed.scheme != "https" or parsed.netloc != "download.pytorch.org" or not parsed.path.startswith("/whl/"):
            raise ValueError("TORCH_INDEX_URL must be an official https://download.pytorch.org/whl/... wheel index.")
        if reinstall_torch or _version("torch") is None or _version("torchvision") is None:
            command = [sys.executable, "-m", "pip", "install", "torch==2.7.0", "torchvision==0.22.0",
                       "--index-url", torch_index_url]
            if reinstall_torch:
                command.append("--upgrade")
            subprocess.check_call(command)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(package_root / "requirements.txt")])
    importlib.invalidate_caches()
    changed = [name for name, old in loaded.items() if old != _version(name)]
    if reinstall_torch:
        changed.extend(name for name in ["torch", "torchvision"] if name in loaded)
    if changed:
        raise RuntimeError(
            "Installation finished, but this kernel already loaded changed packages: " + ", ".join(sorted(set(changed))) +
            ". Choose Kernel > Restart Kernel, then rerun from the first cell. No data or training has started."
        )
    print("Dependencies are installed in this notebook's Python environment.", flush=True)
    return {"installed": True, "python_executable": sys.executable}


def resolve_device(requested="auto", torch_module=None):
    if torch_module is None:
        import torch as torch_module
    value = str(requested).strip().lower()
    if value == "auto":
        return "0" if torch_module.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    index = value.removeprefix("cuda:")
    if not index.isdigit():
        raise ValueError("Choose device=auto, cpu, or one CUDA device such as 0. Multi-GPU training is not enabled in this notebook.")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable in this kernel. Install a matching PyTorch/CUDA build or choose cpu explicitly.")
    if int(index) >= torch_module.cuda.device_count():
        raise ValueError(f"CUDA device {index} does not exist in this process; {torch_module.cuda.device_count()} device(s) are visible.")
    return str(int(index))


def check_notebook_runtime(requested_device="auto"):
    """Exercise actual tensor and torchvision NMS operators without downloading a model."""
    try:
        import torch
        import torchvision
        import ultralytics
        from pycocotools.coco import COCO  # noqa: F401
        from .classic_yolov5 import verify_native_source
        native_source = verify_native_source()
        if ultralytics.__version__ != "8.4.183":
            raise RuntimeError(f"Expected ultralytics 8.4.183; found {ultralytics.__version__}. Rerun the installation cell.")
        device = resolve_device(requested_device, torch)
        target = "cpu" if device == "cpu" else f"cuda:{device}"
        with torch.no_grad():
            tensor = torch.ones((2, 2), device=target)
            if float((tensor @ tensor).sum().item()) != 8.:
                raise RuntimeError("The tensor calculation check failed.")
            boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device=target)
            scores = torch.tensor([.9, .8], device=target)
            kept = torchvision.ops.nms(boxes, scores, .5)
            if kept.cpu().tolist() != [0]:
                raise RuntimeError("The torchvision NMS operator check failed.")
        from .core import environment
        result = {**environment(), "selected_device": device,
                  "selected_device_name": "CPU" if device == "cpu" else torch.cuda.get_device_name(int(device)),
                  "tensor_operation_test": "passed", "torchvision_NMS_test": "passed",
                  "classic_YOLOv5_source": native_source,
                  "model_training_test": "not_run"}
        print(f"Selected device: {result['selected_device_name']} ({device})", flush=True)
        if device == "cpu":
            print("CPU mode selected. Full detector training can be slow; CUDA requires a compatible PyTorch build and driver.", flush=True)
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Notebook runtime check failed in {sys.executable}: {exc}\n"
            "Select the Python (UAV Review) kernel. For Torch/DLL/NMS failures, install torch and torchvision together "
            "using https://pytorch.org/get-started/locally/ in that environment, restart the kernel and rerun setup. "
            "See ANACONDA_START_HERE.md."
        ) from exc


def run_notebook_command(package_root, arguments, log_file=None):
    """Run the CLI in the kernel's environment, stream progress, stop the child on interruption."""
    package_root = Path(package_root).expanduser().resolve()
    command = [sys.executable, "-u", "-m", "uav_review", *map(str, arguments)]
    child_env = dict(os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONPATH"] = str(package_root) + (os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
    # Headless child processes save the report figures; Jupyter displays them later.
    child_env["MPLBACKEND"] = "Agg"
    log = None
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open("a", encoding="utf-8")
    process = None
    try:
        print("Running:", subprocess.list2cmdline(command), flush=True)
        if log:
            log.write("\nRunning: " + subprocess.list2cmdline(command) + "\n")
            log.flush()
        process = subprocess.Popen(command, cwd=package_root, env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        with process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                if log:
                    log.write(line)
                    log.flush()
        code = process.wait()
        if code:
            raise RuntimeError(f"Experiment command exited with code {code}. Review {log_file or 'the output above'} before continuing.")
        return code
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log:
            log.close()


def prepare_notebook_protocol(manifest, sessions, output, **options):
    """Reuse an existing protocol only after checking inputs, settings and frozen paths."""
    from .core import read_json
    from .engine import check_protocol
    from .protocol import create_protocols
    output = Path(output).expanduser().resolve()
    if not output.exists() or not any(output.iterdir()):
        return create_protocols(manifest, sessions, output, **options)
    with tempfile.TemporaryDirectory(prefix="uav_protocol_check_") as tmp:
        candidate = Path(tmp) / "protocol"
        create_protocols(manifest, sessions, candidate, **options)
        desired = sorted(p.name for p in candidate.glob("fold_*"))
        existing = sorted(p.name for p in output.glob("fold_*"))
        if desired != existing:
            raise ValueError("Existing fold set differs. Select a new STUDY_ROOT_INPUT for the changed protocol.")
        for name in desired:
            wanted = read_json(candidate / name / "protocol.json")
            _, frozen, _ = check_protocol(output / name)
            if wanted["protocol_sha"] != frozen["protocol_sha"]:
                raise ValueError("Existing protocol differs from current data/session metadata/settings. Select a new study directory.")
    print(f"Reusing unchanged frozen protocol ({options.get('mode', 'holdout')}): {output}", flush=True)
    return output


def train_notebook_jobs(package_root, config_path, fold_path, output, model_keys=None, seed_values=None):
    """One child process per model/seed; completed jobs are verified before reuse."""
    from .core import digest_object, environment, read_json
    from .engine import check_protocol, get_trained_run
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    fold, protocol, _ = check_protocol(fold_path)
    output = Path(output).resolve()
    runs = []
    selected_models = config["models"] if model_keys is None else model_keys
    selected_seeds = config["seeds"] if seed_values is None else seed_values
    if not set(selected_models) <= set(config["models"]) or not set(selected_seeds) <= set(config["seeds"]):
        raise ValueError("Job selection must be a subset of the frozen common model and seed lists.")
    for model in selected_models:
        for seed in selected_seeds:
            expected_config = copy.deepcopy(config)
            expected_config.update(models=[model], seeds=[seed])
            job = output / f"{model}__seed_{seed}"
            attempts = sorted(p for p in job.glob("attempt_*") if p.is_dir() and p.name.removeprefix("attempt_").isdigit()) if job.exists() else []
            completed = []
            for attempt in attempts:
                record = attempt / f"{model}__seed_{seed}" / "run.json"
                if record.is_file() and read_json(record).get("status") == "TRAINED":
                    run_dir, run, _ = get_trained_run(record.parent)
                    if (run["config_sha"] != digest_object(expected_config)
                            or run["protocol_sha"] != protocol["protocol_sha"]):
                        raise ValueError(f"Completed {model}/{seed} uses different settings. Select a new study directory.")
                    current = environment()["packages"]
                    previous = run.get("environment", {}).get("packages", {})
                    for name in ["torch", "torchvision", "ultralytics", "numpy", "pycocotools"]:
                        if previous.get(name) != current.get(name):
                            raise ValueError(f"{name} changed since {model}/{seed}. Select a new study directory to avoid mixing environments.")
                    completed.append(run_dir)
            if len(completed) > 1:
                raise ValueError(f"Multiple completed attempts for {model}/{seed}; resolve duplicate runs before reporting.")
            if completed:
                print(f"Verified completed job, skipping training: {model}, seed {seed}", flush=True)
                runs.extend(completed)
                continue
            number = max([int(p.name.split("_")[-1]) for p in attempts] or [0]) + 1
            attempt = job / f"attempt_{number:03d}"
            print(f"Training {model}, seed {seed}, attempt {number}", flush=True)
            run_notebook_command(package_root,
                                 ["train", "--config", config_path, "--fold", fold,
                                  "--out", attempt, "--models", model, "--seeds", str(seed)],
                                 log_file=job / f"attempt_{number:03d}.log")
            run_dir, _, _ = get_trained_run(attempt / f"{model}__seed_{seed}")
            runs.append(run_dir)
    return runs


def run_notebook_measurement(package_root, command, run_path, variant="full", precision="fp32",
                             tile_size=320, overlap=.2, instance_conditions=None,
                             seconds=60, warmup=50, platform_type="workstation"):
    """Verify completed measurements on reruns; preserve partial outputs before retrying."""
    from .core import read_json, sha256
    from .engine import get_trained_run
    run_dir, run, _ = get_trained_run(run_path)
    arguments = [command, "--run", run_dir, "--variant", variant, "--precision", precision,
                 "--tile-size", str(tile_size), "--overlap", str(overlap)]
    expected = {"variant": variant, "precision": precision, "tile_size": tile_size,
                "overlap": overlap, "checkpoint_sha256": run["checkpoint_sha256"],
                "protocol_sha": run["protocol_sha"]}
    if command == "evaluate":
        output = run_dir / f"evaluation_{variant}_{precision}"
        marker = output / "test_metrics.json"
        settings_file = output / "frozen_evaluation_settings.json"
        expected["instance_conditions_sha256"] = sha256(instance_conditions) if instance_conditions else None
        if instance_conditions:
            arguments.extend(["--instance-conditions", instance_conditions])
        required = ["test_metrics.json", "test_predictions.json", "val_predictions.json", "operating_point.json",
                    "frozen_evaluation_settings.json", "validation_threshold_sweep.csv", "image_errors.csv",
                    "instance_errors.csv", "size_analysis.csv", "condition_analysis.csv", "occlusion_truncation_analysis.csv"]
    elif command == "benchmark":
        output = run_dir / f"timing_{variant}_{precision}_{platform_type}"
        marker = settings_file = output / "timing_summary.json"
        expected.update(platform_type=platform_type, requested_seconds=seconds, requested_warmup_iterations=warmup)
        arguments.extend(["--seconds", str(seconds), "--warmup", str(warmup), "--platform-type", platform_type])
        required = ["timing_summary.json", "latency_samples.csv", "GPU_sensor_samples.csv"]
    else:
        raise ValueError("Notebook measurements support evaluate or benchmark.")
    if marker.is_file() and all((output / name).is_file() for name in required):
        settings = read_json(settings_file)
        if any(settings.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Completed {output.name} uses different settings. Use a new study for changed experiments.")
        print(f"Reusing completed measurement: {output}", flush=True)
        return output
    if output.exists():
        # Old partial attempts stay outside the report's evaluation_*/timing_* selectors.
        archive = run_dir / "incomplete_outputs"
        archive.mkdir(exist_ok=True)
        number = 1
        while (archive / f"{output.name}__attempt_{number:03d}").exists():
            number += 1
        target = archive / f"{output.name}__attempt_{number:03d}"
        output.rename(target)
        print(f"Preserved partial measurement in {target}", flush=True)
    run_notebook_command(package_root, arguments, log_file=run_dir / f"{output.name}.log")
    if not marker.is_file():
        raise RuntimeError(f"Command returned without its completion file: {marker}")
    return output
