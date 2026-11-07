from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_object(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def environment():
    versions = {}
    for name in ["torch", "torchvision", "ultralytics", "numpy", "Pillow",
                 "pycocotools", "matplotlib", "opencv-python", "motmetrics", "pandas", "scipy", "seaborn", "ultralytics-thop"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    information = {"python": sys.version, "python_executable": sys.executable,
                   "python_prefix": sys.prefix, "conda_environment": (Path(sys.prefix) / "conda-meta").is_dir(),
                   "CONDA_DEFAULT_ENV_variable": os.environ.get("CONDA_DEFAULT_ENV"),
                   "platform": platform.platform(), "packages": versions}
    try:
        import torch
        information["CUDA_runtime"] = torch.version.cuda
        information["cuDNN_version"] = torch.backends.cudnn.version()
        information["CUDA_available"] = torch.cuda.is_available()
    except (ImportError, OSError, RuntimeError) as exc:
        information.update(CUDA_runtime=None, cuDNN_version=None, CUDA_available=False)
        information["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return information


def require_empty(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output already exists: {path}. Choose a fresh output directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path
