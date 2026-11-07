"""Resolve the local source datasets without machine-specific saved paths."""
from __future__ import annotations

import os
from pathlib import Path

from .core import read_json


DATASETS = {
    "thermal_v4": {
        "relative_root": "data/images/labels",
        "modality": "thermal_pseudocolor_export",
        "splits": {"train": "train", "val": "valid", "test": "test"},
        "provenance_note": "Roboflow v4 export; verify source sessions and baked-in augmentation.",
    },
    "rgb_frames": {
        "relative_root": "Dataset/DroneHumanDetection",
        "modality": "visible_rgb",
        "splits": {"train": "train", "val": "val"},
        "provenance_note": "Visible-light samples observed in the uploaded 204/12-image collection.",
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def local_path(value, base=None):
    """Expand user/environment paths; anchor relative configuration to its file."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()


def load_local_settings(config_path, overrides=None):
    config_path = local_path(config_path)
    settings = read_json(config_path)
    dataset = settings.get("dataset", "thermal_v4")
    # Relative paths in JSON belong to that JSON's directory, not the terminal's CWD.
    for key in ["project_root", "dataset_root", "study_root"]:
        settings[key] = local_path(settings[key], config_path.parent) if settings.get(key) else None
    for key, value in (overrides or {}).items():
        if value is not None:
            settings[key] = local_path(value) if key.endswith("_root") else value
    dataset = settings.get("dataset", dataset)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of {list(DATASETS)}.")
    settings["dataset"] = dataset
    settings["device"] = str(settings.get("device", "0"))
    settings["study_root"] = settings["study_root"] or config_path.parent / "experiments" / f"{dataset}_revision_v1"
    if not settings["project_root"] and not settings["dataset_root"]:
        raise ValueError(
            f"Set project_root in {config_path}, or pass --project-root to run_local.py. "
            'Windows JSON example: "project_root": "D:/Research/DroneHumanProject". '
            "For a standalone dataset, set dataset_root instead."
        )
    return settings


def resolve_dataset_location(project_root, dataset_key, dataset_root=None):
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    project = local_path(project_root) if project_root else None
    if project is not None and not project.is_dir():
        raise FileNotFoundError(f"Local project folder does not exist: {project}")
    if dataset_root:
        dataset = local_path(dataset_root)
        if not dataset.is_dir():
            raise FileNotFoundError(f"Local dataset folder does not exist: {dataset}")
        return project, dataset
    if project is None:
        raise ValueError("Provide the full project_root or an explicit dataset_root.")
    relative = DATASETS[dataset_key]["relative_root"]
    direct = project / relative
    if direct.is_dir():
        return project, direct.resolve()
    # A downloaded ZIP or Drive folder may add one containing directory.
    matches = [(child.resolve(), (child / relative).resolve())
               for child in project.iterdir() if child.is_dir() and (child / relative).is_dir()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = "\n".join(str(root) for root, _ in matches)
        raise ValueError(f"Multiple {dataset_key} project copies found. Set project_root to one explicitly:\n{choices}")
    raise FileNotFoundError(
        f"Cannot locate {dataset_key}. Expected: {direct}\n"
        "Keep the downloaded project structure, or use dataset_root / --dataset-root "
        "for the folder containing train, validation and (if applicable) test. "
        "The code will not substitute the other modality or use old custom.yaml paths."
    )


def split_files(dataset_root, directory):
    """Return image files and matching annotations, including .JPG/.TXT exports."""
    split = Path(dataset_root) / directory
    images_dir, labels_dir = split / "images", split / "labels"
    missing = [str(p) for p in [images_dir, labels_dir] if not p.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    # Case collisions cannot be copied reliably between Windows and Linux.
    for kind, paths in [("image", images), ("annotation", labels)]:
        stems = {}
        for path in paths:
            key = path.stem.casefold()
            if key in stems:
                raise ValueError(f"Ambiguous {kind} stem in {split}: {stems[key].name}, {path.name}")
            stems[key] = path
    return images, {p.stem.casefold(): p for p in labels}


def preflight_dataset(project_root, dataset_key, dataset_root=None):
    """Check local layout and file pairing before an audit or any training."""
    project, dataset = resolve_dataset_location(project_root, dataset_key, dataset_root)
    spec = DATASETS[dataset_key]
    counts, errors = {}, []
    for split, directory in spec["splits"].items():
        try:
            images, labels = split_files(dataset, directory)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            continue
        image_stems = {p.stem.casefold() for p in images}
        missing = [p.name for p in images if p.stem.casefold() not in labels]
        orphans = [p.name for key, p in labels.items() if key not in image_stems]
        counts[split] = {"images": len(images), "labels": len(labels),
                         "missing_labels": len(missing), "orphan_labels": len(orphans)}
        if not images:
            errors.append(f"No supported images in {dataset / directory / 'images'}")
        if missing:
            errors.append(f"{directory}: {len(missing)} images lack annotation files, e.g. {missing[:3]}")
        if orphans:
            errors.append(f"{directory}: {len(orphans)} annotations lack matching images, e.g. {orphans[:3]}")
    if errors:
        raise ValueError("Local dataset check failed:\n" + "\n".join(errors) +
                         "\nDownload/extract all files (not shortcuts or CSV inventories), then rerun the check. "
                         "An empty .txt file is valid for a confirmed image with no people.")
    return {"dataset": dataset_key, "declared_modality": spec["modality"],
            "project_root": project.as_posix() if project else None,
            "dataset_root": dataset.as_posix(), "counts": counts,
            "note": "Paths and file pairing checked; annotation contents, source sessions and modality provenance still require audit/review."}


def validate_manifest_paths(rows):
    missing = []
    for row in rows:
        for field in ["image_path", "label_path"]:
            path = Path(row[field])
            if not path.is_absolute() or not path.is_file():
                missing.append(f"{row['image_id']}: {field}={path}")
    if missing:
        raise FileNotFoundError(
            "Audited dataset files cannot be found on this machine:\n" + "\n".join(missing[:5]) +
            "\nRerun the audit and split on this computer after setting its local project_root. "
            "Do not reuse path manifests or split lists generated on another computer."
        )
