"""Data and legacy-run inspection; no neural-network dependencies required."""
from __future__ import annotations

import collections
import math
import re
import shutil
from pathlib import Path

from PIL import Image

from .core import read_csv, sha256, write_csv, write_json
from .paths import DATASETS, IMAGE_SUFFIXES, local_path, preflight_dataset, split_files


def family_key(filename):
    stem = Path(filename).stem.lower()
    return re.sub(r"\.rf\.[^.]+$", "", stem)


def parse_labels(path):
    """Parse boxes and convert valid YOLO polygons to their axis-aligned bounds.

    Mixed box/polygon files are normalized explicitly, never passed to a backend
    that might reinterpret a five-field box as two polygon vertices.
    """
    boxes, problems = [], []
    if not Path(path).is_file():
        return boxes, ["missing_label_file"]
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("non-finite coordinate")
            if len(values) == 5:
                cls, x, y, w, h = values
            elif len(values) >= 7 and (len(values)-1) % 2 == 0:
                cls = values[0]
                xs, ys = values[1::2], values[2::2]
                if min(xs+ys) < 0 or max(xs+ys) > 1:
                    raise ValueError("polygon coordinates outside normalized range")
                x, y = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                w, h = max(xs)-min(xs), max(ys)-min(ys)
            else:
                raise ValueError("expected a five-field box or class plus polygon vertices")
            if cls != 0:
                raise ValueError("expected person class 0")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("coordinates outside normalized range")
            if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                raise ValueError("box crosses image boundary")
            boxes.append([int(cls), x, y, w, h])
        except ValueError as exc:
            problems.append(f"line {lineno}: {exc}")
    return boxes, problems


def pixel_fingerprint(image):
    rgb = image.convert("RGB")
    import hashlib
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def difference_hash(image):
    gray = image.convert("L").resize((9, 8))
    values = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
    bits = [values[r*9+c] > values[r*9+c+1] for r in range(8) for c in range(8)]
    return f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"


def audit_dataset(project_root, dataset_key, output, dataset_root=None):
    """Audit the FULL local project. The output is not a new train/test split."""
    checked = preflight_dataset(project_root, dataset_key, dataset_root)
    spec = DATASETS[dataset_key]
    dataset_root = Path(checked["dataset_root"])
    output = local_path(output)
    if (output / "normalized_dataset").resolve() == dataset_root.resolve():
        raise ValueError("Choose an audit output whose normalized_dataset folder differs from the source dataset.")
    write_json(output / "local_dataset_paths.json", checked)
    rows, issues, summary, instance_template = [], [], {}, []
    for split, directory in spec["splits"].items():
        images, label_files = split_files(dataset_root, directory)
        humans = 0
        for image_path in images:
            label_path = label_files[image_path.stem.casefold()]
            boxes, errors = parse_labels(label_path)
            for error in errors:
                issues.append({"split": split, "severity": "error", "file": str(label_path), "issue": error})
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    pixel_hash = pixel_fingerprint(image)
                    dhash = difference_hash(image)
            except Exception as exc:
                issues.append({"split": split, "severity": "error", "file": str(image_path), "issue": f"unreadable image: {exc}"})
                continue
            humans += len(boxes)
            normalized_images = output / "normalized_dataset" / directory / "images"
            normalized_labels = output / "normalized_dataset" / directory / "labels"
            normalized_images.mkdir(parents=True, exist_ok=True)
            normalized_labels.mkdir(parents=True, exist_ok=True)
            clean_image = normalized_images / image_path.name
            # The normalized label name must match the image exactly for YOLO's loader.
            clean_label = normalized_labels / (image_path.stem + ".txt")
            source_hash = sha256(image_path)
            if not clean_image.exists() or clean_image.samefile(image_path) or sha256(clean_image) != source_hash:
                clean_image.unlink(missing_ok=True)
                # Backends may repair JPEG files in place; source pixels must stay isolated.
                shutil.copy2(image_path, clean_image)
            if not errors:
                clean_label.write_text("".join(" ".join(f"{v:.12g}" for v in box)+"\n" for box in boxes))
            else:
                clean_label.unlink(missing_ok=True)
            raw_lines = [line.split() for line in label_path.read_text().splitlines() if line.strip()] if label_path.exists() else []
            box_rows = sum(len(line) == 5 for line in raw_lines)
            polygon_rows = sum(len(line) > 5 for line in raw_lines)
            image_id = f"{dataset_key}/{directory}/{image_path.name}"
            instance_template.extend({"image_id": image_id, "gt_index": i} for i in range(len(boxes)))
            rows.append({
                "image_id": image_id,
                "dataset": dataset_key, "declared_modality": spec["modality"],
                "original_split": split, "image_path": clean_image.resolve().as_posix(),
                "label_path": clean_label.resolve().as_posix(), "width": width, "height": height,
                "source_image_path": image_path.as_posix(), "source_label_path": label_path.as_posix(),
                "source_label_sha256": sha256(label_path) if label_path.is_file() else "",
                "raw_box_rows": box_rows, "raw_polygon_rows": polygon_rows,
                "image_mode": mode, "n_instances": len(boxes),
                "family_key": family_key(image_path.name), "sha256": source_hash,
                "label_sha256": sha256(clean_label) if clean_label.is_file() else "",
                "label_valid": not bool(errors),
                "pixel_sha256": pixel_hash, "dhash": dhash,
            })
        summary[split] = {"images": len(images), "human_instances": humans}
    exact = collections.defaultdict(list)
    families = collections.defaultdict(list)
    for row in rows:
        exact[row["pixel_sha256"]].append(row)
        families[row["family_key"]].append(row)
    for kind, groups in [("identical_pixels", exact), ("same_source_image_family", families)]:
        for key, items in groups.items():
            if len({r["original_split"] for r in items}) > 1:
                issues.append({"severity": "error", "issue": f"cross-split {kind}",
                               "file": ";".join(r["image_id"] for r in items)})
    # Perceptual candidates require visual review; they are never declared duplicates automatically.
    candidates = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a["original_split"] == b["original_split"]:
                continue
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= 4:
                candidates.append({"image_a": a["image_id"], "image_b": b["image_id"],
                                   "dhash_distance": distance, "decision": "NEEDS_VISUAL_REVIEW"})
    write_csv(output / "image_manifest.csv", rows)
    write_csv(output / "issues.csv", issues, ["severity", "split", "file", "issue"])
    write_csv(output / "near_duplicate_candidates.csv", candidates,
              ["image_a", "image_b", "dhash_distance", "decision"])
    session_fields = ["image_id", "session_id", "source_video", "source_frame",
                      "pair_id", "thermal_clutter", "motion_blur", "provenance_verified"]
    write_csv(output / "session_map_TEMPLATE.csv",
              ({"image_id": r["image_id"]} for r in rows), session_fields)
    write_csv(output / "instance_conditions_TEMPLATE.csv", instance_template,
              ["image_id", "gt_index", "occlusion", "truncation"])
    write_json(output / "dataset_summary.json", {
        "dataset": dataset_key, "declared_modality": spec["modality"],
        "local_dataset_root": dataset_root.as_posix(),
        "provenance_note": spec["provenance_note"], "counts": summary,
        "error_count": sum(i["severity"] == "error" for i in issues),
        "near_duplicate_candidates": len(candidates),
        "session_ids_verified": False,
        "annotation_normalization": "Valid polygons converted to axis-aligned boxes in a separate dataset copy; source files retained.",
        "note": "Image families are not flight sessions. No radiometric or paired-sensor claim is inferred.",
    })
    return rows


def legacy_run_audit(project_root, output):
    import yaml
    rows = []
    fingerprints = collections.defaultdict(list)
    for path in sorted(Path(project_root).rglob("results.csv")):
        raw = read_csv(path)
        if not raw:
            continue
        records = [{k.strip(): v for k, v in row.items()} for row in raw]
        args_path = path.parent / "args.yaml"
        args = yaml.safe_load(args_path.read_text()) if args_path.exists() else {}
        args = args or {}
        epoch_key = next((k for k in records[0] if k.lower() == "epoch"), None)
        epochs = [float(r[epoch_key]) for r in records] if epoch_key else []
        boundaries = [0] + [i for i in range(1, len(epochs)) if epochs[i] <= epochs[i-1]] + [len(records)]
        if len(boundaries) == 2:
            segment_count = 1
        else:
            segment_count = len(boundaries)-1
        ap_key = next((k for k in records[0] if "mAP50-95" in k or "mAP_0.5:0.95" in k), None)
        fingerprint = sha256(path)
        fingerprints[fingerprint].append(str(path))
        for segment, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            values = records[start:stop]
            if not values:
                continue
            best = max(values, key=lambda r: float(r[ap_key])) if ap_key else {}
            rows.append({
                "results_csv": str(path), "csv_sha256": fingerprint,
                "folder_label": path.parent.name, "args_model": args.get("model", ""),
                "args_data": args.get("data", ""), "configured_epochs": args.get("epochs", ""),
                "seed": args.get("seed", ""), "patience": args.get("patience", ""),
                "resume": args.get("resume", ""), "segment": segment,
                "segment_count": segment_count, "logged_rows_in_segment": len(values),
                "best_epoch": best.get(epoch_key, "") if epoch_key else "",
                "best_map50_95": best.get(ap_key, "") if ap_key else "",
                "status": "AMBIGUOUS_CONCATENATED_LOG" if segment_count > 1 else "VERIFY_CHECKPOINT_AND_SPLIT",
                "checkpoint_exists": (path.parent / "weights/best.pt").exists(),
                "warning": "args.yaml may describe only the latest appended run; segment identity must be verified.",
            })
    write_csv(Path(output) / "legacy_run_inventory.csv", rows)
    write_json(Path(output) / "duplicate_log_files.json",
               [files for files in fingerprints.values() if len(files) > 1])
    return rows
