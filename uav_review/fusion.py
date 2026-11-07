"""Registered RGB/thermal input fusion; no automatic or fabricated pairing.

The baseline retains each model's three-channel architecture. It blends aligned
visible RGB and pseudocolor thermal values with a predeclared fixed weight.
It is not radiometric fusion, learned feature fusion or sensor calibration.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .audit import difference_hash, family_key, parse_labels, pixel_fingerprint
from .core import digest_object, read_csv, sha256, write_csv, write_json


PAIR_FIELDS = ["pair_id", "rgb_image", "thermal_image", "label_path", "source_group",
               "paired_verified", "alignment_verified", "labels_verified_for_both",
               "session_id", "provenance_verified"]


def confirmed(value):
    return str(value).strip().lower() in {"true", "yes", "1"}


def fixed_blend(rgb, thermal, rgb_weight):
    """Return rounded, clipped uint8 RGB values; inputs must already be aligned."""
    weight = float(rgb_weight)
    if not math.isfinite(weight) or not 0 < weight < 1:
        raise ValueError("Fusion rgb_weight must be strictly between 0 and 1.")
    a = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    b = np.asarray(thermal.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("RGB and thermal inputs must already have the same aligned pixel dimensions.")
    values = np.clip(np.floor(weight * a + (1 - weight) * b + 0.5), 0, 255).astype(np.uint8)
    return Image.fromarray(values)


def _resolve(root, value):
    if not value or "\n" in value or "\r" in value:
        raise ValueError("Every pair needs an actual local file path, without line breaks.")
    p = Path(value).expanduser()
    p = (root / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Paired input does not exist: {p}")
    return p


def inspect_pairs(package_root, pairs_csv, rgb_weight=0.5):
    """Validate all confirmations, images and labels before generating any data.

    Equal image sizes alone are not evidence of correspondence or registration.
    The three explicit confirmations must come from the acquisition/annotation
    records and visual registration review. Source groups are not flight IDs.
    """
    root = Path(package_root).resolve()
    pairs_csv = Path(pairs_csv)
    if not pairs_csv.is_file():
        return []
    raw = read_csv(pairs_csv)
    if not raw:
        return []
    missing = set(PAIR_FIELDS) - set(raw[0])
    if missing:
        raise ValueError(f"Pair manifest lacks columns: {sorted(missing)}")
    ids, used_rgb, used_thermal = set(), set(), set()
    entries = []
    for line, row in enumerate(raw, 2):
        pid = row["pair_id"].strip()
        if not pid or any(c in pid for c in "\n\r") or pid in ids:
            raise ValueError(f"Pair manifest line {line}: nonempty unique pair_id required.")
        ids.add(pid)
        for field in ["paired_verified", "alignment_verified", "labels_verified_for_both"]:
            if not confirmed(row[field]):
                raise ValueError(f"Pair {pid}: {field}=true must be confirmed from real paired inputs.")
        if not row["source_group"].strip():
            raise ValueError(f"Pair {pid}: specify the source scene/augmentation group; do not invent flights.")
        rp = _resolve(root, row["rgb_image"])
        tp = _resolve(root, row["thermal_image"])
        lp = _resolve(root, row["label_path"])
        if rp == tp:
            raise ValueError(f"Pair {pid}: the same image cannot be both sensor inputs.")
        if rp in used_rgb or tp in used_thermal:
            raise ValueError(f"Pair {pid}: each RGB and thermal file must occur in exactly one pair.")
        used_rgb.add(rp); used_thermal.add(tp)
        boxes, errors = parse_labels(lp)
        if errors:
            raise ValueError(f"Pair {pid}: invalid common-coordinate labels: {errors}")
        with Image.open(rp) as rgb, Image.open(tp) as thermal:
            rgb.load(); thermal.load()
            if rgb.size != thermal.size:
                raise ValueError(f"Pair {pid}: register the images before fusion; sizes {rgb.size} and {thermal.size} differ.")
            if min(rgb.size) < 1:
                raise ValueError(f"Pair {pid}: invalid image dimensions.")
            fused = fixed_blend(rgb, thermal, rgb_weight)
            pixels = {"rgb": pixel_fingerprint(rgb), "thermal": pixel_fingerprint(thermal),
                      "fusion": pixel_fingerprint(fused)}
            if pixels["rgb"] == pixels["thermal"]:
                raise ValueError(f"Pair {pid}: identical RGB/thermal pixels do not establish distinct sensor inputs.")
            width, height = rgb.size
        entry = {**row, "pair_id": pid, "rgb_path": str(rp), "thermal_path": str(tp),
                 "common_label_path": str(lp), "boxes": boxes, "width": width, "height": height,
                 "rgb_sha256": sha256(rp), "thermal_sha256": sha256(tp), "label_sha256": sha256(lp),
                 "pixels": pixels, "family_rgb": family_key(rp.name), "family_thermal": family_key(tp.name)}
        entries.append(entry)
    entries.sort(key=lambda e: e["pair_id"])
    # Union every known relationship across ALL three input variants before
    # drawing the shared split. Reused components/augmentations stay together.
    parent = {e["pair_id"]: e["pair_id"] for e in entries}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    seen = {}
    for e in entries:
        keys = [("source", e["source_group"].strip()), ("rgb_family", e["family_rgb"]),
                ("thermal_family", e["family_thermal"])]
        keys.extend(("pixel", value) for value in e["pixels"].values())
        for key in keys:
            if key in seen: union(e["pair_id"], seen[key])
            else: seen[key] = e["pair_id"]
    for e in entries:
        e["shared_group"] = "paired_group:" + hashlib.sha256(find(e["pair_id"]).encode()).hexdigest()[:20]
    if len({e["shared_group"] for e in entries}) < 3:
        raise ValueError("Paired data need at least three usable groups for a shared train/validation/test partition.")
    return entries


def materialize_paired_sources(entries, output, rgb_weight):
    """Write three matched datasets using the same boxes and case identities."""
    import shutil
    start = time.perf_counter()
    output = Path(output).resolve()
    manifests = {m: [] for m in ["rgb", "thermal", "fusion"]}
    identity = []
    for e in entries:
        # Hashed filenames avoid Windows reserved names and case collisions.
        stem = "pair_" + hashlib.sha256(e["pair_id"].encode()).hexdigest()[:24]
        with Image.open(e["rgb_path"]) as rgb, Image.open(e["thermal_path"]) as thermal:
            fused = fixed_blend(rgb, thermal, rgb_weight)
            for modality in manifests:
                src = Path(e["rgb_path"] if modality == "rgb" else e["thermal_path"])
                suffix = ".png" if modality == "fusion" else src.suffix.lower()
                image = output / modality / "images" / (stem + suffix)
                label = output / modality / "labels" / (stem + ".txt")
                image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
                if modality == "fusion": fused.save(image, format="PNG")
                elif not image.exists() or sha256(image) != sha256(src): shutil.copy2(src, image)
                label.write_text("".join(" ".join(f"{v:.12g}" for v in b) + "\n" for b in e["boxes"]), encoding="utf-8")
                with Image.open(image) as actual:
                    dhash = difference_hash(actual)
                    pixel_hash = pixel_fingerprint(actual)
                    mode = actual.mode
                manifests[modality].append({
                    "image_id": "paired/" + e["pair_id"], "pair_id": e["pair_id"],
                    "dataset": "paired_" + modality,
                    "declared_modality": {"rgb": "visible_rgb", "thermal": "thermal_pseudocolor_export",
                                          "fusion": "registered_RGB_thermal_fixed_input_blend"}[modality],
                    "original_split": "paired_pool", "image_path": image.as_posix(), "label_path": label.as_posix(),
                    "width": e["width"], "height": e["height"], "n_instances": len(e["boxes"]),
                    "source_image_path": src.as_posix() if modality != "fusion" else "derived_from_verified_pair",
                    "source_label_path": e["common_label_path"], "source_label_sha256": e["label_sha256"],
                    "family_key": e["shared_group"], "sha256": sha256(image), "label_sha256": sha256(label),
                    "label_valid": True, "pixel_sha256": pixel_hash, "dhash": dhash, "image_mode": mode,
                    "paired_verified": "true", "alignment_verified": "true",
                    "session_id": e["session_id"].strip(), "provenance_verified": e["provenance_verified"],
                })
        identity.append({k: e[k] for k in ["pair_id", "source_group", "shared_group", "rgb_sha256", "thermal_sha256", "label_sha256"]})
    for modality, rows in manifests.items():
        write_csv(output / modality / "image_manifest.csv", rows)
    write_csv(output / "paired_source_identity.csv", identity)
    write_json(output / "fusion_definition.json", {
        "method": "registered_three_channel_input_blend", "rgb_weight": float(rgb_weight),
        "formula": "floor(alpha * RGB + (1-alpha) * thermal_pseudocolor + 0.5), clipped to [0,255]",
        "alignment": "author-confirmed prealigned common pixel plane; no automatic calibration",
        "ground_truth": "common-coordinate labels verified for both sensors",
        "learned_feature_fusion": False, "radiometric_temperature_fusion": False,
        "paired_case_identity_sha256": digest_object(identity), "paired_cases": len(entries),
        "materialization_wall_seconds_including_IO": time.perf_counter() - start,
        "materialization_timing_scope": "offline RGB/thermal copy, blend, label writing and file fingerprints; excludes acquisition, registration review and model execution"})
    return manifests
