"""Verified session splits and explicitly exploratory image-family holdouts."""
from __future__ import annotations

import collections
import math
import random
from pathlib import Path

from .core import digest_object, read_csv, require_empty, write_csv, write_json
from .paths import local_path, validate_manifest_paths


def attach_sessions(manifest, mapping):
    mapping_by_id = {}
    for item in mapping:
        if item["image_id"] in mapping_by_id:
            raise ValueError(f"Duplicate session-map row: {item['image_id']}")
        mapping_by_id[item["image_id"]] = item
    ids = {r["image_id"] for r in manifest}
    if set(mapping_by_id) != ids:
        raise ValueError("Session map must cover exactly the audited images.")
    rows = []
    for row in manifest:
        meta = mapping_by_id[row["image_id"]]
        if str(row.get("label_valid", "")).lower() != "true":
            raise ValueError(f"Unresolved annotation error: {row['image_id']}")
        if not meta.get("session_id", "").strip() or meta.get("provenance_verified", "").lower() not in {"true", "1", "yes"}:
            raise ValueError(f"Verified session/source provenance missing: {row['image_id']}")
        # All current exports are interpreted only in their declared modality.
        rows.append({**row, **meta})
    if len({r["dataset"] for r in rows}) != 1:
        raise ValueError("A primary experiment must contain exactly one dataset.")
    return rows


def attach_exploratory_groups(manifest):
    """Use audited image families without inventing flight or video identities."""
    if not manifest or len({r["dataset"] for r in manifest}) != 1:
        raise ValueError("An exploratory experiment must contain exactly one nonempty dataset.")
    rows = []
    for row in manifest:
        if str(row.get("label_valid", "")).lower() != "true":
            raise ValueError(f"Unresolved annotation error: {row['image_id']}")
        if not row.get("family_key") or not row.get("pixel_sha256"):
            raise ValueError(f"Missing audited image-family or pixel fingerprint: {row['image_id']}")
        rows.append({**row, "source_group": "image_family:" + row["family_key"],
                     "session_id": "", "source_video": "", "provenance_verified": "false"})
    return rows


def protocol_scope(protocol):
    """Keep report/run scope explicit, including for older verified protocols."""
    exploratory = protocol["mode"] == "exploratory"
    result = {"evaluation_scope": "exploratory_image_family_holdout" if exploratory else "verified_source_session",
            "source_sessions_verified": not exploratory,
            "group_unit": "unverified_image_family_and_pixel_group" if exploratory else "verified_source_session"}
    if protocol.get("paired_sensors_verified"):
        result.update(paired_sensors_verified=True, paired_case_partition_id=protocol["paired_case_partition_id"])
    return result


def leakage_groups(rows, group_field="session_id"):
    """Union the declared groups sharing an exact image or augmentation family."""
    parents = {r[group_field]: r[group_field] for r in rows}
    if not all(parents):
        raise ValueError(f"Every image needs a nonempty {group_field}.")
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    for field in ["family_key", "pixel_sha256"]:
        seen = {}
        for row in rows:
            key = row.get(field)
            if not key:
                raise ValueError(f"Missing audited {field}: {row['image_id']}")
            if key in seen:
                union(row[group_field], seen[key])
            else:
                seen[key] = row[group_field]
    return {session: find(session) for session in parents}


def assert_no_leakage(parts, group_field="session_id"):
    if group_field not in {"session_id", "source_group"}:
        raise ValueError("Unknown grouping field")
    fields = ["image_id", group_field, "family_key", "pixel_sha256"]
    keys = list(parts)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            for field in fields:
                overlap = {r[field] for r in parts[a]} & {r[field] for r in parts[b]}
                if overlap:
                    raise ValueError(f"{field} overlap between {a} and {b}: {next(iter(overlap))}")
    if any(not rows for rows in parts.values()):
        raise ValueError("Every train/validation/test partition must contain images.")


def choose_validation(groups, fraction, seed):
    groups = list(sorted(groups))
    if len(groups) < 2:
        raise ValueError("At least two independent groups required for train/validation.")
    random.Random(seed).shuffle(groups)
    count = min(len(groups)-1, max(1, round(len(groups)*fraction)))
    return set(groups[count:]), set(groups[:count])


def create_protocols(manifest_path, sessions_path, output, mode="holdout", folds=5,
                     seed=42, test_fraction=0.15, val_fraction=0.17647058823529413):
    if mode == "exploratory":
        rows = attach_exploratory_groups(read_csv(manifest_path))
        group_field = "source_group"
    else:
        if not sessions_path or not Path(sessions_path).is_file():
            raise ValueError("Verified source-session metadata is required for holdout/grouped_cv. "
                             "For an explicitly exploratory image-family experiment, select mode='exploratory'.")
        rows = attach_sessions(read_csv(manifest_path), read_csv(sessions_path))
        group_field = "session_id"
    validate_manifest_paths(rows)
    group_map = leakage_groups(rows, group_field)
    for row in rows:
        row["leakage_group"] = group_map[row[group_field]]
    groups = sorted(set(group_map.values()))
    minimum = 3 if mode == "exploratory" else 5
    if len(groups) < minimum:
        raise ValueError(f"Fewer than {minimum} usable groups after leakage merging. Recover more source groups/data.")
    if not 0 < test_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Split fractions must lie strictly between zero and one.")
    rng = random.Random(seed)
    rng.shuffle(groups)
    if mode in {"holdout", "exploratory"}:
        ntest = max(1, round(len(groups)*test_fraction))
        tests = [set(groups[:ntest])]
    elif mode == "grouped_cv":
        if not 3 <= folds <= len(groups):
            raise ValueError("Grouped CV needs between three and the available number of folds.")
        tests = [set(groups[i::folds]) for i in range(folds)]
    else:
        raise ValueError("Unknown protocol mode")
    output = require_empty(local_path(output))
    for fold, test_groups in enumerate(tests):
        train_groups, val_groups = choose_validation(set(groups)-test_groups, val_fraction, seed+fold)
        parts = {name: [r for r in rows if r["leakage_group"] in selected]
                 for name, selected in [("train", train_groups), ("val", val_groups), ("test", test_groups)]}
        assert_no_leakage(parts, group_field)
        fold_dir = output / f"fold_{fold:02d}"
        fold_dir.mkdir()
        assignments = []
        for split, selected in parts.items():
            for row in selected:
                assignments.append({**row, "split": split})
            (fold_dir / f"{split}.txt").write_text(
                "\n".join(r["image_path"] for r in selected)+"\n", encoding="utf-8")
        write_csv(fold_dir / "assignments.csv", assignments)
        # JSON is a valid YAML subset, but use a .yaml name for Ultralytics.
        dataset_yaml = {name: (fold_dir / f"{name}.txt").resolve().as_posix() for name in parts}
        dataset_yaml.update({"path": fold_dir.resolve().as_posix(), "nc": 1, "names": ["Human"]})
        write_json(fold_dir / "data.yaml", dataset_yaml)
        counts = {split: {"images": len(selected), "humans": sum(int(r["n_instances"]) for r in selected),
                          "sessions": len({r["session_id"] for r in selected if r["session_id"]}),
                          "groups": len({r["leakage_group"] for r in selected})}
                  for split, selected in parts.items()}
        protocol = {"dataset": rows[0]["dataset"], "modality": rows[0]["declared_modality"],
                    "mode": mode, "fold": fold, "total_folds": len(tests),
                    "split_seed": seed, "counts": counts,
                    "source_manifest_sha": digest_object(rows), "assignments_sha": digest_object(assignments),
                    "grouping_field": group_field, **protocol_scope({"mode": mode}),
                    "note": ("EXPLORATORY: image families and identical pixels are grouped. "
                             "Temporal/scene dependencies may remain across partitions. No independent-flight, "
                             "unseen-site, paired-sensor or session-bootstrap claim is supported."
                             if mode == "exploratory" else
                             "Unseen sessions do not establish unseen-site generalization.")}
        protocol["protocol_sha"] = digest_object(protocol)
        write_json(fold_dir / "protocol.json", protocol)
    return output
