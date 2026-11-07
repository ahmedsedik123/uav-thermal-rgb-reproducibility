"""Scientific data safeguards and orchestration checks; fixtures are unit tests only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from uav_review.core import read_csv, read_json, sha256, write_csv, write_json
from uav_review.engine import MODELS, check_protocol
from uav_review.fusion import PAIR_FIELDS, fixed_blend, inspect_pairs
from uav_review.modality_study import prepare_study, _shared_partition_id

ROOT = Path(__file__).resolve().parents[1]


def fixture_root(root, count=12):
    config = read_json(ROOT / "multimodal_config.json")
    config["study_name"] = "unit_test_only"
    write_json(root / "multimodal_config.json", config)
    (root / "configs").mkdir(exist_ok=True)
    write_json(root / "configs/primary.json", read_json(ROOT / "configs/primary.json"))
    for modality in ["rgb", "thermal"]:
        for split in (["train", "val"] if modality == "rgb" else ["train", "valid", "test"]):
            for sub in ["images", "labels"]:
                (root / "modalities" / modality / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            split = "train" if i < count - 4 else ("val" if modality == "rgb" else "valid")
            if modality == "thermal" and i >= count - 2: split = "test"
            folder = root / "modalities" / modality / split
            # Distinct sensor fixture colors and content; never delivered as a real dataset.
            color = (20+i*5, 60, 180) if modality == "rgb" else (180, 30+i*5, 50)
            Image.new("RGB", (48, 48), color).save(folder / "images" / f"unit_case_{i}.png")
            (folder / "labels" / f"unit_case_{i}.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    pairfile = root / "modalities/fusion/pairs_VERIFIED.csv"
    write_csv(pairfile, [], PAIR_FIELDS)
    return config


def write_fixture_pairs(root, count=12):
    rows = []
    for i in range(count):
        rgb_split = "train" if i < count - 4 else "val"
        thermal_split = "train" if i < count - 4 else ("valid" if i < count - 2 else "test")
        rows.append({"pair_id": f"unit_only_{i}",
            "rgb_image": f"modalities/rgb/{rgb_split}/images/unit_case_{i}.png",
            "thermal_image": f"modalities/thermal/{thermal_split}/images/unit_case_{i}.png",
            "label_path": f"modalities/rgb/{rgb_split}/labels/unit_case_{i}.txt",
            "source_group": f"unit_group_{i}", "paired_verified": "true", "alignment_verified": "true",
            "labels_verified_for_both": "true", "session_id": "", "provenance_verified": "false"})
    write_csv(root / "modalities/fusion/pairs_VERIFIED.csv", rows, PAIR_FIELDS)
    return rows


class ModalityTests(unittest.TestCase):
    def test_same_ten_distinct_models_and_three_seeds(self):
        config = read_json(ROOT / "multimodal_config.json")
        self.assertEqual(set(config["models"]), set(MODELS))
        self.assertEqual(len(config["models"]), 10)
        self.assertEqual(config["seeds"], [17, 42, 93])
        self.assertEqual(config["modalities"], ["thermal", "rgb", "fusion"])
        self.assertEqual(read_json(ROOT / "configs/primary.json")["training"]["epochs"], 100)

    def test_unpaired_runs_two_modalities_and_does_not_fabricate_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root)
            plan = prepare_study(root)
            self.assertEqual(set(plan["cohorts"]), {"thermal", "rgb"})
            self.assertEqual(plan["scheduled_training_jobs"], 60)
            self.assertFalse(plan["source_sessions_verified"])
            self.assertEqual(plan["modality_status"][-1]["status"], "AWAITING_VERIFIED_PAIRS")
            for m, cohort in plan["cohorts"].items():
                _, p, rows = check_protocol(cohort["folds"][0])
                self.assertEqual(len(rows), 12)
                self.assertFalse(p["source_sessions_verified"])
            again = prepare_study(root)
            self.assertEqual(again["input_signature"], plan["input_signature"])
            self.assertFalse(list((root / "modalities/fusion").rglob("*.png")))

    def test_explicit_paired_design_requires_real_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root)
            with self.assertRaisesRegex(ValueError, "actual registered pairs"):
                prepare_study(root, {"design": "paired"})

    def test_paired_modalities_share_cases_labels_and_exact_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root); pairs = write_fixture_pairs(root)
            originals = {row[f]: sha256(root / row[f]) for row in pairs for f in ["rgb_image", "thermal_image", "label_path"]}
            plan = prepare_study(root)
            self.assertEqual(set(plan["cohorts"]), {"thermal", "rgb", "fusion"})
            self.assertEqual(plan["scheduled_training_jobs"], 90)
            partitions, ids, labels = [], [], []
            for m in ["rgb", "thermal", "fusion"]:
                _, protocol, rows = check_protocol(plan["cohorts"][m]["folds"][0])
                partitions.append(_shared_partition_id(rows))
                ids.append({r["image_id"] for r in rows})
                labels.append({r["image_id"]: Path(r["label_path"]).read_text() for r in rows})
                self.assertTrue(protocol["paired_sensors_verified"])
                self.assertFalse(protocol["source_sessions_verified"])
            self.assertEqual(len(set(partitions)), 1)
            self.assertEqual(ids[0], ids[1]); self.assertEqual(ids[1], ids[2])
            self.assertEqual(labels[0], labels[1]); self.assertEqual(labels[1], labels[2])
            for row in pairs:
                for f in ["rgb_image", "thermal_image", "label_path"]:
                    self.assertEqual(sha256(root / row[f]), originals[row[f]])
            again = prepare_study(root)
            self.assertEqual(plan["input_signature"], again["input_signature"])

    def test_unconfirmed_alignment_and_label_transfer_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root); rows = write_fixture_pairs(root)
            for field in ["paired_verified", "alignment_verified", "labels_verified_for_both"]:
                changed = copy.deepcopy(rows); changed[0][field] = "false"
                write_csv(root / "modalities/fusion/pairs_VERIFIED.csv", changed, PAIR_FIELDS)
                with self.assertRaisesRegex(ValueError, field):
                    inspect_pairs(root, root / "modalities/fusion/pairs_VERIFIED.csv")

    def test_shared_component_and_same_image_fake_pair_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root); rows = write_fixture_pairs(root)
            rows[1]["rgb_image"] = rows[0]["rgb_image"]
            write_csv(root / "modalities/fusion/pairs_VERIFIED.csv", rows, PAIR_FIELDS)
            with self.assertRaisesRegex(ValueError, "exactly one pair"):
                inspect_pairs(root, root / "modalities/fusion/pairs_VERIFIED.csv")
            rows = write_fixture_pairs(root); rows[0]["thermal_image"] = rows[0]["rgb_image"]
            write_csv(root / "modalities/fusion/pairs_VERIFIED.csv", rows, PAIR_FIELDS)
            with self.assertRaisesRegex(ValueError, "same image"):
                inspect_pairs(root, root / "modalities/fusion/pairs_VERIFIED.csv")

    def test_changed_data_cannot_reuse_existing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root); prepare_study(root)
            label = next((root / "modalities/rgb").rglob("*.txt"))
            label.write_text("0 0.5 0.5 0.5 0.5\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_study(root)

    def test_fusion_formula_is_exact_and_rejects_size_mismatch(self):
        rgb = Image.new("RGB", (2, 2), (100, 200, 50))
        thermal = Image.new("RGB", (2, 2), (200, 0, 100))
        fused = np.asarray(fixed_blend(rgb, thermal, .5))
        self.assertEqual(fused[0, 0].tolist(), [150, 100, 75])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            fixed_blend(rgb, Image.new("RGB", (3, 2)), .5)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            fixed_blend(rgb, thermal, 1)

    def test_verified_session_mode_still_requires_acquisition_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_root(root)
            with self.assertRaises((FileNotFoundError, ValueError)):
                prepare_study(root, {"evaluation_mode": "verified_sessions"})


if __name__ == "__main__":
    unittest.main()
