import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from uav_review.classic_yolov5 import (NATIVE_COMMIT, NATIVE_ROOT, blocked_optional_imports,
                                      classic_hyperparameters, normalize_classic_history, verify_native_source)
from uav_review.core import read_csv, write_csv
from uav_review.engine import MODELS, infer_image, load_model, validate_model_identity


ROOT = Path(__file__).resolve().parents[1]


class OriginalModelTests(unittest.TestCase):
    def test_primary_roster_covers_both_paper_lists_without_duplicate_aliases(self):
        config = json.loads((ROOT / "configs/primary.json").read_text())
        expected = {"yolov5s", "yolov5su", "yolov8n", "yolov8m", "yolov8l", "yolov10n",
                    "yolo11n", "yolo12n", "yolo26n", "rtdetr_l"}
        self.assertEqual(set(config["models"]), expected)
        self.assertEqual(len(config["models"]), 10)
        self.assertEqual(len({MODELS[key]["weights"] for key in config["models"]}), 10)
        self.assertEqual(MODELS["yolov5s"]["loader"], "CLASSIC_YOLOV5")
        self.assertEqual(MODELS["yolov5su"]["loader"], "YOLO")
        self.assertNotIn("yolov5", MODELS)

    def test_prediction_uses_the_pinned_backends_current_precision_api(self):
        captured = []
        def predict(**kwargs):
            captured.append(kwargs)
            return [types.SimpleNamespace(boxes=None, speed={})]
        model = types.SimpleNamespace(overrides={"nms": None}, predict=predict)
        infer_image(model, "fixture.jpg", "cpu", 640)
        infer_image(model, "fixture.jpg", "0", 640, half=True)
        self.assertEqual([item["quantize"] for item in captured], [32, 16])
        self.assertTrue(all("half" not in item for item in captured))

    def test_classic_model_never_passes_through_the_modern_loader(self):
        fake = types.SimpleNamespace(model=types.SimpleNamespace(yaml={"yaml_file": "yolov5s.yaml"}), overrides={})
        with patch("uav_review.classic_yolov5.ClassicYolov5", return_value=fake) as native:
            actual = load_model("yolov5s")
        self.assertIs(actual, fake)
        self.assertEqual(str(native.call_args.kwargs["weights"]), "yolov5s.pt")

    def test_model_size_mismatch_is_rejected_within_the_same_family(self):
        fake = types.SimpleNamespace(model=types.SimpleNamespace(yaml={"yaml_file": "yolov8.yaml", "scale": "l"}))
        with self.assertRaisesRegex(ValueError, "scale mismatch"):
            validate_model_identity("yolov8m", fake)

    def test_official_legacy_yolov12_yaml_name_is_accepted_without_accepting_wrong_size(self):
        fake = types.SimpleNamespace(model=types.SimpleNamespace(yaml={"yaml_file": "yolov12n.yaml"}))
        validate_model_identity("yolo12n", fake)
        fake.model.yaml["yaml_file"] = "yolov12l.yaml"
        with self.assertRaisesRegex(ValueError, "scale mismatch"):
            validate_model_identity("yolo12n", fake)

    def test_official_hyphenated_rtdetr_yaml_name_is_accepted_with_correct_variant(self):
        fake = types.SimpleNamespace(model=types.SimpleNamespace(yaml={"yaml_file": "ultralytics/models/rt-detr/rt-detr-l.yaml"}))
        validate_model_identity("rtdetr_l", fake)
        fake.model.yaml["yaml_file"] = "rt-detr-x.yaml"
        with self.assertRaisesRegex(ValueError, "scale mismatch"):
            validate_model_identity("rtdetr_l", fake)

    def test_bundled_classic_source_matches_the_pinned_upstream_files(self):
        verified = verify_native_source()
        self.assertEqual(verified["commit"], NATIVE_COMMIT)
        self.assertTrue((NATIVE_ROOT / "LICENSE").is_file())
        self.assertTrue((NATIVE_ROOT / "models/yolov5s.yaml").is_file())

    def test_classic_training_uses_the_common_budget_and_augmentation_hyperparameters(self):
        common = json.loads((ROOT / "configs/primary.json").read_text())["training"]
        hyp = classic_hyperparameters(common)
        for key in ["lr0", "lrf", "weight_decay", "warmup_epochs", "degrees", "hsv_h", "hsv_s", "hsv_v",
                    "fliplr", "mosaic", "mixup", "copy_paste"]:
            self.assertEqual(hyp[key], common[key])
        self.assertEqual(hyp["mosaic"], 0.)
        self.assertIn("obj", hyp)
        self.assertIn("anchor_t", hyp)

    def test_native_history_is_standardized_without_dropping_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "native.csv", Path(tmp) / "history.csv"
            write_csv(source, [{" epoch": "0", " metrics/mAP_0.5:0.95": ".42",
                                " metrics/mAP_0.5": ".6", " train/obj_loss": ".02"}])
            normalize_classic_history(source, target)
            row = read_csv(target)[0]
            self.assertEqual(row["epoch"], "1")
            self.assertEqual(float(row["metrics/mAP50-95(B)"]), .42)
            self.assertEqual(float(row["metrics/mAP50(B)"]), .6)
            self.assertEqual(float(row["train/obj_loss"]), .02)

    def test_optional_import_state_is_restored_after_training_failure(self):
        original = types.SimpleNamespace()
        with patch.dict(sys.modules, {"albumentations": original}):
            with self.assertRaises(RuntimeError):
                with blocked_optional_imports(["albumentations"]):
                    self.assertIsNone(sys.modules["albumentations"])
                    raise RuntimeError("interrupted test")
            self.assertIs(sys.modules["albumentations"], original)


if __name__ == "__main__":
    unittest.main()
