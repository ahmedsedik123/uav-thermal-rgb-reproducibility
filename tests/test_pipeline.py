import csv
import importlib.util
import tempfile
import unittest
import types
from unittest.mock import patch
from pathlib import Path

from PIL import Image, ImageDraw

from uav_review.audit import audit_dataset, legacy_run_audit, parse_labels
from uav_review.core import read_csv, read_json, write_csv, write_json
from uav_review.engine import check_protocol, tile_origins, load_model
from uav_review.metrics import (coco_metrics, iou, match_boxes, nms, paired_bootstrap,
                                resample_sessions, select_confidence)
from uav_review.protocol import assert_no_leakage, create_protocols


class DetectionMetricsTests(unittest.TestCase):
    def test_model_architecture_mismatch_is_rejected(self):
        fake = types.SimpleNamespace(model=types.SimpleNamespace(yaml={"yaml_file": "yolov8m.yaml"}))
        backend = types.SimpleNamespace(YOLO=lambda _: fake, RTDETR=lambda _: fake)
        with patch.dict("sys.modules", {"ultralytics": backend}):
            with self.assertRaises(ValueError):
                load_model("yolo11n")

    def test_duplicate_detection_is_false_positive(self):
        gt = [[0, 0, 10, 10]]
        predictions = [{"xyxy": gt[0], "score": .9}, {"xyxy": gt[0], "score": .8}]
        matched, extra, missed = match_boxes(gt, predictions)
        self.assertEqual((len(matched), len(extra), len(missed)), (1, 1, 0))

    def test_one_box_cannot_match_two_people(self):
        gt = [[0, 0, 10, 10], [1, 0, 11, 10]]
        matched, extra, missed = match_boxes(gt, [{"xyxy": gt[0], "score": .9}])
        self.assertEqual((len(matched), len(extra), len(missed)), (1, 0, 1))

    def test_iou_and_tiled_merge(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1/3)
        self.assertEqual(len(nms([{"xyxy": [0, 0, 10, 10], "score": .8},
                                  {"xyxy": [0, 0, 10, 10], "score": .7}])), 1)

    def test_tiles_cover_right_edge_and_short_images(self):
        self.assertEqual(tile_origins(120, 320, .2), [0])
        origins = tile_origins(1000, 320, .2)
        self.assertEqual(origins[-1]+320, 1000)
        self.assertTrue(all(b-a <= 320 for a, b in zip(origins, origins[1:])))

    def test_invalid_annotation_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.txt"
            p.write_text("0 .9 .5 .5 .2\n")
            _, errors = parse_labels(p)
            self.assertTrue(errors)

    def test_mixed_polygons_and_boxes_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mixed.txt"
            p.write_text("0 .5 .5 .2 .2\n0 .2 .2 .4 .2 .4 .6 .2 .6\n")
            boxes, errors = parse_labels(p)
            self.assertFalse(errors)
            self.assertEqual(len(boxes), 2)
            self.assertAlmostEqual(boxes[1][1], .3)
            self.assertAlmostEqual(boxes[1][2], .4)
            self.assertAlmostEqual(boxes[1][3], .2)
            self.assertAlmostEqual(boxes[1][4], .4)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional COCO dependency unavailable")
    def test_paired_session_bootstrap_identical_models_have_zero_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "label.txt"
            p.write_text("0 .5 .5 .5 .5\n")
            rows = [{"image_id": name, "session_id": session, "label_path": str(p), "width": 100, "height": 100}
                    for name, session in [("a", "s1"), ("b", "s2")]]
            predictions = {name: [{"xyxy": [25, 25, 75, 75], "score": .9}] for name in ["a", "b"]}
            repeated = {str(seed): predictions for seed in [1, 2, 3]}
            summary, differences = paired_bootstrap(rows, {"A": repeated, "B": repeated}, iterations=3)
            self.assertAlmostEqual(summary[0]["CI_low"], 1.)
            self.assertAlmostEqual(differences[0]["CI_low"], 0.)
            self.assertAlmostEqual(differences[0]["CI_high"], 0.)

    def test_bootstrap_repeated_sessions_get_new_image_ids(self):
        rows = [{"image_id": "a", "session_id": "s"}]
        rr, pp = resample_sessions(rows, {"a": [{"xyxy": [0, 0, 1, 1], "score": .9}]}, ["s", "s"])
        self.assertEqual(len(rr), 2)
        self.assertNotEqual(rr[0]["image_id"], rr[1]["image_id"])
        self.assertEqual(len(pp), 2)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional COCO dependency unavailable")
    def test_official_ap_perfect_and_empty_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "label.txt"
            p.write_text("0 .5 .5 .5 .5\n")
            row = {"image_id": "a", "label_path": str(p), "width": 100, "height": 100}
            perfect = coco_metrics([row], {"a": [{"xyxy": [25, 25, 75, 75], "score": .9}]})
            self.assertAlmostEqual(perfect["AP50_95"], 1.)
            self.assertAlmostEqual(coco_metrics([row], {"a": []})["AP50_95"], 0.)

    def test_confidence_selected_from_given_validation_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "label.txt"
            p.write_text("0 .5 .5 .5 .5\n")
            rows = [{"image_id": "v", "label_path": str(p), "width": 100, "height": 100}]
            predictions = {"v": [{"xyxy": [25, 25, 75, 75], "score": .4},
                                  {"xyxy": [1, 1, 8, 8], "score": .15}]}
            threshold, _ = select_confidence(rows, predictions)
            self.assertTrue(.15 < threshold <= .4)


class ProtocolTests(unittest.TestCase):
    def make_project(self, root):
        thermal = Path(root) / "data/images/labels"
        for directory in ["train", "valid", "test"]:
            (thermal / directory / "images").mkdir(parents=True)
            (thermal / directory / "labels").mkdir()
        metadata = []
        for i in range(20):
            directory = "train" if i < 12 else "valid" if i < 16 else "test"
            name = f"image_{i}.jpg"
            image = Image.new("RGB", (80, 60), (i*9, i*7, i*5))
            ImageDraw.Draw(image).rectangle((i, i, i+10, i+8), fill=(255, 200-i, 10))
            image.save(thermal / directory / "images" / name)
            (thermal / directory / "labels" / name.replace(".jpg", ".txt")).write_text("0 .5 .5 .25 .5\n")
            metadata.append({"image_id": f"thermal_v4/{directory}/{name}",
                             "session_id": f"flight_{i//2}", "source_video": f"flight_{i//2}.mp4",
                             "provenance_verified": "true"})
        return metadata

    def test_split_is_grouped_and_detects_changed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = self.make_project(root)
            audit_dataset(root, "thermal_v4", root / "audit")
            write_csv(root / "sessions.csv", meta)
            create_protocols(root / "audit/image_manifest.csv", root / "sessions.csv", root / "splits")
            fold, protocol, rows = check_protocol(root / "splits/fold_00")
            parts = {key: [r for r in rows if r["split"] == key] for key in ["train", "val", "test"]}
            assert_no_leakage(parts)
            self.assertEqual(sum(len(r) for r in parts.values()), 20)
            self.assertEqual(protocol["group_unit"], "verified_source_session")
            Path(rows[0]["label_path"]).write_text("0 .4 .4 .1 .1\n")
            with self.assertRaises(ValueError):
                check_protocol(fold)

    def test_missing_session_map_stops_training_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_project(root)
            audit_dataset(root, "thermal_v4", root / "audit")
            with self.assertRaises(ValueError):
                create_protocols(root / "audit/image_manifest.csv",
                                 root / "audit/session_map_TEMPLATE.csv", root / "splits")

    def test_cv_tests_every_image_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = self.make_project(root)
            audit_dataset(root, "thermal_v4", root / "audit")
            write_csv(root / "sessions.csv", meta)
            create_protocols(root / "audit/image_manifest.csv", root / "sessions.csv",
                             root / "splits", mode="grouped_cv", folds=5)
            tested = []
            for fold in (root / "splits").glob("fold_*"):
                self.assertEqual(read_json(fold / "protocol.json")["total_folds"], 5)
                tested.extend(r["image_id"] for r in read_csv(fold / "assignments.csv") if r["split"] == "test")
            self.assertEqual(len(tested), 20)
            self.assertEqual(len(set(tested)), 20)

    def test_pixel_duplicates_across_splits_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_project(root)
            thermal = root / "data/images/labels"
            (thermal / "test/images/image_18.jpg").write_bytes((thermal / "train/images/image_0.jpg").read_bytes())
            audit_dataset(root, "thermal_v4", root / "audit")
            issues = read_csv(root / "audit/issues.csv")
            self.assertTrue(any("identical_pixels" in r["issue"] for r in issues))

    def test_normalized_copy_preserves_source_and_converts_polygons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_project(root)
            source = root / "data/images/labels/train/labels/image_0.txt"
            original = "0 .5 .5 .25 .5\n0 .1 .1 .2 .1 .2 .3 .1 .3\n"
            source.write_text(original)
            rows = audit_dataset(root, "thermal_v4", root / "audit")
            selected = next(r for r in rows if r["image_id"].endswith("/image_0.jpg"))
            self.assertEqual(source.read_text(), original)
            clean = Path(selected["label_path"])
            self.assertNotEqual(source, clean)
            self.assertTrue(all(len(line.split()) == 5 for line in clean.read_text().splitlines()))
            self.assertEqual(selected["raw_polygon_rows"], 1)

    def test_legacy_appended_runs_are_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs/yolov11"
            run.mkdir(parents=True)
            (run / "args.yaml").write_text("model: yolov8m.pt\nepochs: 3\nseed: 0\n")
            write_csv(run / "results.csv", [
                {"epoch": 1, "metrics/mAP50-95(B)": .2},
                {"epoch": 2, "metrics/mAP50-95(B)": .3},
                {"epoch": 1, "metrics/mAP50-95(B)": .4},
                {"epoch": 2, "metrics/mAP50-95(B)": .5}])
            rows = legacy_run_audit(root, root / "audit")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["args_model"], "yolov8m.pt")
            self.assertTrue(all(r["status"] == "AMBIGUOUS_CONCATENATED_LOG" for r in rows))

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional COCO dependency unavailable")
    def test_report_aggregates_consistent_seed_records(self):
        from uav_review.report import summarize
        from uav_review.metrics import point_metrics, truths
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = self.make_project(root)
            audit_dataset(root, "thermal_v4", root / "audit")
            write_csv(root / "sessions.csv", meta)
            create_protocols(root / "audit/image_manifest.csv", root / "sessions.csv", root / "splits")
            fold = root / "splits/fold_00"
            protocol = read_json(fold / "protocol.json")
            rows = [r for r in read_csv(fold / "assignments.csv") if r["split"] == "test"]
            predictions = {r["image_id"]: [{"xyxy": box, "score": .9} for box in truths(r)] for r in rows}
            for model in ["yolov8n", "yolo26n"]:
                for seed in [17, 42, 93]:
                    run = root / "training" / f"{model}_{seed}"
                    run.mkdir(parents=True)
                    write_json(run / "run.json", {
                        "status": "TRAINED", "model_key": model, "seed": seed, "dataset": "thermal_v4",
                        "modality": "thermal_pseudocolor_export", "fold": str(fold),
                        "protocol_sha": protocol["protocol_sha"], "config": {"training": {"imgsz": 640}}})
                    evaluation = run / "evaluation_full_fp32"
                    write_json(evaluation / "test_predictions.json", predictions)
                    write_json(evaluation / "frozen_evaluation_settings.json", {
                        "variant": "full", "precision": "fp32", "imgsz": 640})
                    write_json(evaluation / "test_metrics.json", {
                        **coco_metrics(rows, predictions), **point_metrics(rows, predictions, .25),
                        "protocol_sha": protocol["protocol_sha"], "variant": "full", "precision": "fp32"})
            with patch("uav_review.report.plot_comparison"), patch("uav_review.report.plot_history"), patch("uav_review.report.plot_detection_outputs"):
                summarize(root / "training", root / "report", bootstrap_iterations=3)
            summaries = list((root / "report").glob("*/model_comparison_mean_SD.csv"))
            self.assertEqual(len(summaries), 1)
            results = read_csv(summaries[0])
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r["completed_seeds"] == "3" for r in results))
            self.assertTrue(all(abs(float(r["AP50_95_mean"])-1.) < 1e-8 for r in results))


if __name__ == "__main__":
    unittest.main()
