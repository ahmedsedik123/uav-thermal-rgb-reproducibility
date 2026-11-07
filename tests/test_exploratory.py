"""Exploratory use must run without fabricating source-session evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_pipeline as fixtures
from uav_review.audit import audit_dataset
from uav_review.core import read_csv, read_json, write_csv, write_json
from uav_review.engine import check_protocol
from uav_review.metrics import coco_metrics, paired_bootstrap, point_metrics, truths
from uav_review.notebook import prepare_notebook_protocol
from uav_review.protocol import create_protocols


PACKAGE = Path(__file__).resolve().parents[1]


class ExploratoryTests(unittest.TestCase):
    def make_audit(self, root):
        fixtures.ProtocolTests().make_project(root)
        audit_dataset(root, "thermal_v4", root / "audit")
        return root / "audit/image_manifest.csv"

    def make_protocol(self, root):
        manifest = self.make_audit(root)
        create_protocols(manifest, None, root / "protocol", mode="exploratory")
        return root / "protocol/fold_00"

    def test_exploratory_runs_without_a_session_csv_and_keeps_sessions_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold = self.make_protocol(root)
            _, protocol, rows = check_protocol(fold)
            self.assertEqual(protocol["mode"], "exploratory")
            self.assertFalse(protocol["source_sessions_verified"])
            self.assertEqual(protocol["grouping_field"], "source_group")
            self.assertEqual(protocol["total_folds"], 1)
            self.assertEqual(len(rows), 20)
            self.assertEqual({r["split"] for r in rows}, {"train", "val", "test"})
            self.assertTrue(all(r["session_id"] == "" and r["provenance_verified"] == "false" for r in rows))
            self.assertTrue(all(part["sessions"] == 0 for part in protocol["counts"].values()))
            self.assertFalse((root / "audit/session_map_VERIFIED.csv").exists())

    def test_augmentations_and_identical_pixels_cannot_cross_exploratory_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures.ProtocolTests().make_project(root)
            thermal = root / "data/images/labels"
            shutil.copy2(thermal / "train/images/image_1.jpg", thermal / "test/images/image_18.jpg")
            audit_dataset(root, "thermal_v4", root / "audit")
            manifest = root / "audit/image_manifest.csv"
            rows = read_csv(manifest)
            first = next(r for r in rows if r["image_id"].endswith("/image_0.jpg"))
            later = next(r for r in rows if r["image_id"].endswith("/image_15.jpg"))
            later["family_key"] = first["family_key"]
            write_csv(manifest, rows)
            create_protocols(manifest, None, root / "protocol", mode="exploratory")
            _, _, assigned = check_protocol(root / "protocol/fold_00")
            for field in ["source_group", "family_key", "pixel_sha256"]:
                seen = {}
                for row in assigned:
                    seen.setdefault(row[field], set()).add(row["split"])
                self.assertTrue(all(len(splits) == 1 for splits in seen.values()), field)

    def test_verified_protocol_still_rejects_missing_or_unverified_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_audit(root)
            with self.assertRaisesRegex(ValueError, "Verified source-session"):
                create_protocols(manifest, None, root / "strict", mode="grouped_cv")
            with self.assertRaisesRegex(ValueError, "Verified session/source"):
                create_protocols(manifest, root / "audit/session_map_TEMPLATE.csv", root / "strict", mode="holdout")
            self.assertFalse((root / "strict").exists())

    def test_notebook_reuses_only_the_same_frozen_exploratory_split(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest = self.make_audit(root)
            prepare_notebook_protocol(manifest, None, root / "protocol", mode="exploratory", seed=42)
            before = (root / "protocol/fold_00/protocol.json").read_bytes()
            prepare_notebook_protocol(manifest, None, root / "protocol", mode="exploratory", seed=42)
            self.assertEqual(before, (root / "protocol/fold_00/protocol.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare_notebook_protocol(manifest, None, root / "protocol", mode="exploratory", seed=43)

    def test_cli_can_create_exploratory_partitions_without_sessions_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_audit(root)
            result = subprocess.run([sys.executable, "-m", "uav_review", "split", "--manifest", str(manifest),
                                     "--out", str(root / "protocol"), "--mode", "exploratory"],
                                    cwd=PACKAGE, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(read_json(root / "protocol/fold_00/protocol.json")["source_sessions_verified"])

    def test_session_bootstrap_cannot_treat_image_families_as_verified_sessions(self):
        rows = [{"image_id": "a", "session_id": "", "leakage_group": "family_a", "provenance_verified": "false"},
                {"image_id": "b", "session_id": "", "leakage_group": "family_b", "provenance_verified": "false"}]
        with self.assertRaisesRegex(ValueError, "verified source sessions"):
            paired_bootstrap(rows, {"model": {"1": {}, "2": {}, "3": {}}}, iterations=3)

    def test_exploratory_protocol_cannot_declare_verified_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fold = self.make_protocol(Path(tmp))
            protocol = read_json(fold / "protocol.json")
            protocol["source_sessions_verified"] = True
            write_json(fold / "protocol.json", protocol)
            with self.assertRaisesRegex(ValueError, "must not declare verified"):
                check_protocol(fold)

    def test_default_notebook_protocol_cell_proceeds_without_verified_csv(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.make_audit(root)
            notebook = read_json(PACKAGE / "Original_Notebook/Bundled_Thermal_Study.ipynb")
            cell = next(c for c in notebook["cells"] if "protocol" in c.get("metadata", {}).get("tags", []))
            namespace = {"EVALUATION_MODE": "exploratory", "AUDIT": root / "audit", "STUDY": root,
                         "read_csv": read_csv, "read_json": read_json}
            exec(compile("".join(cell["source"]), "notebook-protocol-cell", "exec"), namespace)
            self.assertEqual(namespace["BOOTSTRAP_ITERATIONS"], 0)
            self.assertEqual(namespace["MODE"], "exploratory")
            self.assertTrue((root / "protocol/fold_00/data.yaml").is_file())
            namespace["EVALUATION_MODE"] = "verified_sessions"
            with self.assertRaisesRegex(RuntimeError, "Complete verified source-session metadata"):
                exec(compile("".join(cell["source"]), "notebook-protocol-cell", "exec"), namespace)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "COCO dependency unavailable")
    def test_exploratory_reports_keep_scope_and_reject_session_intervals(self):
        from uav_review.report import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold = self.make_protocol(root)
            protocol = read_json(fold / "protocol.json")
            rows = [r for r in read_csv(fold / "assignments.csv") if r["split"] == "test"]
            predictions = {r["image_id"]: [{"xyxy": box, "score": .9} for box in truths(r)] for r in rows}
            run = root / "training/model_seed_17"
            write_json(run / "run.json", {"status": "TRAINED", "model_key": "yolov8n", "seed": 17,
                       "dataset": "thermal_v4", "modality": "thermal_pseudocolor_export", "fold": str(fold),
                       "protocol_sha": protocol["protocol_sha"], "config": {"training": {"imgsz": 640}}})
            evaluation = run / "evaluation_full_fp32"
            write_json(evaluation / "test_predictions.json", predictions)
            write_json(evaluation / "frozen_evaluation_settings.json", {"variant": "full", "precision": "fp32", "imgsz": 640})
            write_json(evaluation / "test_metrics.json", {**coco_metrics(rows, predictions),
                       **point_metrics(rows, predictions, .25), "protocol_sha": protocol["protocol_sha"],
                       "variant": "full", "precision": "fp32"})
            with patch("uav_review.report.plot_comparison"), patch("uav_review.report.plot_history"), \
                    patch("uav_review.report.plot_detection_outputs"):
                summarize(root / "training", root / "report", bootstrap_iterations=0)
            folder = next((root / "report").iterdir())
            self.assertIn("EXPLORATORY", (folder / "RESULT_SCOPE.txt").read_text())
            for filename in ["test_results_by_seed.csv", "model_comparison_mean_SD.csv"]:
                for row in read_csv(folder / filename):
                    self.assertEqual(row["evaluation_scope"], "exploratory_image_family_holdout")
                    self.assertEqual(row["source_sessions_verified"], "False")
            self.assertFalse(list(folder.glob("*confidence_intervals*")))
            with self.assertRaisesRegex(ValueError, "Use --bootstrap 0"):
                summarize(root / "training", root / "invalid_intervals", bootstrap_iterations=3)
            self.assertFalse((root / "invalid_intervals").exists())


if __name__ == "__main__":
    unittest.main()
