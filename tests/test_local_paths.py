"""Local dataset integration checks; no detector weights or CUDA required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from uav_review.audit import audit_dataset
from uav_review.core import read_csv, read_json, write_csv, write_json
from uav_review.engine import check_protocol
from uav_review.paths import load_local_settings, preflight_dataset
from uav_review.protocol import create_protocols


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class LocalDatasetTests(unittest.TestCase):
    def make_project(self, project, dataset="thermal_v4"):
        relative = "data/images/labels" if dataset == "thermal_v4" else "Dataset/DroneHumanDetection"
        root = project / relative
        directories = ["train", "valid", "test"] if dataset == "thermal_v4" else ["train", "val"]
        metadata = []
        for split in directories:
            for subdir in ["images", "labels"]:
                (root / split / subdir).mkdir(parents=True)
            for i in range(3):
                name = f"frame_{split}_{i}.JPG"
                Image.new("RGB", (16, 16), (30+len(metadata), 20+i*20, 10)).save(root / split / "images" / name)
                (root / split / "labels" / f"frame_{split}_{i}.TXT").write_text("0 .5 .5 .25 .25\n")
                metadata.append({"image_id": f"{dataset}/{split}/{name}", "session_id": f"session_{len(metadata)}",
                                 "source_video": f"original_{len(metadata)}.mp4", "provenance_verified": "true"})
        return root, metadata

    def test_config_anchors_paths_to_config_file_and_supports_unicode_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Research Folder أحمد"
            project = base / "Local Project"
            self.make_project(project)
            config = base / "local_config.json"
            write_json(config, {"project_root": "Local Project", "study_root": "My Experiments", "dataset": "thermal_v4"})
            settings = load_local_settings(config)
            self.assertEqual(settings["project_root"], project)
            self.assertEqual(settings["study_root"], base / "My Experiments")
            checked = preflight_dataset(settings["project_root"], settings["dataset"])
            self.assertEqual(sum(r["images"] for r in checked["counts"].values()), 9)

    def test_launcher_runs_from_other_directory_and_builds_local_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "Drone Project أحمد"
            dataset, _ = self.make_project(project)
            config = base / "local.json"
            write_json(config, {"project_root": project.name, "study_root": "Output Folder", "dataset": "thermal_v4"})
            outside = base / "unrelated cwd"
            outside.mkdir()
            command = [sys.executable, str(PACKAGE_ROOT / "run_local.py"), "--config", str(config)]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            checked = subprocess.run(command + ["--check-only"], cwd=outside, env=env,
                                     capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn(dataset.as_posix(), checked.stdout)
            self.assertFalse((base / "Output Folder").exists())
            audited = subprocess.run(command, cwd=outside, env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(audited.returncode, 0, audited.stderr)
            rows = read_csv(base / "Output Folder/audit/image_manifest.csv")
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(Path(r["image_path"]).is_absolute() and Path(r["image_path"]).is_file() for r in rows))
            self.assertTrue(all(Path(r["label_path"]).suffix == ".txt" and Path(r["label_path"]).is_file() for r in rows))

    def test_single_download_wrapper_detected_but_multiple_copies_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "Downloaded Folder"
            project = wrapper / "DroneHumanProject"
            self.make_project(project)
            self.assertEqual(preflight_dataset(wrapper, "thermal_v4")["project_root"], project.as_posix())
            self.make_project(wrapper / "Another Copy")
            with self.assertRaisesRegex(ValueError, "Multiple"):
                preflight_dataset(wrapper, "thermal_v4")

    def test_standalone_dataset_override_and_modality_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "Original Project"
            thermal, _ = self.make_project(project)
            moved = Path(tmp) / "Moved Thermal Dataset"
            thermal.rename(moved)
            checked = preflight_dataset(None, "thermal_v4", moved)
            self.assertEqual(checked["dataset_root"], moved.as_posix())
            self.assertIsNone(checked["project_root"])
            # An available RGB dataset must never silently replace missing thermal files.
            self.make_project(project, "rgb_frames")
            with self.assertRaisesRegex(FileNotFoundError, "Cannot locate thermal_v4"):
                preflight_dataset(project, "thermal_v4")
            self.assertEqual(preflight_dataset(project, "rgb_frames")["counts"]["val"]["images"], 3)

    def test_missing_pairs_empty_split_and_ambiguous_names_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root, _ = self.make_project(project)
            label = root / "train/labels/frame_train_0.TXT"
            contents = label.read_text()
            label.unlink()
            with self.assertRaisesRegex(ValueError, "lack annotation"):
                preflight_dataset(project, "thermal_v4")
            label.write_text(contents)
            Image.new("RGB", (16, 16)).save(root / "train/images/frame_train_0.png")
            with self.assertRaisesRegex(ValueError, "Ambiguous image stem"):
                preflight_dataset(project, "thermal_v4")
            (root / "train/images/frame_train_0.png").unlink()
            for path in (root / "test/images").iterdir():
                path.unlink()
            with self.assertRaisesRegex(ValueError, "No supported images"):
                preflight_dataset(project, "thermal_v4")

    def test_generated_backend_paths_are_local_and_stale_lists_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "Data Folder أحمد"
            _, metadata = self.make_project(project)
            audit = base / "Local Study/audit"
            rows = audit_dataset(project, "thermal_v4", audit)
            write_csv(audit / "verified.csv", metadata)
            protocols = base / "Local Study/protocol"
            create_protocols(audit / "image_manifest.csv", audit / "verified.csv", protocols)
            fold = protocols / "fold_00"
            _, _, assignments = check_protocol(fold)
            yaml = read_json(fold / "data.yaml")
            self.assertEqual(yaml["path"], fold.as_posix())
            for split in ["train", "val", "test"]:
                listed = Path(yaml[split]).read_text(encoding="utf-8").splitlines()
                self.assertEqual(listed, [r["image_path"] for r in assignments if r["split"] == split])
                for image_path in listed:
                    image = Path(image_path)
                    label = image.parent.parent / "labels" / (image.stem + ".txt")
                    self.assertTrue(image.is_file() and label.is_file())
            train_list = fold / "train.txt"
            contents = train_list.read_text(encoding="utf-8")
            train_list.write_text("/old/computer/missing.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train.txt differs"):
                check_protocol(fold)
            train_list.write_text(contents, encoding="utf-8")
            yaml["test"] = "/old/computer/test.txt"
            write_json(fold / "data.yaml", yaml)
            with self.assertRaisesRegex(ValueError, "stale or changed test path"):
                check_protocol(fold)
            # A copied manifest is rejected before it can create another protocol.
            Path(rows[0]["image_path"]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Rerun the audit and split"):
                create_protocols(audit / "image_manifest.csv", audit / "verified.csv", base / "Another Protocol")

    def test_blank_configuration_explains_the_required_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "local.json"
            write_json(config, {"project_root": "", "dataset_root": ""})
            with self.assertRaisesRegex(ValueError, "Set project_root"):
                load_local_settings(config)

    def test_normalized_images_are_independent_of_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "Project"
            self.make_project(project)
            rows = audit_dataset(project, "thermal_v4", Path(tmp) / "audit")
            source, normalized = Path(rows[0]["source_image_path"]), Path(rows[0]["image_path"])
            before = source.read_bytes()
            # An audit created by an earlier package may have used hard links.
            normalized.unlink()
            os.link(source, normalized)
            audit_dataset(project, "thermal_v4", Path(tmp) / "audit")
            Image.new("RGB", (16, 16), (255, 0, 0)).save(normalized)
            self.assertEqual(source.read_bytes(), before)
            self.assertNotEqual(normalized.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
