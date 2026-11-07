import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from uav_review.core import digest_object, environment, read_json, sha256, write_csv, write_json
from uav_review.notebook import (install_notebook_dependencies, prepare_notebook_protocol,
                                 resolve_device, run_notebook_command, run_notebook_measurement,
                                 train_notebook_jobs)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def make_protocol(self, root):
        from test_pipeline import ProtocolTests
        from uav_review.audit import audit_dataset
        metadata = ProtocolTests().make_project(root)
        audit_dataset(root, "thermal_v4", root / "audit")
        write_csv(root / "sessions.csv", metadata)
        prepare_notebook_protocol(root / "audit/image_manifest.csv", root / "sessions.csv",
                                  root / "protocol", mode="grouped_cv", folds=5)
        return metadata, root / "protocol/fold_00"

    def test_installation_uses_current_kernel_python_and_preserves_torch_by_default(self):
        with patch("uav_review.notebook.subprocess.check_call") as invoke, contextlib.redirect_stdout(io.StringIO()):
            install_notebook_dependencies(PACKAGE_ROOT)
        self.assertEqual(invoke.call_count, 1)
        args = invoke.call_args.args[0]
        self.assertEqual(args[:4], [sys.executable, "-m", "pip", "install"])
        self.assertEqual(args[-1], str(PACKAGE_ROOT / "requirements.txt"))

    def test_loaded_package_change_requires_kernel_restart(self):
        calls = 0
        def version(name):
            nonlocal calls
            if name == "numpy":
                calls += 1
                return "1.0" if calls == 1 else "2.0"
            return "unchanged"
        with patch.dict(sys.modules, {"numpy": types.SimpleNamespace(__version__="1.0")}), \
                patch("uav_review.notebook._version", side_effect=version), \
                patch("uav_review.notebook.subprocess.check_call"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Restart Kernel"):
                install_notebook_dependencies(PACKAGE_ROOT)

    def test_explicit_torch_replacement_requires_official_index(self):
        with patch("uav_review.notebook.subprocess.check_call") as invoke, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                install_notebook_dependencies(PACKAGE_ROOT, reinstall_torch=True)
            with self.assertRaisesRegex(ValueError, "official"):
                install_notebook_dependencies(PACKAGE_ROOT, torch_index_url="https://example.org/wheels")
            invoke.assert_not_called()

    def test_device_selection_does_not_silently_replace_explicit_cuda(self):
        cpu = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
        gpu = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
        self.assertEqual(resolve_device("auto", cpu), "cpu")
        self.assertEqual(resolve_device("auto", gpu), "0")
        self.assertEqual(resolve_device("cuda:1", gpu), "1")
        with self.assertRaisesRegex(RuntimeError, "explicitly requested"):
            resolve_device("0", cpu)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            resolve_device("2", gpu)
        with self.assertRaises(ValueError):
            resolve_device("0,1", gpu)

    def test_real_child_command_uses_kernel_environment_and_saves_progress(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            log = Path(tmp) / "Output Folder أحمد/doctor.log"
            self.assertEqual(run_notebook_command(PACKAGE_ROOT, ["doctor"], log_file=log), 0)
            text = log.read_text(encoding="utf-8")
            self.assertIn("CUDA_available", text)
            self.assertIn("python_executable", text)
            self.assertIn("uav_review", text)

    def test_child_failure_is_reported_with_its_log(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            log = Path(tmp) / "failed.log"
            with self.assertRaisesRegex(RuntimeError, "exited with code"):
                run_notebook_command(PACKAGE_ROOT, ["unknown-command"], log_file=log)
            self.assertTrue(log.is_file())
            self.assertIn("invalid choice", log.read_text(encoding="utf-8"))

    def test_notebook_interrupt_terminates_its_active_child(self):
        class InterruptedOutput(io.StringIO):
            def __next__(self):
                raise KeyboardInterrupt()
        child = Mock(stdout=InterruptedOutput())
        child.poll.return_value = None
        child.wait.return_value = 0
        with patch("uav_review.notebook.subprocess.Popen", return_value=child), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                run_notebook_command(PACKAGE_ROOT, ["doctor"])
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=10)
        self.assertTrue(child.stdout.closed)

    def test_frozen_protocol_reuse_checks_changed_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            metadata, fold = self.make_protocol(root)
            options = (root / "audit/image_manifest.csv", root / "sessions.csv", root / "protocol")
            before = (fold / "protocol.json").read_bytes()
            prepare_notebook_protocol(*options, mode="grouped_cv", folds=5)
            self.assertEqual((fold / "protocol.json").read_bytes(), before)
            metadata[0]["session_id"] = "different_source_session"
            write_csv(root / "sessions.csv", metadata)
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare_notebook_protocol(*options, mode="grouped_cv", folds=5)

    def test_training_retry_preserves_partial_attempt_and_reuses_only_matching_completed_job(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            _, fold = self.make_protocol(root)
            protocol = read_json(fold / "protocol.json")
            config = {"models": ["yolov8n"], "seeds": [17], "device": "cpu", "training": {"workers": 0}}
            config_path = root / "config.json"
            write_json(config_path, config)
            output = root / "training"
            job = output / "yolov8n__seed_17"
            write_json(job / "attempt_001/yolov8n__seed_17/run.json", {"status": "FAILED"})
            (job / "attempt_001.log").write_text("preserve original failure")
            def complete(_package, arguments, log_file=None):
                attempt = Path(arguments[arguments.index("--out")+1])
                run = attempt / "yolov8n__seed_17"
                run.mkdir(parents=True)
                checkpoint = run / "fixture.pt"
                checkpoint.write_bytes(b"orchestration test only; never loaded as model weights")
                write_json(run / "run.json", {
                    "status": "TRAINED", "model_key": "yolov8n", "seed": 17, "fold": str(fold),
                    "config": config, "config_sha": digest_object(config), "environment": environment(),
                    "protocol_sha": protocol["protocol_sha"], "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint)})
                return 0
            with patch("uav_review.notebook.run_notebook_command", side_effect=complete) as invoke:
                first = train_notebook_jobs(PACKAGE_ROOT, config_path, fold, output)
                second = train_notebook_jobs(PACKAGE_ROOT, config_path, fold, output)
                self.assertEqual(invoke.call_count, 1)
                self.assertEqual(first, second)
                self.assertIn("attempt_002", str(first[0]))
                self.assertTrue((job / "attempt_001.log").is_file())
                self.assertEqual(read_json(job / "attempt_001/yolov8n__seed_17/run.json")["status"], "FAILED")
                config["training"]["workers"] = 1
                write_json(config_path, config)
                with self.assertRaisesRegex(ValueError, "different settings"):
                    train_notebook_jobs(PACKAGE_ROOT, config_path, fold, output)

    def test_measurement_retry_archives_partial_outputs_and_checks_frozen_parameters(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp) / "run"
            run.mkdir()
            output = run / "evaluation_full_fp32"
            write_json(output / "test_predictions.json", {"unfinished": True})
            metadata = {"checkpoint_sha256": "fixture_hash", "protocol_sha": "fixture_protocol"}
            def complete(_package, arguments, log_file=None):
                output.mkdir()
                for name in ["test_metrics.json", "test_predictions.json", "val_predictions.json", "operating_point.json"]:
                    write_json(output / name, {})
                for name in ["validation_threshold_sweep.csv", "image_errors.csv", "instance_errors.csv", "size_analysis.csv",
                             "condition_analysis.csv", "occlusion_truncation_analysis.csv"]:
                    (output / name).write_text("header\n")
                write_json(output / "frozen_evaluation_settings.json", {**metadata, "variant": "full", "precision": "fp32",
                    "tile_size": 320, "overlap": .2, "instance_conditions_sha256": None})
            with patch("uav_review.engine.get_trained_run", return_value=(run, metadata, [])), \
                    patch("uav_review.notebook.run_notebook_command", side_effect=complete) as invoke:
                run_notebook_measurement(PACKAGE_ROOT, "evaluate", run)
                run_notebook_measurement(PACKAGE_ROOT, "evaluate", run)
                self.assertEqual(invoke.call_count, 1)
                self.assertTrue((run / "incomplete_outputs/evaluation_full_fp32__attempt_001/test_predictions.json").is_file())
                with self.assertRaisesRegex(ValueError, "different settings"):
                    run_notebook_measurement(PACKAGE_ROOT, "evaluate", run, overlap=.4)


if __name__ == "__main__":
    unittest.main()
