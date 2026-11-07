from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="UAV human-detection revision pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit images, annotations and old run logs.")
    audit.add_argument("--project-root")
    audit.add_argument("--dataset-root", help="Explicit standalone dataset folder")
    audit.add_argument("--dataset", choices=["thermal_v4", "rgb_frames"], default="thermal_v4")
    audit.add_argument("--out", required=True)
    check = commands.add_parser("check-data", help="Check local dataset paths and image/annotation pairing.")
    check.add_argument("--project-root")
    check.add_argument("--dataset-root")
    check.add_argument("--dataset", choices=["thermal_v4", "rgb_frames"], default="thermal_v4")
    split = commands.add_parser("split", help="Create verified session partitions or an explicit exploratory family holdout.")
    split.add_argument("--manifest", required=True)
    split.add_argument("--sessions", help="Verified source-session CSV; required except in exploratory mode.")
    split.add_argument("--out", required=True)
    split.add_argument("--mode", choices=["holdout", "grouped_cv", "exploratory"], default="holdout")
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--test-fraction", type=float, default=0.15)
    split.add_argument("--val-fraction-of-remainder", type=float, default=0.17647058823529413)
    train = commands.add_parser("train", help="Train fresh models for every seed.")
    train.add_argument("--config", required=True)
    train.add_argument("--fold", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--models", nargs="+")
    train.add_argument("--seeds", nargs="+", type=int)
    evaluate = commands.add_parser("evaluate", help="Choose confidence on validation, then evaluate the held-out test.")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--variant", choices=["full", "tiled"], default="full")
    evaluate.add_argument("--tile-size", type=int, default=320)
    evaluate.add_argument("--overlap", type=float, default=0.2)
    evaluate.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    evaluate.add_argument("--instance-conditions", help="Completed per-instance occlusion/truncation CSV")
    timing = commands.add_parser("benchmark", help="Measure actual-host processing latency and sustained rate.")
    timing.add_argument("--run", required=True)
    timing.add_argument("--seconds", type=float, default=60)
    timing.add_argument("--warmup", type=int, default=50)
    timing.add_argument("--platform-type", choices=["workstation", "embedded"], required=True)
    timing.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    timing.add_argument("--variant", choices=["full", "tiled"], default="full")
    timing.add_argument("--tile-size", type=int, default=320)
    timing.add_argument("--overlap", type=float, default=0.2)
    report = commands.add_parser("report", help="Generate consistent tables, figures and optional session/seed CIs.")
    report.add_argument("--results", required=True)
    report.add_argument("--out", required=True)
    report.add_argument("--bootstrap", type=int, default=0)
    tracking = commands.add_parser("track", help="Optional independently annotated video experiment.")
    tracking.add_argument("--run", required=True)
    tracking.add_argument("--video", required=True)
    tracking.add_argument("--out", required=True)
    tracking.add_argument("--ground-truth")
    tracking.add_argument("--session", help="Verified held-out session ID; required with sequence ground truth.")
    tracking.add_argument("--tracker", choices=["bytetrack.yaml", "botsort.yaml"], default="bytetrack.yaml")
    tracking.add_argument("--zones", help="JSON of image-plane normalized polygon vertices")
    agreement = commands.add_parser("agreement", help="Compute a specified inter-annotator box-matching statistic.")
    agreement.add_argument("--manifest", required=True)
    agreement.add_argument("--rater-a", required=True)
    agreement.add_argument("--rater-b", required=True)
    agreement.add_argument("--out", required=True)
    agreement.add_argument("--iou", type=float, default=0.5)
    export = commands.add_parser("export", help="Export FP32 ONNX; target-device validation remains separate.")
    export.add_argument("--run", required=True)
    export.add_argument("--out", required=True)
    commands.add_parser("doctor", help="Display installed packages, CUDA, and registered models.")
    args = parser.parse_args()
    if args.command == "audit":
        from .audit import audit_dataset, legacy_run_audit
        from .paths import resolve_dataset_location
        project, dataset = resolve_dataset_location(args.project_root, args.dataset, args.dataset_root)
        audit_dataset(project, args.dataset, args.out, dataset_root=dataset)
        if project is not None:
            legacy_run_audit(project, args.out)
        print(f"Audit written to {args.out}. Complete verified session metadata before splitting.")
    elif args.command == "check-data":
        from .paths import preflight_dataset
        print(json.dumps(preflight_dataset(args.project_root, args.dataset, args.dataset_root), indent=2))
    elif args.command == "split":
        from .protocol import create_protocols
        print(create_protocols(args.manifest, args.sessions, args.out, args.mode, args.folds,
                               args.seed, args.test_fraction, args.val_fraction_of_remainder))
    elif args.command == "train":
        from .engine import train
        print(train(args.config, args.fold, args.out, args.models, args.seeds))
    elif args.command == "evaluate":
        from .engine import evaluate
        print(evaluate(args.run, args.variant, args.tile_size, args.overlap, args.precision, args.instance_conditions))
    elif args.command == "benchmark":
        from .engine import benchmark
        print(benchmark(args.run, args.seconds, args.warmup, args.precision, args.platform_type,
                        args.variant, args.tile_size, args.overlap))
    elif args.command == "report":
        from .report import summarize
        print(summarize(args.results, args.out, args.bootstrap))
    elif args.command == "track":
        from .tracking import track_video
        print(track_video(args.run, args.video, args.out, args.ground_truth, args.tracker, args.zones, args.session))
    elif args.command == "agreement":
        from .annotations import annotation_agreement
        print(json.dumps(annotation_agreement(args.manifest, args.rater_a, args.rater_b, args.out, args.iou), indent=2))
    elif args.command == "export":
        from .engine import export_onnx
        print(export_onnx(args.run, args.out))
    elif args.command == "doctor":
        from .core import environment
        from .engine import MODELS
        information = environment()
        try:
            import torch
            information["CUDA_available"] = torch.cuda.is_available()
            information["CUDA_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except (ImportError, OSError, RuntimeError) as exc:
            information["CUDA_available"] = False
            information["torch_import_error"] = f"{type(exc).__name__}: {exc}"
        information["models"] = MODELS
        print(json.dumps(information, indent=2))


if __name__ == "__main__":
    main()
