"""Local dataset setup. Safe to launch from a different working directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check and audit the locally downloaded UAV dataset.")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "local_config.json"))
    parser.add_argument("--project-root", help="Full local project folder; overrides local_config.json")
    parser.add_argument("--dataset", choices=["thermal_v4", "rgb_frames"])
    parser.add_argument("--dataset-root", help="Explicit standalone dataset folder, if moved from its original layout")
    parser.add_argument("--study-root", help="Local output directory; overrides local_config.json")
    parser.add_argument("--check-only", action="store_true", help="Check paths and file pairing without writing outputs")
    parser.add_argument("--legacy-logs", action="store_true", help="Also inspect old results.csv logs below project_root")
    args = parser.parse_args(argv)
    from uav_review.paths import load_local_settings, preflight_dataset
    try:
        settings = load_local_settings(args.config, {
            "project_root": args.project_root, "dataset": args.dataset,
            "dataset_root": args.dataset_root, "study_root": args.study_root,
        })
        checked = preflight_dataset(settings["project_root"], settings["dataset"], settings["dataset_root"])
        print(json.dumps(checked, indent=2, ensure_ascii=False))
        if args.check_only:
            print("Local dataset paths and image/annotation pairs are ready for the audit.")
            return 0
        from uav_review.audit import audit_dataset, legacy_run_audit
        audit = settings["study_root"] / "audit"
        audit_dataset(checked["project_root"], settings["dataset"], audit, dataset_root=checked["dataset_root"])
        if args.legacy_logs:
            if checked["project_root"] is None:
                raise ValueError("--legacy-logs requires project_root as well as any standalone dataset_root.")
            legacy_run_audit(checked["project_root"], audit / "legacy")
        print(f"Local audit: {audit}")
        print(f"Review annotation issues: {audit / 'issues.csv'}")
        print(f"Complete source provenance: {audit / 'session_map_TEMPLATE.csv'}")
        print("No training was started. Continue in UAV_Revision_Study.ipynb after verified session metadata is available.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
