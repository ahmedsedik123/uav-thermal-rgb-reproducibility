"""One-epoch engineering checks on real input subsets, NOT publication results."""
import argparse
import copy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON returned by the real-data preparation stage")
    parser.add_argument("--out", required=True, help="Dedicated engineering-check directory")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from uav_review.core import digest_object, read_csv, read_json, write_csv, write_json
    from uav_review.modality_study import execute_study
    source = read_json(args.plan)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    plan = copy.deepcopy(source)
    plan["study_dir"] = str(out)
    plan["config"].update(study_name="CPU_ENGINEERING_SMOKE_ONLY", seeds=[17], device="cpu")
    plan["config"]["timing"].update(seconds=2, warmup=2)
    for modality, cohort in plan["cohorts"].items():
        original = Path(cohort["folds"][0])
        items = read_csv(original / "assignments.csv")
        selected = []
        for split, count in [("train", 8), ("val", 2), ("test", 2)]:
            candidates = [r for r in items if r["split"] == split and int(r["n_instances"]) > 0]
            selected.extend(candidates[:count])
        fold = out / modality / "protocol/fold_00"
        fold.mkdir(parents=True, exist_ok=True)
        write_csv(fold / "assignments.csv", selected)
        for split in ["train", "val", "test"]:
            (fold / f"{split}.txt").write_text("\n".join(r["image_path"] for r in selected if r["split"] == split)+"\n", encoding="utf-8")
        write_json(fold / "data.yaml", {"path": str(fold), "nc": 1, "names": ["Human"],
            **{s: str(fold / f"{s}.txt") for s in ["train", "val", "test"]}})
        protocol = read_json(original / "protocol.json")
        protocol.update(assignments_sha=digest_object(selected), source_manifest_sha=digest_object(selected),
            engineering_validation_only=True, total_folds=1, fold=0,
            note="CPU ENGINEERING SMOKE ONLY. One epoch at 128 pixels on 8 training, 2 validation and 2 test images. Not the manuscript experiment.",
            counts={s: {"images": sum(r["split"] == s for r in selected),
                        "humans": sum(int(r["n_instances"]) for r in selected if r["split"] == s),
                        "sessions": 0,
                        "groups": len({r["leakage_group"] for r in selected if r["split"] == s})}
                    for s in ["train", "val", "test"]})
        protocol.pop("protocol_sha", None);protocol["protocol_sha"] = digest_object(protocol)
        write_json(fold / "protocol.json", protocol)
        cohort.update(folds=[str(fold)], images=len(selected))
    write_json(out / "SMOKE_PLAN_NOT_PAPER_RESULTS.json", plan)
    (out / "EXPERIMENT_PURPOSE.txt").write_text(
        "ENGINEERING VALIDATION ONLY\nAll ten models use the same 1-epoch, 128-pixel CPU check on each real-data modality. "
        "This does not provide the requested 100-epoch/three-seed study or valid modality rankings. Fusion has no real paired inputs.\n")
    results = execute_study(plan, device="cpu", training_overrides={
        "epochs": 1, "imgsz": 128, "batch": 2, "workers": 0, "plots": False, "amp": False})
    for r in results: r["purpose"] = "CPU_ENGINEERING_VALIDATION_ONLY_NOT_PAPER_RESULTS"
    write_csv(out / "ENGINEERING_CHECK_RESULTS.csv", results)
    print("ENGINEERING CHECK COMPLETE", sum(r["stage"] == "complete" for r in results), "/", len(results), flush=True)


if __name__ == "__main__":
    main()
