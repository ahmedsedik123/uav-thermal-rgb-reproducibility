"""Detection metrics on original-image boxes, including session-aware uncertainty."""
from __future__ import annotations

import contextlib
import io
from collections import defaultdict

import numpy as np

from .audit import parse_labels


def truths(row):
    boxes, errors = parse_labels(row["label_path"])
    if errors:
        raise ValueError(f"Invalid labels for {row['image_id']}: {errors}")
    width, height = int(row["width"]), int(row["height"])
    return [[max(0., (x-w/2)*width), max(0., (y-h/2)*height),
             min(width, (x+w/2)*width), min(height, (y+h/2)*height)]
            for _, x, y, w, h in boxes]


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., right-left)*max(0., bottom-top)
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return intersection / max(area_a+area_b-intersection, 1e-12)


def nms(predictions, threshold=0.55):
    remaining = sorted(predictions, key=lambda p: p["score"], reverse=True)
    keep = []
    while remaining:
        first, *remaining = remaining
        keep.append(first)
        remaining = [p for p in remaining if iou(first["xyxy"], p["xyxy"]) < threshold]
    return keep


def match_boxes(gt, predictions, confidence=0.25, iou_threshold=0.5):
    """Score-ordered one-to-one matching; one detection cannot explain several people."""
    predictions = sorted((p for p in predictions if p["score"] >= confidence),
                         key=lambda p: p["score"], reverse=True)
    unmatched = set(range(len(gt)))
    matches, false_positives = [], []
    for prediction in predictions:
        overlaps = [(iou(prediction["xyxy"], gt[j]), j) for j in unmatched]
        overlap, index = max(overlaps, default=(0., -1))
        if overlap >= iou_threshold:
            matches.append((index, prediction))
            unmatched.remove(index)
        else:
            false_positives.append(prediction)
    return matches, false_positives, sorted(unmatched)


def point_metrics(rows, predictions, confidence):
    tp = fp = fn = 0
    for row in rows:
        matched, extras, missed = match_boxes(truths(row), predictions.get(row["image_id"], []), confidence)
        tp += len(matched)
        fp += len(extras)
        fn += len(missed)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
            "F1": 2*precision*recall/(precision+recall) if precision+recall else 0.,
            "confidence": float(confidence), "matching_IoU": 0.5}


def select_confidence(validation_rows, predictions):
    candidates = [point_metrics(validation_rows, predictions, value)
                  for value in np.linspace(0.05, 0.95, 19)]
    # A predeclared tie rule prefers fewer low-confidence detections.
    best = max(candidates, key=lambda item: (item["F1"], item["confidence"]))
    return best["confidence"], candidates


def coco_metrics(rows, predictions, return_curve=False):
    """Official COCO evaluator: IoU .50:.05:.95, 101 recall points, maxDets=100."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Install pycocotools for official COCO AP; no substitute scores are generated.") from exc
    dataset = {"info": {}, "images": [], "annotations": [], "categories": [{"id": 1, "name": "Human"}]}
    detections = []
    annotation_id = 1
    for image_number, row in enumerate(rows, 1):
        dataset["images"].append({"id": image_number, "width": int(row["width"]), "height": int(row["height"])})
        for box in truths(row):
            x, y, right, bottom = box
            width, height = right-x, bottom-y
            dataset["annotations"].append({"id": annotation_id, "image_id": image_number,
                                           "category_id": 1, "bbox": [x, y, width, height],
                                           "area": width*height, "iscrowd": 0})
            annotation_id += 1
        for prediction in predictions.get(row["image_id"], []):
            x, y, right, bottom = prediction["xyxy"]
            detections.append({"image_id": image_number, "category_id": 1, "score": prediction["score"],
                               "bbox": [x, y, max(0., right-x), max(0., bottom-y)]})
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = dataset
        ground_truth.createIndex()
        if detections:
            result = ground_truth.loadRes(detections)
        else:
            result = COCO()
            result.dataset = {**dataset, "annotations": []}
            result.createIndex()
        evaluator = COCOeval(ground_truth, result, "bbox")
        evaluator.params.imgIds = list(range(1, len(rows)+1))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ["AP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    output = {name: float(value) if value >= 0 else None for name, value in zip(names, evaluator.stats)}
    output["max_detections_per_image"] = 100
    output["area_definition"] = "bounding-box area in supplied original-image pixels"
    if return_curve:
        # IoU=.50, person class, all area, maxDet=100.
        precision = evaluator.eval["precision"][0, :, 0, 0, 2]
        output["PR50"] = [{"recall": float(r), "precision": float(p)}
                          for r, p in zip(evaluator.params.recThrs, precision) if p >= 0]
    return output


def error_analysis(rows, predictions, confidence, instance_conditions=None):
    images, instances = [], []
    metadata = {}
    for item in instance_conditions or []:
        key = (item["image_id"], int(item["gt_index"]))
        if key in metadata:
            raise ValueError("Repeated instance-condition annotation")
        metadata[key] = item
    for row in rows:
        gt = truths(row)
        matched, extras, missed = match_boxes(gt, predictions.get(row["image_id"], []), confidence)
        matched_indices = {index for index, _ in matched}
        images.append({"image_id": row["image_id"], "session_id": row["session_id"],
                       "GT_people": len(gt), "TP": len(matched), "FP": len(extras), "FN": len(missed),
                       "thermal_clutter": row.get("thermal_clutter", ""),
                       "motion_blur": row.get("motion_blur", "")})
        for j, box in enumerate(gt):
            height = box[3]-box[1]
            size = "<15" if height < 15 else "15-31" if height < 32 else "32-63" if height < 64 else ">=64"
            instances.append({"image_id": row["image_id"], "gt_index": j,
                              "height_original_pixels": height, "height_bin": size,
                              "occlusion": metadata.get((row["image_id"], j), {}).get("occlusion", "unknown") or "unknown",
                              "truncation": metadata.get((row["image_id"], j), {}).get("truncation", "unknown") or "unknown",
                              "detected": int(j in matched_indices)})
    sizes = []
    for size in ["<15", "15-31", "32-63", ">=64"]:
        subset = [r for r in instances if r["height_bin"] == size]
        sizes.append({"height_bin": size, "GT_people": len(subset),
                      "TP": sum(r["detected"] for r in subset),
                      "recall": sum(r["detected"] for r in subset)/len(subset) if subset else None})
    conditions = []
    for field in ["thermal_clutter", "motion_blur"]:
        for value in ["yes", "no"]:
            subset = [r for r in images if str(r.get(field, "")).lower() == value]
            tp, fp, fn = [sum(r[k] for r in subset) for k in ["TP", "FP", "FN"]]
            conditions.append({"condition": field, "value": value, "images": len(subset),
                               "TP": tp, "FP": fp, "FN": fn,
                               "precision": tp/(tp+fp) if tp+fp else None,
                               "recall": tp/(tp+fn) if tp+fn else None,
                               "FP_per_image": fp/len(subset) if subset else None,
                               "status": "measured" if subset else "NEEDS_MANUAL_CONDITION_LABELS"})
    instance_strata = []
    for field in ["occlusion", "truncation"]:
        for value in sorted({r[field] for r in instances}):
            subset = [r for r in instances if r[field] == value]
            instance_strata.append({
                "condition": field, "value": value, "GT_people": len(subset),
                "TP": sum(r["detected"] for r in subset),
                "recall": sum(r["detected"] for r in subset)/len(subset) if subset else None,
                "status": "NEEDS_MANUAL_CONDITION_LABELS" if value == "unknown" else "measured"})
    return images, instances, sizes, conditions, instance_strata


def resample_sessions(rows, predictions, sampled_groups):
    """Copy repeated clusters to new image IDs so COCO does not deduplicate bootstrap samples."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("leakage_group", row["session_id"])].append(row)
    sample_rows, sample_predictions = [], {}
    for occurrence, group in enumerate(sampled_groups):
        for row in groups[group]:
            key = f"{occurrence}:{row['image_id']}"
            sample_rows.append({**row, "image_id": key})
            sample_predictions[key] = predictions.get(row["image_id"], [])
    return sample_rows, sample_predictions


def paired_bootstrap(rows, predictions_by_model, iterations=1000, random_seed=2026):
    """Hierarchical seed/session bootstrap with paired session draws between models.

    predictions_by_model[model][training_seed] maps image IDs to detections.
    Training seeds are resampled independently within each model. Session draws are
    shared between models, preserving the shared test set in model differences.
    """
    if any(str(r.get("provenance_verified", "true")).lower() not in {"true", "1", "yes"}
           for r in rows):
        raise ValueError("Session bootstrap requires verified source sessions; exploratory image families are not sessions.")
    groups = sorted({r.get("leakage_group", r["session_id"]) for r in rows})
    if len(groups) < 2:
        raise ValueError("At least two independent held-out groups are needed for a session bootstrap.")
    if any(len(seeds) < 3 for seeds in predictions_by_model.values()):
        raise ValueError("At least three completed training seeds per model are required by this analysis.")
    rng = np.random.default_rng(random_seed)
    values = {model: [] for model in predictions_by_model}
    for _ in range(iterations):
        sampled = list(rng.choice(groups, size=len(groups), replace=True))
        for model, seeds in predictions_by_model.items():
            seed_ids = sorted(seeds)
            selected = rng.choice(seed_ids, size=len(seed_ids), replace=True)
            scores = []
            for seed in selected:
                sample_rows, sample_predictions = resample_sessions(rows, seeds[seed], sampled)
                scores.append(coco_metrics(sample_rows, sample_predictions)["AP50_95"])
            values[model].append(float(np.mean(scores)))
    summary = []
    for model, samples in values.items():
        point = np.mean([coco_metrics(rows, p)["AP50_95"] for p in predictions_by_model[model].values()])
        lower, upper = np.quantile(samples, [0.025, 0.975])
        summary.append({"model": model, "AP50_95_mean_over_seeds": float(point),
                        "CI_low": float(lower), "CI_high": float(upper),
                        "test_groups": len(groups), "training_seeds": len(predictions_by_model[model]),
                        "bootstrap_iterations": iterations})
    differences = []
    models = sorted(values)
    for i, a in enumerate(models):
        for b in models[i+1:]:
            delta = np.asarray(values[a])-np.asarray(values[b])
            low, high = np.quantile(delta, [0.025, 0.975])
            differences.append({"model_A": a, "model_B": b, "difference": "AP50_95_A_minus_B",
                                "CI_low": float(low), "CI_high": float(high),
                                "includes_zero": bool(low <= 0 <= high)})
    return summary, differences
