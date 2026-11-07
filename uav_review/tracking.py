"""Optional sequence experiment; track IDs are never equated with verified identities."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .core import read_csv, read_json, require_empty, write_csv, write_json
from .engine import get_trained_run, load_model


def track_video(run_path, video, output, ground_truth=None, tracker="bytetrack.yaml", zones_path=None,
                session_id=None):
    import cv2
    run_dir, run, assignments = get_trained_run(run_path)
    if ground_truth:
        test_sessions = {r["session_id"] for r in assignments if r["split"] == "test"}
        if not session_id or session_id not in test_sessions:
            raise ValueError("Quantitative sequence evaluation requires a verified held-out test-session ID.")
    operating = run_dir / "evaluation_full_fp32" / "operating_point.json"
    if not operating.is_file():
        raise ValueError("Run the validation-selected full-frame evaluation first.")
    confidence = read_json(operating)["confidence"]
    model = load_model(run["model_key"], run["checkpoint"])
    output = require_empty(output)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    zones = read_json(zones_path) if zones_path else {}
    rows, frame_rows, seen = [], [], set()
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            result = model.track(frame, persist=True, tracker=tracker, conf=confidence,
                                 imgsz=run["config"]["training"]["imgsz"],
                                 device=run["config"]["device"], verbose=False)[0]
            boxes = result.boxes
            count = len(boxes) if boxes is not None else 0
            zone_counts = {name: 0 for name in zones}
            if boxes is not None and boxes.id is not None:
                for xyxy, identity, score in zip(boxes.xyxy.cpu().tolist(),
                                                  boxes.id.cpu().tolist(), boxes.conf.cpu().tolist()):
                    left, top, right, bottom = xyxy
                    identity = int(identity)
                    seen.add(identity)
                    rows.append([frame_number, identity, left, top, right-left, bottom-top, score, -1, -1, -1])
                    point = ((left+right)/2, bottom)
                    for name, vertices in zones.items():
                        polygon = np.asarray([[x*frame.shape[1], y*frame.shape[0]] for x, y in vertices], dtype=np.float32)
                        zone_counts[name] += int(cv2.pointPolygonTest(polygon, point, False) >= 0)
            frame_rows.append({"frame": frame_number, "detected_people": count,
                               "distinct_track_ids_so_far": len(seen),
                               **{f"zone_{key}_occupancy": value for key, value in zone_counts.items()}})
    finally:
        capture.release()
    if not frame_rows:
        raise ValueError("No readable frames")
    mot_file = output / "predictions_MOT.txt"
    with mot_file.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    write_csv(output / "frame_counts.csv", frame_rows)
    summary = {"video": str(Path(video).resolve()), "frames": frame_number, "tracker": tracker,
               "declared_source_session": session_id,
               "distinct_track_ids": len(seen), "confidence_selected_on": "validation",
               "confidence": confidence, "checkpoint_sha256": run["checkpoint_sha256"],
               "identity_note": "Distinct track IDs can overcount or merge real people; they are not validated unique people.",
               "zone_note": "Optional image-plane polygon occupancy; no geographic geofencing is inferred."}
    if ground_truth:
        summary["tracking_evaluation"] = evaluate_mot(ground_truth, mot_file, frame_number)
    else:
        summary["tracking_evaluation"] = {"status": "NEEDS_INDEPENDENT_SEQUENCE_GROUND_TRUTH"}
    write_json(output / "tracking_summary.json", summary)
    return output


def read_mot(path, ground_truth=False):
    records = []
    with Path(path).open() as stream:
        for line in csv.reader(stream):
            if not line:
                continue
            values = [float(v) for v in line]
            if len(values) < 7:
                raise ValueError("MOT rows need frame,id,x,y,width,height,confidence.")
            if ground_truth and values[6] <= 0:
                continue
            frame, identity, x, y, w, h = values[:6]
            records.append({"frame": int(frame), "id": int(identity), "xywh": [x, y, w, h]})
    return records


def evaluate_mot(gt_path, pred_path, number_frames):
    import motmetrics as mm
    gt, predictions = read_mot(gt_path, True), read_mot(pred_path)
    accumulator = mm.MOTAccumulator(auto_id=True)
    absolute_errors = []
    for frame in range(1, number_frames+1):
        actual = [r for r in gt if r["frame"] == frame]
        predicted = [r for r in predictions if r["frame"] == frame]
        distances = mm.distances.iou_matrix(
            np.asarray([r["xywh"] for r in actual]).reshape(-1, 4),
            np.asarray([r["xywh"] for r in predicted]).reshape(-1, 4), max_iou=0.5)
        accumulator.update([r["id"] for r in actual], [r["id"] for r in predicted], distances)
        absolute_errors.append(abs(len(actual)-len(predicted)))
    values = mm.metrics.create().compute(
        accumulator, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses"], name="sequence").iloc[0]
    result = {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
    true_unique = len({r["id"] for r in gt})
    pred_unique = len({r["id"] for r in predictions})
    result.update(frame_count_MAE=float(np.mean(absolute_errors)),
                  GT_unique_people=true_unique, predicted_distinct_track_ids=pred_unique,
                  unique_count_absolute_error=abs(pred_unique-true_unique),
                  matching_IoU=0.5,
                  assumption="GT must annotate all people over the complete evaluated sequence, including empty frames.")
    return result
