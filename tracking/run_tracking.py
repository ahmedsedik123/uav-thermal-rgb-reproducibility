"""Run the archived thermal YOLOv8m / ByteTrack protocol on an original video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAME_FIELDS = [
    "frame", "time_s", "detections", "visible_tracked_persons",
    "cumulative_track_ids", "down_crossings", "up_crossings",
]
TRACK_FIELDS = [
    "frame", "time_s", "track_id", "confidence", "x1", "y1", "x2", "y2",
    "center_x", "center_y",
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class LineCrossings:
    line_y: int
    margin: int
    max_gap: int
    last_frame: dict[int, int] = field(default_factory=dict)
    last_stable_side: dict[int, int] = field(default_factory=dict)
    down: int = 0
    up: int = 0
    down_ids: set[int] = field(default_factory=set)
    up_ids: set[int] = field(default_factory=set)

    def observe(self, frame: int, track_id: int, center_y: int) -> None:
        side = (
            -1 if center_y < self.line_y - self.margin
            else 1 if center_y > self.line_y + self.margin
            else 0
        )
        previous_frame = self.last_frame.get(track_id)
        previous_side = self.last_stable_side.get(track_id, 0)
        if previous_frame is not None and frame - previous_frame <= self.max_gap:
            if previous_side == -1 and side == 1:
                self.down += 1
                self.down_ids.add(track_id)
            elif previous_side == 1 and side == -1:
                self.up += 1
                self.up_ids.add(track_id)
        if side:
            self.last_stable_side[track_id] = side
        self.last_frame[track_id] = frame


def runtime_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "opencv-python", "lap"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run(video: Path, checkpoint: Path, output: Path, device: str | None = None) -> dict:
    settings = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))
    params = settings["parameters"]
    for name, path in (("Original input video", video), ("Checkpoint", checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    video_digest = sha256(video)
    if video_digest == settings["artifacts"]["tracked_video.mp4"]["sha256"]:
        raise ValueError("Use the original input clip, not the archived annotated output.")
    checkpoint_digest = sha256(checkpoint)
    if checkpoint_digest != settings["artifacts"][settings["checkpoint_asset"]]["sha256"]:
        raise ValueError("Checkpoint hash differs from the archived tracking best.pt.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {output}")

    # Import the inference dependencies only when actually running the experiment.
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    class_id = params["class_id"]
    class_name = str(model.names.get(class_id, "")).strip().lower()
    if class_name not in {"person", "human", "people", "item"}:
        raise ValueError(f"Class {class_id} has unexpected label {class_name!r}.")
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    writer = None
    seen: set[int] = set()
    frame_number = total_visible = maximum_visible = 0
    track_options = dict(
        persist=True, tracker=params["tracker_config"], conf=params["confidence"],
        iou=params["iou"], imgsz=params["image_size"], classes=[class_id], verbose=False,
    )
    if device is not None:
        track_options["device"] = device
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open source video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError("Source video has invalid FPS or dimensions.")
        line_y = int(height * params["line_fraction"])
        crossings = LineCrossings(line_y, params["line_margin_px"], params["max_gap_frames"])
        writer = cv2.VideoWriter(
            str(output / "tracked_video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )
        if not writer.isOpened():
            raise ValueError("Could not create the annotated MP4 video.")

        with (
            (output / "counting_results.csv").open("w", newline="", encoding="utf-8") as counts,
            (output / "tracking_details.csv").open("w", newline="", encoding="utf-8") as tracks,
        ):
            count_writer = csv.writer(counts)
            track_writer = csv.writer(tracks)
            count_writer.writerow(FRAME_FIELDS)
            track_writer.writerow(TRACK_FIELDS)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                time_s = round((frame_number - 1) / fps, 3)
                boxes = model.track(source=frame, **track_options)[0].boxes
                detections = len(boxes) if boxes is not None else 0
                active: set[int] = set()
                if boxes is not None and boxes.id is not None:
                    for box, track_id, confidence in zip(
                        boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().tolist(),
                        boxes.conf.cpu().numpy(), strict=True,
                    ):
                        track_id = int(track_id)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        active.add(track_id)
                        seen.add(track_id)
                        crossings.observe(frame_number, track_id, cy)
                        track_writer.writerow((
                            frame_number, time_s, track_id, round(float(confidence), 4),
                            x1, y1, x2, y2, cx, cy,
                        ))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame, f"Person ID {track_id}: {confidence:.2f}",
                            (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (0, 255, 0), 2,
                        )
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                visible = len(active)
                total_visible += visible
                maximum_visible = max(maximum_visible, visible)
                cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (min(width, 420), 155), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                for index, label in enumerate((
                    f"Visible tracked: {visible}", f"Distinct track IDs: {len(seen)}",
                    f"Down crossings: {crossings.down}", f"Up crossings: {crossings.up}",
                )):
                    cv2.putText(
                        frame, label, (10, 28 + index * 31), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2,
                    )
                count_writer.writerow((
                    frame_number, time_s, detections, visible, len(seen),
                    crossings.down, crossings.up,
                ))
                writer.write(frame)
                if frame_number % 100 == 0:
                    print(f"Frame {frame_number} | visible: {visible} | track IDs: {len(seen)}")
        if not frame_number:
            raise ValueError("No decodable video frames were found.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    summary = {
        "detector": settings["detector"], "modality": settings["modality"],
        "training_seed": settings["training_seed"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest, "model_class_id": class_id,
        "model_class_name": class_name, "tracker": "ByteTrack", "video": str(video),
        "source_video_sha256": video_digest, "frames_processed": frame_number,
        "video_fps": fps, "duration_seconds": round(frame_number / fps, 2),
        "confidence_threshold": params["confidence"], "iou_threshold": params["iou"],
        "input_size": params["image_size"], "total_distinct_track_ids": len(seen),
        "downward_crossing_events": crossings.down, "upward_crossing_events": crossings.up,
        "distinct_downward_crossing_ids": len(crossings.down_ids),
        "distinct_upward_crossing_ids": len(crossings.up_ids),
        "average_visible_tracked_persons": round(total_visible / frame_number, 3),
        "maximum_visible_tracked_persons": maximum_visible,
        "tracking_accuracy_validated": False,
        "counting_line_y": line_y, "tracking_parameters": params, "device": device,
        "runtime": runtime_versions(),
    }
    (output / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Original unannotated video")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Archived tracking best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("tracking_runs/thermal_seed17"))
    parser.add_argument("--device", help="Optional inference device, e.g. cpu or 0")
    args = parser.parse_args()
    summary = run(args.video, args.checkpoint, args.output_dir, args.device)
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
