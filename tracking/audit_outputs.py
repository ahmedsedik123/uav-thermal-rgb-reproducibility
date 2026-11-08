"""Check the supplied tracking CSVs, summary and optional MP4 without inference."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from run_tracking import FRAME_FIELDS, HERE, TRACK_FIELDS, LineCrossings, sha256


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def read_rows(path: Path, fields: list[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, f"{path.name}: unexpected columns")
        return list(reader)


def verify_asset(path: Path, entry: dict) -> None:
    require(path.stat().st_size == entry["bytes"], f"{path.name}: size differs")
    require(sha256(path) == entry["sha256"], f"{path.name}: SHA-256 differs")


def audit(video: Path | None = None) -> dict:
    settings = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))
    outputs = HERE / "outputs"
    for name in ("counting_results.csv", "tracking_details.csv", "tracking_summary.json"):
        verify_asset(outputs / name, settings["artifacts"][name])

    counts = read_rows(outputs / "counting_results.csv", FRAME_FIELDS)
    tracks = read_rows(outputs / "tracking_details.csv", TRACK_FIELDS)
    summary = json.loads((outputs / "tracking_summary.json").read_text(encoding="utf-8"))
    props = settings["annotated_video"]
    fps, width, height = props["fps"], props["width"], props["height"]
    require(summary["video_fps"] == fps, "Summary FPS differs")
    require(len(counts) == props["frames"] == summary["frames_processed"], "Frame count differs")
    require(round(len(counts) / fps, 2) == summary["duration_seconds"], "Duration differs")
    params = settings["parameters"]
    for key, value in (
        ("model_class_id", params["class_id"]), ("confidence_threshold", params["confidence"]),
        ("iou_threshold", params["iou"]), ("input_size", params["image_size"]),
        ("training_seed", settings["training_seed"]), ("detector", settings["detector"]),
        ("modality", settings["modality"]), ("tracker", "ByteTrack"),
        ("tracking_accuracy_validated", False),
    ):
        require(summary[key] == value, f"Summary setting differs: {key}")

    grouped = defaultdict(list)
    observations = Counter()
    for row in tracks:
        frame, track_id = int(row["frame"]), int(row["track_id"])
        require(1 <= frame <= len(counts), f"Out-of-range tracking frame {frame}")
        require(abs(float(row["time_s"]) - round((frame - 1) / fps, 3)) < 1e-6,
                f"Tracking timestamp differs at frame {frame}")
        x1, y1, x2, y2, cx, cy = (
            int(row[k]) for k in ("x1", "y1", "x2", "y2", "center_x", "center_y")
        )
        require(0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height, "Invalid box coordinates")
        require((cx, cy) == ((x1 + x2) // 2, (y1 + y2) // 2), "Incorrect box center")
        require(0.0 <= float(row["confidence"]) <= 1.0, "Invalid confidence")
        grouped[frame].append((track_id, cy))
        observations[track_id] += 1

    crossings = LineCrossings(
        int(height * params["line_fraction"]), params["line_margin_px"], params["max_gap_frames"],
    )
    seen: set[int] = set()
    visible_counts = []
    for frame, row in enumerate(counts, 1):
        require(int(row["frame"]) == frame, f"Nonconsecutive frame index at {frame}")
        require(abs(float(row["time_s"]) - round((frame - 1) / fps, 3)) < 1e-6,
                f"Counting timestamp differs at frame {frame}")
        active = {i for i, _ in grouped[frame]}
        visible = int(row["visible_tracked_persons"])
        require(len(active) == len(grouped[frame]) == visible, f"Track count differs at {frame}")
        require(int(row["detections"]) >= visible, f"Detection count too small at {frame}")
        seen.update(active)
        require(int(row["cumulative_track_ids"]) == len(seen), f"Cumulative IDs differ at {frame}")
        for track_id, cy in grouped[frame]:
            crossings.observe(frame, track_id, cy)
        require(int(row["down_crossings"]) == crossings.down and
                int(row["up_crossings"]) == crossings.up, f"Crossings differ at {frame}")
        visible_counts.append(visible)

    for key, value in (
        ("total_distinct_track_ids", len(seen)),
        ("average_visible_tracked_persons", round(sum(visible_counts) / len(counts), 3)),
        ("maximum_visible_tracked_persons", max(visible_counts)),
        ("downward_crossing_events", crossings.down), ("upward_crossing_events", crossings.up),
        ("distinct_downward_crossing_ids", len(crossings.down_ids)),
        ("distinct_upward_crossing_ids", len(crossings.up_ids)),
    ):
        require(summary[key] == value, f"Summary statistic differs: {key}")

    if video is not None:
        verify_asset(video, settings["artifacts"]["tracked_video.mp4"])
        command = [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
            "-of", "json", str(video),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        require(not result.stderr.strip(), "ffprobe reported decode errors")
        stream = json.loads(result.stdout)["streams"][0]
        require((stream["width"], stream["height"]) == (width, height), "Video dimensions differ")
        require(Fraction(stream["avg_frame_rate"]) == fps, "Video FPS differs")
        require(int(stream["nb_read_frames"]) == len(counts), "Decoded frame count differs")
        require(abs(float(stream["duration"]) - len(counts) / fps) < 0.001,
                "Decoded duration differs")

    return {
        "frames": len(counts), "fps": fps, "duration_s": round(len(counts) / fps, 2),
        "frames_with_assigned_tracks": sum(v > 0 for v in visible_counts),
        "tracked_box_observations": len(tracks),
        "distinct_track_ids": len(seen),
        "track_ids_observed_at_most_twice": sum(n <= 2 for n in observations.values()),
        "mean_visible_tracks_per_frame": round(sum(visible_counts) / len(counts), 3),
        "maximum_visible_tracks_per_frame": max(visible_counts),
        "last_active_frame_time_s": max(
            (float(r["time_s"]) for r in tracks), default=None,
        ),
        "downward_crossings": crossings.down, "upward_crossings": crossings.up,
        "annotated_video_decoded_and_hashed": video is not None,
        "tracking_accuracy_validated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Annotated release MP4; requires ffprobe")
    print(json.dumps(audit(parser.parse_args().video), indent=2))
