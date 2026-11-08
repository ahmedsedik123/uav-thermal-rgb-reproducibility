
# ================================================================
# UAV THERMAL HUMAN DETECTION, TRACKING AND COUNTING
# YOLOv8m (Seed 17) + ByteTrack
# ================================================================

from pathlib import Path
import cv2
import csv
import json
import numpy as np
from ultralytics import YOLO

# ================================================================
# 1. PROJECT CONFIGURATION
# ================================================================

PROJECT_DIR = Path(r"C:\Users\pc\UAV_Full_Project")

MODEL_NAME = "yolov8m__seed_17"

MODEL_PATH = (
    PROJECT_DIR
    / "experiments"
    / "all_modalities_v1"
    / "thermal"
    / "training"
    / "fold_00"
    / MODEL_NAME
    / "attempt_001"
    / MODEL_NAME
    / "fit"
    / "weights"
    / "best.pt"
)

# Search for the video in the project or notebook directory.
video_candidates = [
    PROJECT_DIR / "test_video.mp4",
    Path.cwd() / "test_video.mp4"
]

VIDEO_PATH = next(
    (p for p in video_candidates if p.is_file()),
    None
)

# Output directory
OUTPUT_DIR = (
    PROJECT_DIR
    / "tracking_counting_results"
    / MODEL_NAME
)

OUTPUT_VIDEO = OUTPUT_DIR / "tracked_video.mp4"
OUTPUT_COUNTS = OUTPUT_DIR / "counting_results.csv"
OUTPUT_TRACKS = OUTPUT_DIR / "tracking_details.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "tracking_summary.json"

# ================================================================
# 2. DETECTION AND TRACKING PARAMETERS
# ================================================================

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMG_SIZE = 640
TRACKER = "bytetrack.yaml"

# Your trained checkpoint uses class 0 = "item".
# This class should correspond to annotated humans.
PERSON_CLASS = 0

COUNTING_LINE_POSITION = 0.50
LINE_MARGIN = 8
MAX_CROSSING_GAP = 10

# ================================================================
# 3. VERIFY FILE PATHS
# ================================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained checkpoint not found:\n{MODEL_PATH}"
    )

if VIDEO_PATH is None:
    raise FileNotFoundError(
        "test_video.mp4 was not found in the project "
        "or notebook directory."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("UAV THERMAL HUMAN TRACKING AND COUNTING")
print("=" * 65)
print("Checkpoint:", MODEL_PATH)
print("Video:", VIDEO_PATH)

# ================================================================
# 4. LOAD TRAINED MODEL AND VERIFY CLASS
# ================================================================

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")
print("Model classes:", model.names)

class_name = str(
    model.names.get(PERSON_CLASS, "")
).strip().lower()

ACCEPTED_CLASS_NAMES = {
    "person", "human", "people", "item"
}

if class_name not in ACCEPTED_CLASS_NAMES:
    raise ValueError(
        f"Unexpected class name: '{class_name}'. "
        "Verify the training dataset class mapping."
    )

print("Selected human class ID:", PERSON_CLASS)
print("Stored class label:", class_name)

if class_name == "item":
    print(
        "Note: Class 'item' is treated as human "
        "according to the single-class dataset protocol."
    )

# ================================================================
# 5. OPEN INPUT VIDEO
# ================================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open test video.")

FPS = cap.get(cv2.CAP_PROP_FPS)

if FPS <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS.")

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

TOTAL_FRAMES = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

line_y = int(HEIGHT * COUNTING_LINE_POSITION)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (WIDTH, HEIGHT)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video.")

print("\nVideo information:")
print("Resolution:", WIDTH, "x", HEIGHT)
print("Frame rate:", FPS)
print("Total frames:", TOTAL_FRAMES)

# ================================================================
# 6. INITIALIZE TRACKING AND COUNTING VARIABLES
# ================================================================

unique_track_ids = set()

previous_positions = {}
last_stable_side = {}

crossing_down_ids = set()
crossing_up_ids = set()

crossing_down_events = 0
crossing_up_events = 0

frame_records = []
tracking_records = []

frame_number = 0

# ================================================================
# 7. RUN HUMAN DETECTION AND TRACKING
# ================================================================

print("\nStarting processing...")

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Run YOLOv8m and ByteTrack.
        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            classes=[PERSON_CLASS],
            verbose=False
        )

        result = results[0]

        current_track_ids = set()
        detection_count = 0

        if result.boxes is not None:

            detection_count = len(result.boxes)

            if result.boxes.id is not None:

                boxes = (
                    result.boxes.xyxy.cpu().numpy()
                )

                track_ids = (
                    result.boxes.id.int()
                    .cpu().tolist()
                )

                confidences = (
                    result.boxes.conf.cpu().numpy()
                )

                # =================================================
                # PROCESS INDIVIDUAL DETECTIONS
                # =================================================

                for box, track_id, confidence in zip(
                    boxes, track_ids, confidences
                ):

                    track_id = int(track_id)

                    x1, y1, x2, y2 = map(int, box)

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    current_track_ids.add(track_id)
                    unique_track_ids.add(track_id)

                    # Determine position relative to counting line.
                    if center_y < line_y - LINE_MARGIN:
                        stable_side = -1

                    elif center_y > line_y + LINE_MARGIN:
                        stable_side = 1

                    else:
                        stable_side = 0

                    # =================================================
                    # LINE-CROSSING DETECTION
                    # =================================================

                    previous = previous_positions.get(track_id)

                    if previous is not None:

                        gap = (
                            frame_number - previous["frame"]
                        )

                        if gap <= MAX_CROSSING_GAP:

                            old_side = last_stable_side.get(
                                track_id, 0
                            )

                            if old_side == -1 and stable_side == 1:

                                crossing_down_events += 1
                                crossing_down_ids.add(track_id)

                            elif old_side == 1 and stable_side == -1:

                                crossing_up_events += 1
                                crossing_up_ids.add(track_id)

                    if stable_side != 0:
                        last_stable_side[track_id] = stable_side

                    previous_positions[track_id] = {
                        "frame": frame_number,
                        "center_x": center_x,
                        "center_y": center_y
                    }

                    # =================================================
                    # SAVE TRACKING DETAILS
                    # =================================================

                    tracking_records.append({
                        "frame": frame_number,
                        "time_s": round(
                            (frame_number - 1) / FPS, 3
                        ),
                        "track_id": track_id,
                        "confidence": round(
                            float(confidence), 4
                        ),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": center_x,
                        "center_y": center_y
                    })

                    # Draw bounding box.
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Draw tracking ID.
                    cv2.putText(
                        frame,
                        f"Person ID {track_id}: {confidence:.2f}",
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 255, 0),
                        2
                    )

                    # Draw center point.
                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        4,
                        (0, 255, 255),
                        -1
                    )

        # ========================================================
        # 8. COUNT VISIBLE PERSONS AND TRACK IDs
        # ========================================================

        visible_persons = len(current_track_ids)
        cumulative_track_ids = len(unique_track_ids)

        # Draw counting line.
        cv2.line(
            frame,
            (0, line_y),
            (WIDTH, line_y),
            (0, 255, 255),
            2
        )

        # ========================================================
        # 9. DISPLAY COUNTING DASHBOARD
        # ========================================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (min(WIDTH, 420), 155),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.65,
            frame,
            0.35,
            0,
            frame
        )

        dashboard = [
            f"Visible tracked: {visible_persons}",
            f"Distinct track IDs: {cumulative_track_ids}",
            f"Down crossings: {crossing_down_events}",
            f"Up crossings: {crossing_up_events}"
        ]

        for index, text in enumerate(dashboard):

            cv2.putText(
                frame,
                text,
                (10, 28 + index * 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

        # ========================================================
        # 10. STORE FRAME-LEVEL RESULTS
        # ========================================================

        frame_records.append({
            "frame": frame_number,
            "time_s": round(
                (frame_number - 1) / FPS, 3
            ),
            "detections": detection_count,
            "visible_tracked_persons": visible_persons,
            "cumulative_track_ids": cumulative_track_ids,
            "down_crossings": crossing_down_events,
            "up_crossings": crossing_up_events
        })

        writer.write(frame)

        if frame_number % 100 == 0:

            print(
                f"Frame {frame_number}/{TOTAL_FRAMES} | "
                f"Visible: {visible_persons} | "
                f"IDs: {cumulative_track_ids} | "
                f"Down: {crossing_down_events} | "
                f"Up: {crossing_up_events}"
            )

finally:

    cap.release()
    writer.release()

# ================================================================
# 11. EXPORT COUNTING RESULTS
# ================================================================

count_fields = [
    "frame",
    "time_s",
    "detections",
    "visible_tracked_persons",
    "cumulative_track_ids",
    "down_crossings",
    "up_crossings"
]

with open(
    OUTPUT_COUNTS,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    csv_writer = csv.DictWriter(
        f,
        fieldnames=count_fields
    )

    csv_writer.writeheader()
    csv_writer.writerows(frame_records)

# ================================================================
# 12. EXPORT INDIVIDUAL TRACKING RESULTS
# ================================================================

track_fields = [
    "frame",
    "time_s",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y"
]

with open(
    OUTPUT_TRACKS,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    csv_writer = csv.DictWriter(
        f,
        fieldnames=track_fields
    )

    csv_writer.writeheader()
    csv_writer.writerows(tracking_records)

# ================================================================
# 13. GENERATE EXPERIMENTAL SUMMARY
# ================================================================

average_visible = (
    float(np.mean([
        r["visible_tracked_persons"]
        for r in frame_records
    ]))
    if frame_records else 0.0
)

maximum_visible = max(
    (
        r["visible_tracked_persons"]
        for r in frame_records
    ),
    default=0
)

summary = {
    "detector": "YOLOv8m",
    "modality": "thermal",
    "training_seed": 17,
    "checkpoint": str(MODEL_PATH),
    "model_class_id": PERSON_CLASS,
    "model_class_name": class_name,
    "tracker": "ByteTrack",
    "video": str(VIDEO_PATH),
    "frames_processed": frame_number,
    "video_fps": FPS,
    "duration_seconds": round(
        frame_number / FPS, 2
    ),
    "confidence_threshold": CONF_THRESHOLD,
    "iou_threshold": IOU_THRESHOLD,
    "input_size": IMG_SIZE,
    "total_distinct_track_ids": len(unique_track_ids),
    "downward_crossing_events": crossing_down_events,
    "upward_crossing_events": crossing_up_events,
    "distinct_downward_crossing_ids": len(crossing_down_ids),
    "distinct_upward_crossing_ids": len(crossing_up_ids),
    "average_visible_tracked_persons": round(
        average_visible, 3
    ),
    "maximum_visible_tracked_persons": maximum_visible,
    "tracking_accuracy_validated": False
}

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(summary, f, indent=4)

# ================================================================
# 14. DISPLAY RESULTS
# ================================================================

print("\n" + "=" * 65)
print("TRACKING AND COUNTING RESULTS")
print("=" * 65)

print("Detector: YOLOv8m")
print("Training seed: 17")
print("Model class:", class_name)
print("Tracking method: ByteTrack")

print("Frames processed:", frame_number)
print("Distinct track IDs:", len(unique_track_ids))
print("Downward crossings:", crossing_down_events)
print("Upward crossings:", crossing_up_events)
print("Average visible tracked persons:", round(
    average_visible, 2
))
print("Maximum visible tracked persons:", maximum_visible)

print("\nOUTPUT FILES")
print("Annotated video:", OUTPUT_VIDEO.resolve())
print("Counting results:", OUTPUT_COUNTS.resolve())
print("Tracking details:", OUTPUT_TRACKS.resolve())
print("Summary:", OUTPUT_SUMMARY.resolve())

print("\nProcessing completed successfully.")
