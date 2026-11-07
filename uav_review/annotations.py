"""Explicit box-matching agreement; does not invent an inter-annotator percentage."""
from pathlib import Path

from .audit import parse_labels
from .core import read_csv, write_json
from .metrics import match_boxes


def annotation_agreement(manifest_path, rater_a, rater_b, output, iou_threshold=0.5):
    count_a = count_b = matches = 0
    overlaps = []
    from .metrics import iou
    rows = read_csv(manifest_path)
    for row in rows:
        name = Path(row["image_path"]).stem + ".txt"
        a, errors_a = parse_labels(Path(rater_a) / name)
        b, errors_b = parse_labels(Path(rater_b) / name)
        if errors_a or errors_b:
            raise ValueError(f"Both independent annotations are needed for {name}")
        width, height = int(row["width"]), int(row["height"])
        def xyxy(values):
            return [[(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
                    for _, x, y, w, h in values]
        aa, bb = xyxy(a), xyxy(b)
        paired, _, _ = match_boxes(aa, [{"xyxy": box, "score": 1.} for box in bb], 0., iou_threshold)
        matches += len(paired)
        count_a += len(aa)
        count_b += len(bb)
        overlaps.extend(iou(aa[index], prediction["xyxy"]) for index, prediction in paired)
    result = {"images_double_annotated": len(rows), "rater_A_boxes": count_a,
              "rater_B_boxes": count_b, "matched_boxes": matches,
              "matching_IoU_threshold": iou_threshold,
              "symmetric_box_matching_F1": 2*matches/(count_a+count_b) if count_a+count_b else None,
              "mean_IoU_of_matched_boxes": sum(overlaps)/len(overlaps) if overlaps else None,
              "definition": "Score-ordered one-to-one box matching; equal-score rater boxes retain annotation order.",
              "note": "This is a specified detection-box agreement statistic, not Cohen kappa."}
    write_json(output, result)
    return result
