"""Isolated pretrained-model inference check. This is NOT an accuracy experiment."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--image", action="append", default=[], help="MODALITY=local_image_path")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=128)
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    from uav_review.core import environment, sha256, write_json
    from uav_review.engine import MODELS, load_model
    result = {"model": args.model, "status": "INITIALIZING", "checks": [],
              "purpose": "PRETRAINED_INITIALIZATION_AND_INFERENCE_CHECK_ONLY_NOT_PAPER_RESULTS",
              "input_resolution": args.imgsz, "environment": environment(), "device": args.device}
    try:
        start = time.perf_counter()
        model = load_model(args.model, weights_dir=ROOT / "weights")
        result.update(initialization_seconds=time.perf_counter()-start,
                      weights_sha256=sha256(ROOT / "weights" / MODELS[args.model]["weights"]),
                      initial_classes=len(model.names), parameters=sum(p.numel() for p in model.model.parameters()))
        for item in args.image:
            modality, source = item.split("=", 1)
            start = time.perf_counter()
            prediction = model.predict(source=source, imgsz=args.imgsz, device=args.device,
                conf=.25, iou=.7, max_det=300, batch=1, quantize=32,
                nms=MODELS[args.model]["nms"], save=False, verbose=False)[0]
            result["checks"].append({"modality": modality, "image_sha256": sha256(source),
                "input_image": Path(source).name, "status": "PASSED",
                "returned_boxes": len(prediction.boxes) if prediction.boxes is not None else 0,
                "first_call_elapsed_seconds_NOT_A_BENCHMARK": time.perf_counter()-start})
        result["status"] = "PASSED"
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out, result)
    print(f"{args.model}: {result['status']} ({len(result['checks'])} real-input checks)", flush=True)


if __name__ == "__main__":
    main()
