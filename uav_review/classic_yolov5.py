"""Classic anchor-based YOLOv5s via the bundled, pinned official implementation.

This adapter keeps classic yolov5s.pt separate from the modern yolov5su.pt path.
The shared pipeline owns grouped data, final COCO metrics and host timing.
"""
from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .core import read_csv, sha256, write_csv


NATIVE_COMMIT = "af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce"
NATIVE_ROOT = Path(__file__).resolve().parents[1] / "third_party/yolov5"
CLASSIC_WEIGHTS_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"


def verify_native_source(root=NATIVE_ROOT):
    root = Path(root).resolve()
    pin = json.loads((root / "SOURCE_PIN.json").read_text(encoding="utf-8"))
    if pin["commit"] != NATIVE_COMMIT:
        raise ValueError("Bundled classic YOLOv5 source has an unexpected commit.")
    for relative, expected in pin["files"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Bundled classic YOLOv5 source is missing or changed: {relative}")
    return {"repository": pin["repository"], "commit": pin["commit"],
            "source_manifest_sha256": sha256(root / "SOURCE_PIN.json")}


@contextlib.contextmanager
def blocked_optional_imports(names):
    """Keep local runs independent of optional cloud loggers/undeclared transforms."""
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in names}
    try:
        for name in names:
            sys.modules[name] = None
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def classic_hyperparameters(common, root=NATIVE_ROOT):
    import yaml
    hyp = yaml.safe_load((Path(root) / "data/hyps/hyp.scratch-low.yaml").read_text(encoding="utf-8"))
    shared = ["lr0", "lrf", "weight_decay", "warmup_epochs", "hsv_h", "hsv_s", "hsv_v",
              "degrees", "translate", "scale", "fliplr", "flipud", "mosaic", "mixup", "copy_paste"]
    for key in shared:
        hyp[key] = common[key]
    return hyp


def normalize_classic_history(source, target):
    mapping = {"metrics/precision": "metrics/precision(B)", "metrics/recall": "metrics/recall(B)",
               "metrics/mAP_0.5": "metrics/mAP50(B)", "metrics/mAP_0.5:0.95": "metrics/mAP50-95(B)"}
    rows = []
    for item in read_csv(source):
        row = {mapping.get(k.strip(), k.strip()): v.strip() for k, v in item.items()}
        # Native logs start at epoch zero; the shared histories use completed epoch counts.
        row["epoch"] = str(int(float(row["epoch"])) + 1)
        rows.append(row)
    if not rows:
        raise ValueError("Classic YOLOv5 produced no epoch history.")
    write_csv(target, rows)


class ClassicYolov5:
    def __init__(self, weights=None, checkpoint=None):
        import torch
        self.backend_details = {"backend": "native_classic_yolov5", **verify_native_source(),
                                "head": "anchor_based_external_NMS", "autoanchor": False}
        # These top-level modules are the official repository's serialization namespace.
        for name in ["models", "utils", "train", "val", "export"]:
            loaded = sys.modules.get(name)
            origin = getattr(loaded, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(NATIVE_ROOT):
                raise RuntimeError("Classic YOLOv5 needs its own process to avoid module-name collisions. Use the notebook's isolated training/evaluation cells.")
        sys.path.insert(0, str(NATIVE_ROOT))
        with blocked_optional_imports(["wandb", "comet_ml", "clearml"]):
            self.native_train = importlib.import_module("train")
            self.experimental = importlib.import_module("models.experimental")
            self.general = importlib.import_module("utils.general")
            self.augmentations = importlib.import_module("utils.augmentations")
            self.callbacks_class = importlib.import_module("utils.callbacks").Callbacks
            self.torch_utils = importlib.import_module("utils.torch_utils")
        initial = Path(checkpoint or weights or (NATIVE_ROOT.parent / "weights/yolov5s.pt")).resolve()
        if checkpoint and not initial.is_file():
            raise FileNotFoundError(initial)
        if not initial.is_file():
            initial.parent.mkdir(parents=True, exist_ok=True)
            temporary = initial.with_suffix(".download")
            try:
                torch.hub.download_url_to_file(CLASSIC_WEIGHTS_URL, str(temporary), progress=True)
                temporary.replace(initial)
            finally:
                temporary.unlink(missing_ok=True)
        self.ckpt_path = str(initial)
        self.model = self.experimental.attempt_load(str(initial), device=torch.device("cpu"), fuse=False)
        head = self.model.model[-1]
        if type(head).__module__ != "models.yolo" or type(head).__name__ != "Detect" or not hasattr(head, "anchors"):
            raise ValueError("Expected a classic anchor-based YOLOv5 Detect head; a YOLOv5u checkpoint cannot use this backend.")
        if self.model.yaml.get("depth_multiple") != .33 or self.model.yaml.get("width_multiple") != .5:
            raise ValueError("Expected the classic YOLOv5 small architecture.")
        self.overrides = {"nms": None}
        self._callbacks = []
        self._fused = False
        self.predictor = SimpleNamespace(device=torch.device("cpu"))
        self._tracker = None
        self._tracker_name = None

    @property
    def names(self):
        return self.model.names

    def add_callback(self, event, callback):
        if event != "on_model_save":
            raise ValueError(f"Unsupported classic adapter callback: {event}")
        self._callbacks.append(callback)

    def train(self, **common):
        original_argv = sys.argv
        try:
            sys.argv = ["native_yolov5"]
            opt = self.native_train.parse_opt(known=True)
        finally:
            sys.argv = original_argv
        fit = Path(common["project"]).resolve() / common["name"]
        if fit.exists():
            raise FileExistsError(f"Classic training output already exists: {fit}")
        options = {"weights": self.ckpt_path, "cfg": str(NATIVE_ROOT / "models/yolov5s.yaml"),
                   "data": str(Path(common["data"]).resolve()), "epochs": int(common["epochs"]),
                   "batch_size": int(common["batch"]), "imgsz": int(common["imgsz"]),
                   "workers": int(common["workers"]), "device": str(common["device"]),
                   "optimizer": common["optimizer"], "patience": int(common["patience"]),
                   "seed": int(common["seed"]), "save_dir": str(fit), "project": str(fit.parent),
                   "name": fit.name, "resume": False, "single_cls": True, "noautoanchor": True,
                   "noplots": not common["plots"], "noval": False, "nosave": False,
                   "save_period": -1, "cache": "ram" if common["cache"] else None,
                   "cos_lr": False, "rect": False, "multi_scale": False}
        for key, value in options.items():
            setattr(opt, key, value)
        hyp = classic_hyperparameters(common)
        opt.hyp = hyp
        device = self.torch_utils.select_device(opt.device, batch_size=opt.batch_size)
        self.trainer = SimpleNamespace(args=opt, batch_size=opt.batch_size, device=device,
                                       save_dir=fit, metrics={}, epoch=-1, last=fit / "weights/last.pt")
        callbacks = self.callbacks_class()

        def epoch_metrics(values, epoch, *_):
            if len(values) < 7:
                raise RuntimeError("Unexpected native validation metric layout.")
            self.trainer.epoch = epoch
            self.trainer.metrics = {"metrics/mAP50-95(B)": float(values[6])}

        def saved(last, epoch, *_):
            self.trainer.last = Path(last)
            self.trainer.epoch = epoch
            for callback in self._callbacks:
                callback(self.trainer)

        callbacks.register_action("on_fit_epoch_end", callback=epoch_metrics)
        callbacks.register_action("on_model_save", callback=saved)
        original_check_amp = self.native_train.check_amp

        def recorded_amp(model):
            actual = original_check_amp(model) if common["amp"] else False
            opt.actual_amp = bool(actual)
            self.backend_details["actual_amp"] = bool(actual)
            return actual

        self.native_train.check_amp = recorded_amp
        try:
            with blocked_optional_imports(["albumentations"]):
                self.native_train.train(hyp, opt, device, callbacks)
        finally:
            self.native_train.check_amp = original_check_amp
        self.ckpt_path = str(self.trainer.last)
        self.model = self.experimental.attempt_load(self.ckpt_path, device=device, fuse=False)
        self.trainer.history_path = fit / "canonical_history.csv"
        normalize_classic_history(fit / "results.csv", self.trainer.history_path)
        return self

    def predict(self, source, imgsz=640, device="cpu", conf=.001, iou=.7, max_det=300,
                half=False, quantize=None, nms=None, **_):
        import numpy as np
        import torch
        from PIL import Image
        from ultralytics.engine.results import Results
        if quantize is not None:
            if quantize not in {16, 32, "fp16", "fp32"}:
                raise ValueError("Classic inference supports FP16 or FP32.")
            half = quantize in {16, "fp16"}
        if nms is False:
            raise ValueError("Classic YOLOv5 uses external NMS.")
        target = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}" if str(device).isdigit() else device)
        self.predictor.device = target
        self.model.to(target)
        if not self._fused:
            self.model.fuse()
            self._fused = True
        self.model.half() if half else self.model.float()
        self.model.eval()

        def clock():
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            return time.perf_counter()

        start = clock()
        if isinstance(source, Image.Image):
            original = np.asarray(source.convert("RGB"))
            filename = "image"
        else:
            with Image.open(source) as image:
                original = np.asarray(image.convert("RGB"))
            filename = str(source)
        resized = self.augmentations.letterbox(original, new_shape=imgsz, auto=True,
                                               stride=int(self.model.stride.max()))[0]
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).to(target)
        tensor = (tensor.half() if half else tensor.float()) / 255.
        tensor = tensor.unsqueeze(0)
        prepared = clock()
        with torch.inference_mode():
            raw = self.model(tensor)
        inferred = clock()
        detections = self.general.non_max_suppression(raw, conf_thres=conf, iou_thres=iou, max_det=max_det)[0]
        if len(detections):
            detections[:, :4] = self.general.scale_boxes(tensor.shape[2:], detections[:, :4], original.shape)
        finished = clock()
        result = Results(orig_img=np.ascontiguousarray(original[:, :, ::-1]), path=filename,
                         names=self.names, boxes=detections,
                         speed={"preprocess": (prepared-start)*1000, "inference": (inferred-prepared)*1000,
                                "postprocess": (finished-inferred)*1000})
        return [result]

    def track(self, frame, persist=True, tracker="bytetrack.yaml", **kwargs):
        import numpy as np
        import torch
        from PIL import Image
        import ultralytics
        import yaml
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT
        if tracker not in {"bytetrack.yaml", "botsort.yaml"}:
            raise ValueError("Choose the supplied ByteTrack or BoT-SORT configuration.")
        if self._tracker is None or not persist or self._tracker_name != tracker:
            config = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/trackers" / tracker).read_text(encoding="utf-8"))
            # Default BoT-SORT has no ReID; native feature extraction is not advertised.
            if config.get("with_reid"):
                raise ValueError("Classic YOLOv5 tracking supports BoT-SORT with_reid=False.")
            self._tracker = (BYTETracker if config["tracker_type"] == "bytetrack" else BOTSORT)(SimpleNamespace(**config))
            self._tracker_name = tracker
        result = self.predict(Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])), **kwargs)[0]
        tracks = self._tracker.update(result.boxes.cpu().numpy(), result.orig_img)
        if len(tracks):
            result = result[tracks[:, -1].astype(int)]
            result.update(boxes=torch.as_tensor(tracks[:, :-1], device=result.boxes.data.device))
        elif any(not item.is_activated for item in self._tracker.tracked_stracks):
            result = result[:0]
        return [result]

    def export(self, format="onnx", imgsz=640, device="cpu", quantize=32, **kwargs):
        if format != "onnx":
            raise ValueError("The shared classic export path supports ONNX.")
        if quantize not in {None, 32, "fp32"}:
            raise ValueError("The shared export command produces FP32 ONNX.")
        native_export = importlib.import_module("export")
        paths = native_export.run(weights=self.ckpt_path, imgsz=(imgsz, imgsz), batch_size=1,
                                  include=("onnx",), device=str(device), half=False, **kwargs)
        outputs = [Path(path) for path in paths if str(path).endswith(".onnx")]
        if len(outputs) != 1 or not outputs[0].is_file():
            raise RuntimeError("The native ONNX exporter did not produce one model file.")
        return str(outputs[0])
