"""Generate explicitly separate resolution and online-augmentation experiments."""
import copy
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = json.loads((root / "configs/primary.json").read_text())
target = root / "configs/ablations"
target.mkdir(parents=True, exist_ok=True)
for size in [960, 1280]:
    config = copy.deepcopy(base)
    config["models"] = ["yolo11n"]
    config["training"]["imgsz"] = size
    (target / f"yolo11n_resolution_{size}.json").write_text(json.dumps(config, indent=2))
config = copy.deepcopy(base)
config["models"] = ["yolo11n"]
for key in ["degrees", "translate", "scale", "fliplr", "flipud"]:
    config["training"][key] = 0.
(target / "yolo11n_no_extra_online_geometry.json").write_text(json.dumps(config, indent=2))
print(f"Created three ablation configurations in {target}")
print("The 640-pixel comparator is the primary run; tiled inference reuses its checkpoint.")
print("Disabling online augmentation does not remove transformations already baked into the supplied export.")
