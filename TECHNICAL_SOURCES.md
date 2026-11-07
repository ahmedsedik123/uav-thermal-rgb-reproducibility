# Technical sources for the implementation

- Existing source project, initial weights, source images and labels: the authors' full UAV project package and DroneResults. Input bytes and model identities are checked in `validation`.
- Ultralytics training interface: https://docs.ultralytics.com/modes/train/
- Ultralytics detection dataset format: https://docs.ultralytics.com/datasets/detect/
- Matching Torch and Torchvision 2.7.0/0.22.0 builds: https://pytorch.org/get-started/previous-versions/
- Original anchor-based YOLOv5 source: https://github.com/ultralytics/yolov5 at the commit in `third_party/yolov5/SOURCE_PIN.json`. Its original source/license and checksums remain bundled.

The registered fixed-weight input blend is an explicitly defined baseline implemented here, not a claimed novel architecture or a reproduced learned feature-fusion method. Real fusion data and experimental results were not present in the supplied material.
