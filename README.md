# UAV thermal and RGB human detection: reproducibility materials

This repository documents the image benchmark and the separate video tracking experiment in *Deep Learning for UAV-Based Human Detection: A Unified Evaluation of Detection Accuracy, Computational Performance, and Temporal Tracking*, by Haleem Farman, Ahmed Sedik, and Moustafa M. Nasralla.

## Reported scope

The study compared ten detectors on separate, unpaired thermal and RGB image cohorts. For each detector and cohort, seeds 17, 42 and 93 were run for 100 epochs each (60 training runs in total). Workstation profiling used seed 17 for one session per detector and cohort (20 sessions on an NVIDIA RTX A6000). Paired fusion and generalization to independent flights were not evaluated.

| Cohort | Highest mean test AP50:95 (mean ± sample SD) | Highest workstation throughput |
|---|---|---|
| Thermal | YOLOv8m: 0.5947 ± 0.0044 | YOLOv5su: 139.47 images/s |
| RGB | YOLO12n: 0.4217 ± 0.0117 | YOLOv10n: 91.36 images/s |

The YOLOv8m and ByteTrack thermal video experiment processed 2,611 frames (104.44 s at 25 FPS). It produced 330 tracked box observations and 26 distinct track IDs, which are not a count of unique people. See [the tracking records](tracking/README.md) for the full Table 14 comparison.

## Dataset partitions (manuscript Table 3)

The table below reproduces the dataset counts reported in Table 3, **Experimental partitions and image-family count**.

| Cohort | Subset | Images | Human boxes |
|---|---|---:|---:|
| Thermal | Training | 1,240 | 11,320 |
| Thermal | Validation | 270 | 2,220 |
| Thermal | Test | 310 | 3,360 |
| RGB | Training | 1,520 | 20,270 |
| RGB | Validation | 320 | 4,620 |
| RGB | Test | 320 | 4,350 |

The Table 3 entries total **1,820 thermal images (16,900 human boxes)** and **2,160 RGB images (29,240 human boxes)**.

**Downloadable archive.** The release contains **182 thermal images (1,690 human boxes)** and **216 RGB images (2,924 human boxes)** verified against the archived split assignments. The [split-by-split comparison](PAPER_NUMBERS.md#dataset-partitions) shows how the archived counts differ from Table 3. Use the archived counts to reproduce the benchmark.

## Files and access

Access to GitHub release assets follows the repository's visibility. The additional checkpoint files indexed on Drive require separate authorization.

The [image benchmark release v1.0.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.0.0) contains the original code and records archive, images, initial weights, and selected image benchmark checkpoints. Its full records archive contains all 60 training histories and logs.

The [tracking release v1.1.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.1.0) adds the supplied tracking code, annotated MP4, per-frame counts, track details, summary, and the archived checkpoint corresponding to the tracking script. See [the tracking instructions](tracking/README.md).

- **Code and setup:** the notebook, `uav_review/`, recorded package requirements, environment file, and pinned native YOLOv5 source. The notebook source is retained; execution output is supplied separately in the run records.
- **Images and annotations:** extract `UAV_Images_and_Annotations.zip` into the project root. It contains the exact 398 input images, their original annotations, the source notices and dataset settings. Extract `UAV_Initial_Weights.zip` into the same folder to add the ten recorded initialization checkpoints. The archived bytes match the study input manifest and the recorded initial-weight hashes.
- **Splits:** within `UAV_Code_Splits_and_Logs.zip`, `records/thermal/protocol/` and `records/rgb/protocol/` contain the recorded train, validation, and test assignments and protocol metadata.
- **Logs and results:** the same archive contains resolved training arguments, selected epochs, 100-epoch histories, console logs, evaluation logs, and available timing logs under `records/{thermal,rgb}/runs/`. Per-seed results and summary tables are in the corresponding `report/` folders in the archive and on GitHub. Local machine path prefixes are redacted in these record copies; the original and release file digests are listed.
- **Trained checkpoints:** `records/run_checkpoint_index.csv` identifies the 60 validation-selected checkpoint files, their expected SHA-256 hashes, byte sizes, and existing author-controlled Drive locations. Three image benchmark checkpoints are included in v1.0.0; the index locates the other files. Drive permissions are unchanged. Obtain access from the data owner and verify each checksum after download.
- **Video experiment:** `tracking/` holds the original code, portable runner, run settings and verified per-frame outputs. The annotated result contains 2,611 frames at 25 FPS and 26 distinct track IDs. The original unannotated input clip is not included.

See [MATERIALS.md](MATERIALS.md) for provenance and access details.

## Run locally in Anaconda/JupyterLab

1. Extract `UAV_Code_Splits_and_Logs.zip`, `UAV_Images_and_Annotations.zip` and `UAV_Initial_Weights.zip` from v1.0.0 into the same project directory, then run `python verify_materials.py`. The script checks the archived input bytes and their hashes, the split counts, and the number of run records. It does not execute model files.
2. Create the environment with `conda env create -f environment.yml`, activate the environment named in that file, and use the matching PyTorch and torchvision builds for your device. The recorded run used Python 3.11.5, PyTorch 2.7.0+cu118, torchvision 0.22.0+cu118, Ultralytics 8.4.183, and CUDA 11.8 on Windows 10.
3. Start JupyterLab from the project directory and open `UAV_Revision_Study.ipynb`. Run the setup and dataset audit cells first. Keep the two cohorts separate, and check regenerated assignments against the recorded split manifests before training.
4. The notebook's full training stage starts 60 jobs when enabled. Existing metrics can be audited directly from the supplied CSVs without retraining. Run records are historical evidence, not executable configuration paths.
