# UAV thermal and RGB human detection: reproducibility materials

This package documents the image benchmark in *A Real-Time UAV Edge-Intelligence Framework for Human Detection in Thermal Imagery Using Advanced Deep Learning Techniques*, by Haleem Farman, Ahmed Sedik, and Moustafa M. Nasralla. This private repository accompanies the study. Obtain repository and asset access before citing it as available to readers.

## Reported scope

Ten models were trained for 100 epochs with seeds 17, 42, and 93 on each of two separate, unpaired image cohorts. The 60 run records support the image results; 20 workstation timing sessions used the seed-17 checkpoints. Fusion was not evaluated. The split records group image families and identical pixels; acquisition sessions were not verified.

| Cohort | Training | Validation | Test | Total images | Human boxes |
|---|---:|---:|---:|---:|---:|
| Thermal | 124 | 27 | 31 | 182 | 1,690 |
| RGB | 152 | 32 | 32 | 216 | 2,924 |

## Files and access

The [releases page](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases) contains the full code and records archive plus the data and initial weights. Download and extract the full code and records archive first; it contains all 60 training histories and logs.

- **Code and setup:** the notebook, `uav_review/`, recorded package requirements, environment file, and pinned native YOLOv5 source. The notebook source is retained; execution output is supplied separately in the run records.
- **Images and annotations:** extract `UAV_Images_and_Annotations.zip` into the project root. It contains the exact 398 input images, their original annotations, the source notices and dataset settings. Extract `UAV_Initial_Weights.zip` into the same folder to add the ten recorded initialization checkpoints. The archived bytes match the study input manifest and the recorded initial-weight hashes.
- **Splits:** `records/thermal/protocol/` and `records/rgb/protocol/` contain the recorded train, validation, and test assignments and protocol metadata.
- **Logs and results:** `records/{thermal,rgb}/runs/` contains resolved training arguments, selected epochs, 100-epoch histories, console logs, evaluation logs, and available timing logs. Per-seed results and summary tables are in the corresponding `report/` folders. Local machine path prefixes are redacted in these record copies; the original and release file digests are listed.
- **Trained checkpoints:** `records/run_checkpoint_index.csv` identifies the 60 validation-selected checkpoint files, their expected SHA-256 hashes, byte sizes, and existing author-controlled Drive locations. These trained binaries are not bundled. Drive permissions are unchanged. Obtain access from the data owner and verify each checksum after download.
- **Video experiment:** the 2,611-frame thermal video, its exact tracker configuration, and per-frame outputs for Section 4.5 were not found in the archived image benchmark. This package does not document a reproduced video result.

See [MATERIALS.md](MATERIALS.md) for provenance and access details.

## Run locally in Anaconda/JupyterLab

1. Extract both input archives beside this README and run `python verify_materials.py`. The script checks the archived input bytes and their hashes, the split counts, and the number of run records. It does not execute model files.
2. Create the environment with `conda env create -f environment.yml`, activate the environment named in that file, and use the matching PyTorch and torchvision builds for your device. The recorded run used Python 3.11.5, PyTorch 2.7.0+cu118, torchvision 0.22.0+cu118, Ultralytics 8.4.183, and CUDA 11.8 on Windows 10.
3. Start JupyterLab from the project directory and open `UAV_Revision_Study.ipynb`. Run the setup and dataset audit cells first. Keep the two cohorts separate, and check regenerated assignments against the recorded split manifests before training.
4. The notebook's full training stage starts 60 jobs when enabled. Existing metrics can be audited directly from the supplied CSVs without retraining. Run records are historical evidence, not executable configuration paths.
