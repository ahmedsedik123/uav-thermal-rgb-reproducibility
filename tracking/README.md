# Thermal UAV tracking and counting

The [tracking release v1.1.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.1.0) archives the supplied video experiment separately from the [image benchmark v1.0.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.0.0). Both releases require access to this private repository.

| File | Contents |
|---|---|
| `original_tracking_code.py` | Original script, preserved byte for byte. It retains the original workstation paths. |
| `run_tracking.py` | Portable runner with explicit source video, checkpoint and output paths. |
| `experiment.json` | Recorded parameters, artifact hashes and provenance. |
| `outputs/counting_results.csv` | Counts for every frame, including frames with zero tracks. |
| `outputs/tracking_details.csv` | Box, confidence, center and track ID for each observation. |
| `outputs/tracking_summary.json` | Original run summary. |
| `outputs/audit_summary.json` | Counts independently checked against the original outputs, plus a video check. |
| `tracked_video.mp4` | Release asset: supplied annotated output video. |
| `Thermal_YOLOv8m_seed17_tracking_best.pt` | Release asset: the archived `fit/weights/best.pt` corresponding to the path in the tracking summary. |

## Recorded experiment

Thermal YOLOv8m, training seed 17, class 0 (`item` in the one-class dataset), ByteTrack (`bytetrack.yaml`), input size 640, confidence 0.25, IoU 0.50. The horizontal counting line is at half the frame height, with an 8-pixel margin and a maximum 10-frame observation gap. These settings come from the supplied script and summary. The exact software versions and resolved ByteTrack YAML of that run were not recorded.

The archive has 2,611 frames at 25 FPS (104.44 s), 288 frames with assigned tracks, 330 tracked-box observations, 26 distinct track IDs and no line crossings. The mean and maximum visible track counts are 0.126 and 3. The last active frame begins at 44.04 s, and 11 IDs were observed at most twice. Track IDs are algorithm outputs; no identity ground truth was supplied to establish the number of unique people or tracking accuracy.

The recovered checkpoint matches the recorded archived path. The original tracking summary did not capture a checksum, so the manifest fingerprints the recovered copy. The `Thermal_YOLOv8m_seed17.pt` asset in the image release is a different file (`best_mAP50_95.pt`); use the tracking checkpoint for this runner.

## Verify and run locally

From the repository root, audit the recorded outputs with Python 3.11. Add the downloaded annotated MP4 to the audit to check its hash and decode every frame (`ffprobe` must be on PATH).

```sh
python tracking/audit_outputs.py
python tracking/audit_outputs.py --video path/to/tracked_video.mp4
```

To run inference, use the repository's Anaconda/Python environment and its documented PyTorch setup, install the `lap` dependency used by tracking, and supply the original unannotated clip plus the tracking checkpoint. In JupyterLab, the same command can be called with `subprocess.run` using the notebook kernel's Python executable.

```sh
python -m pip install "lap>=0.5.12"
python tracking/run_tracking.py --video path/to/test_video.mp4 --checkpoint path/to/Thermal_YOLOv8m_seed17_tracking_best.pt --output-dir tracking_runs/thermal_seed17
```

Use `--device cpu` for CPU inference or `--device 0` to request the first GPU. Choose a new or empty output directory. The runner keeps the archived parameters, records hashes of the actual inputs and logs package versions for new runs. The historical `test_video.mp4` is not in the release. The annotated result is for review and should not be used as the detector input.
