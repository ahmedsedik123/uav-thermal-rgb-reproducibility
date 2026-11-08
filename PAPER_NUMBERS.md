# Manuscript numbers and archived evidence

This comparison covers the supplied revision of *Deep Learning for UAV-Based Human Detection: A Unified Evaluation of Detection Accuracy, Computational Performance, and Temporal Tracking*. The image benchmark and thermal video demonstration are separate experiments.

## Dataset partitions

The recorded input archive contains **182 thermal images (1,690 boxes)** and **216 RGB images (2,924 boxes)**. The [recorded split counts](records/split_counts.csv) agree with the assignment files and the totals in the image and annotation archive. Table 3 in the supplied manuscript prints counts that are ten times the archived counts for every split:

| Cohort | Split | Images in Table 3 | Archived images | Boxes in Table 3 | Archived boxes |
|---|---|---:|---:|---:|---:|
| Thermal | Training | 1,240 | 124 | 11,320 | 1,132 |
| Thermal | Validation | 270 | 27 | 2,220 | 222 |
| Thermal | Test | 310 | 31 | 3,360 | 336 |
| RGB | Training | 1,520 | 152 | 20,270 | 2,027 |
| RGB | Validation | 320 | 32 | 4,620 | 462 |
| RGB | Test | 320 | 32 | 4,350 | 435 |

Use the archived counts for reproduction. Section 4.2 already reports the archived test totals of 336 thermal and 435 RGB human boxes. The assignments group related images and identical pixels; they do not identify verified flight sessions. The acquisition figures in Table 2 and the raw-frame extraction and annotation agreement claims in Section 3.1 cannot be checked from these exports. They need source flight, extraction and annotation records.

## Protocol and performance

The run records confirm 60 runs: ten detectors, two cohorts and three seeds (17, 42 and 93), with 100 epochs per run. Table 5's common recipe is documented in the [thermal configuration](records/thermal/report/cohort_definition.json) and [RGB configuration](records/rgb/report/cohort_definition.json). All archived histories contain 100 epoch rows. The 20 RTX A6000 profiling sessions used seed-17 checkpoints, FP32, batch size 1, 50 warm-up iterations and approximately 60 seconds of measurement per detector and cohort.

| Cohort | Highest mean test AP50:95 (mean ± sample SD) | Highest workstation throughput |
|---|---|---|
| Thermal | YOLOv8m: 0.5947 ± 0.0044 | YOLOv5su: 139.47 images/s |
| RGB | YOLO12n: 0.4217 ± 0.0117 | YOLOv10n: 91.36 images/s |

The parameter and operation reductions in the abstract compare trained YOLOv8m with trained YOLOv8l: 40.7% fewer parameters and 52.2% fewer profiled operations. The fastest measured throughput and the highest mean test AP50:95 need not come from the same detector.

The following 507 numerical entries agree with the archived results at the precision printed in the manuscript. A mean and its standard deviation count as separate entries.

| Manuscript table | Entries checked | Evidence |
|---|---:|---|
| 4: Model complexity | 20 | Trained single-class parameter and GFLOP fields in each archived run record |
| 7: Thermal test metrics | 100 | [Thermal mean and sample SD](records/thermal/report/model_comparison_mean_SD.csv) |
| 8: RGB test metrics | 100 | [RGB mean and sample SD](records/rgb/report/model_comparison_mean_SD.csv) |
| 9: Small-target AP and errors | 80 | Aggregated [thermal](records/thermal/report/test_results_by_seed.csv) and [RGB](records/rgb/report/test_results_by_seed.csv) per-seed results |
| 10: Training wall time | 40 | Mean and sample SD from the three archived run records per detector and cohort |
| 11: Accuracy and throughput frontier | 36 | Cohort accuracy, runtime and trained model complexity records |
| 12: Thermal workstation profiling | 60 | [Thermal measurements](records/thermal/report/runtime_measurements.csv) |
| 13: RGB workstation profiling | 60 | [RGB measurements](records/rgb/report/runtime_measurements.csv) |
| 14: Thermal video tracking | 11 | [Checked tracking outputs](tracking/outputs/audit_summary.json); percentages derived from counts |

The per-run `run.json` files, histories and partition assignments are in `UAV_Code_Splits_and_Logs.zip` in [image benchmark v1.0.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.0.0). [Tracking release v1.1.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.1.0) includes the annotated thermal video and its original outputs. The 26 tracking IDs are algorithm assignments; the records do not establish the number of unique people.
