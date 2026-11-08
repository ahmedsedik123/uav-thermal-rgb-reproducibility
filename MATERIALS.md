# Provenance and release status

## Relation to the earlier study

Farman et al., *Enhancing UAV-Based Human Detection in Thermal Imaging with Custom Dataset*, is Reference [12] in the supplied manuscript. The present image benchmark evaluates ten models under one documented protocol on separate thermal and visible RGB exports. It adds repeated training, held-out testing, target-size error analysis, and workstation resource measurements. The reported image results are traceable to the 60 current runs and the separate timing records.

The thermal source export includes a Roboflow notice dated 15 November 2023, with the dataset URL and a Public Domain designation. That notice is preserved with the images. The available records do not map specific images, source videos, or annotations to the earlier publication. No claim of a newly collected independent dataset relative to Reference [12] is made.

## Reviewer access

| Material | What has been recovered | Access condition |
|---|---|---|
| Images and annotations | Exact 182 thermal and 216 RGB exports and their annotations | Supply the image and annotation archive to authorized readers. A public redistribution notice for the RGB export was not supplied. |
| Splits and study settings | Exact recorded exploratory holdout assignments and protocol identifiers | Included in this package. |
| Code and environment | Notebook source, archived implementation, requirements, native YOLOv5 source pin, and recorded environments | Included. Preserve the upstream license and notices. No new license is asserted for the authors' code. |
| Initial pretrained weights | Ten files verified against recorded SHA-256 hashes | In the initial weights archive. Preserve the original model terms. |
| Selected trained checkpoints | 60 indexed files, approximately 8.05 GB in total | Existing Drive locations are author-controlled. This package does not grant access to those files. |
| Training, evaluation, and timing logs | 60 training histories, training and evaluation logs, 20 timing logs, and run-level metadata | Included. Original absolute machine paths are redacted in the release copies. |
| Video and tracker outputs | Supplied tracking code, annotated MP4, frame and track CSVs, summary, and the corresponding archived `fit/weights/best.pt` | [Tracking release v1.1.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.1.0). The unannotated source clip and identity ground truth are not supplied. Public redistribution clearance for the annotated video is not documented in the supplied materials. |

The source package was recovered from the supplied project archive. It contains optional workflows beyond the reported image benchmark; the completion records specify which experiments actually ran. Full execution-time provenance of every source file is not established by the runtime logs. For the video run, the supplied code calls `bytetrack.yaml`; the exact package versions and resolved tracker configuration were not recorded. The tracking checkpoint is the recovered file corresponding to the path in the supplied summary.

## Dataset counts

The archived assignments and image archive support 124/27/31 thermal and 152/32/32 RGB train/validation/test images. Table 3 of the supplied manuscript lists ten times as many images and human boxes in every split. See [the comparison with the manuscript](PAPER_NUMBERS.md#dataset-partitions) for the recorded counts and the values printed in the paper. The recorded splits group image families and identical pixels; they are not verified independent-flight splits.
