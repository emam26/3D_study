# 3D Representation Oracle Study

This is a standalone research project for comparing deterministic visible RGB-D
representations on NYUv2 and SUN RGB-D. It is not a segmentation model and does
not train a neural network.

The study will compare organized point clouds, surfels, partial meshes, sparse
voxels, single-view sparse TSDFs, superpoint graphs, and deterministic
multiview projections. Ground truth is used only for oracle labeling and
evaluation—not to construct the primary geometry.

## Current status

The first deterministic pipeline is implemented and smoke-tested on NYUv2:
backprojection, organized point clouds, surfels, partial meshes, sparse
voxels, single-view sparse TSDFs, superpoint regions, oracle labels,
z-buffer rendering, coverage-aware semantic metrics, boundary metrics, and
virtual-camera projections. The fixed pilot study uses 10 images from each
dataset, with no full-dataset training or sweep.

## Run the study

```bash
# From this repository root
python -m repstudy.validate_dataset --config configs/nyuv2_smoke.yaml
python -m repstudy.run_study --config configs/nyuv2_smoke.yaml
python -m repstudy.validate_dataset --config configs/sunrgbd_smoke.yaml
python -m repstudy.run_study --config configs/sunrgbd_smoke.yaml
```

Each configuration processes 10 images. The runner writes one directory per
sample plus `study_summary.json` and `study_summary.csv`.
Afterward, open `notebooks/02_batch_visualization_overview.ipynb` to create
one six-panel overview image for every sample.
Use `--max-samples 1` for a quick check, `--sample-id 00001` for a specific
frame, and `--overwrite` to recompute an existing result.

## How to read the first results

`valid_depth` mIoU evaluates only pixels with valid depth and representation
coverage. `full_label` mIoU counts uncovered labeled pixels as missing. Always
read mIoU together with `valid_depth_coverage` and `valid_depth_missing_rate`;
an identity-like point sample can have high covered-pixel accuracy while still
covering only a small fraction of the image.

## Learning note

Each representation keeps the source image pixels that contributed to each
3D element. That bookkeeping is what makes deterministic oracle labeling and
the missing-coverage measurement possible; no neural network is trained in
this study milestone.

Generated data, datasets, credentials, and checkpoints will remain outside Git.
