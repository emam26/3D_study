# 3D Representation Oracle Study

This is a standalone research project for comparing deterministic visible RGB-D
representations on NYUv2 and SUN RGB-D.

The study will compare organized point clouds, surfels, partial meshes, sparse
voxels, single-view sparse TSDFs, superpoint regions, kNN graphs, adaptive
octrees, local geometric descriptors, and deterministic multiview projections. Ground truth is used only for oracle labeling and
evaluation—not to construct the primary geometry.

## Current status

The first deterministic pipeline is implemented and smoke-tested on NYUv2:
backprojection, organized point clouds, surfels, partial meshes, sparse
voxels, single-view sparse TSDFs, superpoint regions, oracle labels,
z-buffer rendering, coverage-aware semantic metrics, boundary metrics, kNN
graphs, adaptive octrees, local descriptors, and virtual-camera projections.
The fixed pilot study uses 10 images from each
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
one contact-sheet overview image for every sample.

The two raw-geometry grids are saved as:

```text
outputs/nyuv2/nyuv2_representation_smoke/3d_representation_grid.png
outputs/sunrgbd/sunrgbd_representation_smoke/3d_representation_grid.png
```

They can also be viewed together from
`notebooks/03_raw_3d_representation_grids.ipynb`.

The matching 2D oracle-projection grids are saved as:

```text
outputs/nyuv2/nyuv2_representation_smoke/2d_representation_grid.png
outputs/sunrgbd/sunrgbd_representation_smoke/2d_representation_grid.png
```

Use `notebooks/04_2d_representation_grids.ipynb` to view them together.
The camera, left/right oblique, and elevated projections for point clouds,
graphs, octrees, and descriptors are saved as:

```text
outputs/nyuv2/nyuv2_representation_smoke/projection_grid.png
outputs/sunrgbd/sunrgbd_representation_smoke/projection_grid.png
```

Use `notebooks/05_multiview_projection_grids.ipynb` to view those projections.
Use `--max-samples 1` for a quick check, `--sample-id 00001` for a specific
frame, and `--overwrite` to recompute an existing result.

## catDog asset visualization

The local `data/catDog.png` is a 2D RGB scene. A separate Cat OBJ asset is
also present under `data/cat3D/`. The following grids use the RGB scene as a
reference and compute true mesh, point-cloud, voxel, octree, graph, descriptor,
normal, and projection views from the OBJ asset:

```text
outputs/catdog/catdog_3d_representation_grid.png
outputs/catdog/catdog_2d_representation_grid.png
outputs/catdog/cat3d_9representation_grid.png
```

The 3D grids use the same nine representations as the main study: point cloud,
surfel, mesh, voxel, surface TSDF, superpoint regions, graph, octree, and
descriptor. The OBJ asset is not camera-aligned to the two cats in the RGB
scene, so the RGB grid is an asset comparison rather than a reconstruction of
that photograph. The `cat3d_9representation_grid.png` file contains only the
actual 3D Cat OBJ representations.
Use `notebooks/06_catdog_asset_grids.ipynb` to view both grids.

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
