# 3D Representation Oracle Study

This is a standalone research project for comparing deterministic visible RGB-D
representations on NYUv2 and SUN RGB-D.

## Cat3D 3D representation atlas

<p align="center">
  <img src="docs/figures/cat3d_original_plus_9_grid.png" alt="Cat3D original model plus nine 3D representations" width="100%">
</p>

The atlas starts with the original Cat OBJ/CAD model and then shows its nine
derived 3D representations.

## Visual overview

The main generated figures are included here so the README starts with the
actual representation results. The 2D grids show camera-space projections;
the 3D grids show the raw geometric structures.

| Dataset | 2D representation grid | 3D representation grid |
| --- | --- | --- |
| NYUv2 | <img src="docs/figures/nyuv2_2d_representation_grid.png" alt="NYUv2 2D representation grid" width="420"> | <img src="docs/figures/nyuv2_3d_representation_grid.png" alt="NYUv2 3D representation grid" width="420"> |
| SUN RGB-D | <img src="docs/figures/sunrgbd_2d_representation_grid.png" alt="SUN RGB-D 2D representation grid" width="420"> | <img src="docs/figures/sunrgbd_3d_representation_grid.png" alt="SUN RGB-D 3D representation grid" width="420"> |
| Cat3D OBJ | - | <img src="docs/figures/cat3d_original_plus_9_grid.png" alt="Cat3D original model plus nine representations" width="420"> |

The Cat3D figure contains the original OBJ/CAD model plus nine derived 3D
representations. The separate `catDog.png` RGB scene is intentionally not
included in the output atlas.

## Additional analysis

The pilot comparison below reports valid-depth mIoU, coverage, full-label mIoU,
and construction time for all nine representations on both RGB-D datasets.

<p align="center">
  <img src="docs/figures/representation_benchmark.png" alt="NYUv2 and SUN RGB-D representation benchmark" width="100%">
</p>

For Cat3D, the additional figures show orthographic and perspective
projections, depth and descriptor maps, surface normals, representation sizes,
and the local descriptor distribution.

| Cat3D projection analysis | Cat3D statistics |
| --- | --- |
| <img src="docs/figures/cat3d_projection_analysis.png" alt="Cat3D projection and geometric analysis" width="520"> | <img src="docs/figures/cat3d_statistics.png" alt="Cat3D representation statistics" width="520"> |

The study will compare organized point clouds, surfels, partial meshes, sparse
voxels, single-view sparse TSDFs, superpoint regions, kNN graphs, adaptive
octrees, local geometric descriptors, and deterministic multiview projections. Ground truth is used only for oracle labeling and
evaluation -- not to construct the primary geometry.

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

## Cat3D asset visualization

The local `data/cat3D/` folder contains a real Cat OBJ mesh with MTL and texture
files. The output atlas contains the original OBJ/CAD model plus the same nine
representations used by the main study: point cloud, surfel, mesh, voxel,
surface TSDF, superpoint regions, graph, octree, and descriptor.

```text
outputs/cat3d/cat3d_original_plus_9_grid.png
outputs/cat3d/cat3d_projection_analysis.png
outputs/cat3d/cat3d_statistics.png
```

The separate `data/catDog.png` RGB scene is not included in this 3D output,
because it is not a calibrated geometry source.
Use `notebooks/06_catdog_asset_grids.ipynb` to view the Cat3D atlas.

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
