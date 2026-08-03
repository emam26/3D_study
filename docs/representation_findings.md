# Extended representation findings

These findings summarize the fixed ten-image pilots. They are an oracle study
of visible RGB-D geometry, not a trained segmentation benchmark.

| Decision | NYUv2 | SUN RGB-D |
| --- | --- | --- |
| Dense default | Voxel (96.4% full-label mIoU) | Voxel (60.3% full-label mIoU) |
| Boundary quality | Mesh (71.4% boundary F1) | Mesh (43.4% boundary F1) |
| Compact full coverage | Superpoint (972 elements) | Superpoint (904 elements) |
| Fast full coverage | Octree (0.46 s/image) | Octree (0.38 s/image) |

## NYUv2

- Highest full-label mIoU: **Voxel** (96.4%).
- Highest boundary F1: **Mesh** (71.4%).
- Smallest full-coverage representation: **Superpoint** (972 elements).
- Fastest full-coverage representation: **Octree** (0.46 seconds/image).

## SUN RGB-D

- Highest full-label mIoU: **Voxel** (60.3%).
- Highest boundary F1: **Mesh** (43.4%).
- Smallest full-coverage representation: **Superpoint** (904 elements).
- Fastest full-coverage representation: **Octree** (0.38 seconds/image).

## Interpretation

Point clouds, surfels, graphs, and descriptors use fixed point budgets in this
pilot. Their valid-depth mIoU can look perfect because it is measured only on
the few pixels they cover. The coverage-optimism plot and full-label mIoU expose
that limitation.

The most defensible first choice for dense RGB-D geometry in this pilot is the
voxel representation: it reaches full coverage with the highest full-label
mIoU on both datasets. Octrees are the compact alternative; meshes preserve
boundaries best but are much more expensive. These are pilot findings, not
universal rankings.
