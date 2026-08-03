# Core representation study protocol

This document defines the scientific boundaries and evaluation rules used by
the 10-image NYUv2 and 10-image SUN RGB-D pilot. For figures, results, and run
commands, start with the repository `README.md`.

## Scientific scope

This project measures how well deterministic representations preserve the
visible surface of one RGB-D frame. A single depth image is a partial 2.5D
observation; the study must not claim complete room reconstruction or invent
unseen geometry.

## Evaluation tracks

- Track A (`valid_depth`): valid semantic labels, valid depth, and representation coverage.
- Track B (`full_label`): all valid semantic labels; uncovered pixels count as missing.

Coverage is always reported next to Track A so a low-coverage representation
cannot appear strong by evaluating only its easiest pixels.

## Implementation status

1. **Complete:** common `RGBDSample` contract, coordinate conventions,
   NYUv2 validation, and synthetic integration tests.
2. **Complete:** backprojection, point cloud, voxel, mesh, oracle labels,
   z-buffer rendering, and coverage-aware semantic metrics.
3. **Complete:** surfels, single-view sparse TSDF, superpoints, kNN graphs,
   adaptive octrees, local geometric descriptors, and deterministic
   virtual-camera projections.
4. **Complete:** headless visualization panels, compressed map exports, and
   multi-sample aggregation for the fixed 10-image pilots.
5. **Complete:** rate-distortion tables, cross-dataset ranking, per-image
   variability, coverage-optimism analysis, and virtual-view robustness. Both
   NYUv2 and SUN RGB-D adapters are validated; no full-dataset run is planned.

## Coordinate convention

Pixels use `(v, u)` row/column indexing. Camera coordinates are right-handed
with X right, Y down, and Z forward, measured in metres. For intrinsics K:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
```

Every representation element must retain the original pixels that contributed
to it, enabling deterministic oracle labels and debugging of back-projection.
