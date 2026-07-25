# Small-sample RGB-D experiments

This note documents the follow-up experiments that extend the deterministic
representation study. They use the same paired pilot sample as the main
analysis: ten NYUv2 images and ten SUN RGB-D images. No full-dataset training
is performed, and no learned model is fitted.

## Protocol

The runner is:

```bash
python -m repstudy.run_small_sample_experiments
```

All settings reuse the baseline per-sample geometry records. The controlled
variants change one resolution or budget at a time. Ground-truth labels are
used for evaluation and for selecting the adaptive threshold on the tuning
half only; they are not used to construct a representation or to choose a
representation for a test image.

## 1. Rate-distortion pilot

The nine styles are evaluated at lower, baseline, and/or higher resolutions:

- point cloud, surfel, and descriptor point budgets: 1k, 5k, and 10k;
- graph point budgets: 1k, 2.5k, and 5k;
- voxel and TSDF voxel sizes: coarse, baseline, and fine;
- mesh depth-jump threshold: coarse and baseline;
- superpoint region budgets: 500, 1k, and 2k;
- octree leaf limits: 256, 128, and 64 points.

The reported rate is the mean number of geometry elements per image. The
distortion proxy is full-label mIoU, with error bars showing image-to-image
variation. In this pilot, NYUv2 voxel mIoU rises from 92.1% to 96.4% to 98.6%
as the mean element count rises from 6.8k to 21.9k to 62.6k. SUN RGB-D rises
from 58.9% to 60.3% to 61.2%. This supports treating resolution as an explicit
experimental factor rather than comparing only one default setting.

## 2. Depth-corruption robustness

Voxel, octree, and superpoint representations are recomputed under:

- clean depth;
- 10% random depth dropout;
- 30% random depth dropout;
- Gaussian depth noise with 1 cm standard deviation;
- Gaussian depth noise with 3 cm standard deviation.

The corruption masks and noise are deterministic per dataset and sample. At
30% dropout, mean full-label mIoU decreases to 67.8%, 65.6%, and 60.6% for
NYUv2 voxel, octree, and superpoint representations, respectively. SUN RGB-D
decreases to 42.3%, 41.1%, and 38.5%. Noise is less harmful than dropout here
because the metric is computed relative to the remaining valid-depth support.
Consequently, the apparently unchanged 100% coverage for dense styles should
not be interpreted as recovery of the removed measurements.

## 3. Adaptive hybrid pilot

The hybrid selector uses only input geometry and RGB information:

```text
edge score = 0.65 * normalized depth edge + 0.35 * normalized RGB edge
high edge score  -> mesh
interior         -> voxel
fallback         -> octree
```

Candidate edge quantiles 0.60, 0.75, and 0.90 are scored on the first five
images of each dataset. The selected quantile is then frozen and evaluated on
the last five images. In the pilot, q=0.60 is selected for both datasets. The
held-out hybrid reaches 98.2% full-label mIoU on NYUv2 versus 96.8% for voxel,
and 65.6% on SUN RGB-D versus 64.9% for voxel. These are encouraging paired
pilot measurements, not a universal claim: the held-out split has only five
images per dataset.

## 4. Efficiency proxy

The efficiency table records construction time and a lower-bound storage proxy
for XYZ coordinates (`3 * float32` bytes per geometric element). It is useful
for comparing scale, but it is not a complete serialized-size benchmark:
adjacency, normals, colors, labels, hierarchy metadata, and compression are
not included in that proxy.

## Outputs

The runner writes the following reproducible tables and figures under
`outputs/experiments/`:

```text
rate_distortion_metrics.csv
depth_corruption_metrics.csv
adaptive_hybrid_metrics.csv
representation_efficiency.csv
rate_distortion.png
depth_corruption_robustness.png
adaptive_hybrid_results.png
representation_efficiency.png
small_sample_experiments.md
```

Tracked copies of the figures are stored in `docs/figures/` so that the
README remains readable on GitHub. The study remains an oracle analysis of
visible RGB-D geometry; a learned segmentation benchmark is a separate future
experiment.
