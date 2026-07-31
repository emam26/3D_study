# V1 RGB SegFormer results

V1 is the RGB-only control configuration in the 26-version architecture
screening study. It uses a pretrained SegFormer-B0, one full epoch, batch size
4, 480 x 480 inputs, and the common evaluation code.

## Reported full-run results

| Dataset | Train / validation samples | Mean train loss | Validation mIoU | Pixel accuracy | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| NYUv2 | 795 / 654 | 2.718368 | 0.093332 | 0.522884 | Completed |
| SUN RGB-D | 8,268 / 2,067 (current 80/20 screening split) | - | - | - | Pending full run |

The NYUv2 values above are from the completed one-epoch run reported from the
WSL GPU environment. Per-class IoU is saved in that run's `metrics.json`; it is
not reproduced in this summary until the file is copied or regenerated locally.

The SUN RGB-D row is intentionally not filled with smoke-test values. A
two-sample smoke run verifies the pipeline only; it is not evidence for the
full-dataset comparison.

## Run both datasets

From the repository root, execute the full protocol once:

```bash
python run_v1_all.py
```

The wrapper runs training and validation for both datasets, verifies the
checkpoint, metrics JSON, and qualitative grid, and writes a combined summary
to `outputs/segmentation/v1_run_summary.json`.

## Required artifacts per dataset

```text
outputs/segmentation/v1_rgb_segformer_<dataset>/
├── v1_rgb_segformer.pth
├── metrics.json
└── <dataset>_qualitative_grid.png
```

> **Observation:** The first V1 result is a control measurement, not a final
> segmentation claim. Every V2-V26 result must use the same protocol and report
> both quantitative metrics and the qualitative grid before it is compared with
> V1.
