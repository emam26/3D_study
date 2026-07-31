# V1 RGB SegFormer results

V1 is the RGB-only control configuration in the 26-version architecture
screening study. The current screening protocol uses a pretrained SegFormer-B0,
five full epochs, batch size 4, 480 x 480 inputs, and the common evaluation
code.

## Historical one-epoch result

| Dataset | Train / validation samples | Mean train loss | Validation mIoU | Pixel accuracy | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| NYUv2 | 795 / 654 | 2.718368 | 0.093332 | 0.522884 | Completed |

The NYUv2 values above are from the completed one-epoch run reported from the
WSL GPU environment. Per-class IoU is saved in that run's `metrics.json`; it is
not reproduced in this summary until the file is copied or regenerated locally.

## Current five-epoch screening result

| Dataset | Train / validation samples | Best epoch | Best mIoU | Pixel accuracy | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| NYUv2 | 795 / 654 | - | - | - | Pending full run |
| SUN RGB-D | 8,268 / 2,067 (current 80/20 screening split) | - | - | - | Pending full run |

The historical NYUv2 values above are from the one-epoch run reported from the
WSL GPU environment. The two-sample smoke run verifies the pipeline only and
is not evidence for the full-dataset comparison.

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
outputs/segmentation/v1_rgb_segformer_5ep_<dataset>/
├── v1_rgb_segformer.pth
├── metrics.json
├── history.json
└── <dataset>_qualitative_grid.png
```

> **Observation:** The first V1 result is a control measurement, not a final
> segmentation claim. Every V2-V26 result must use the same protocol and report
> both quantitative metrics and the qualitative grid before it is compared with
> V1.
