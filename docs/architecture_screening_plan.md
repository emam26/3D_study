# 26-Version Architecture Screening Study

This is a separate learned-model study inside the independent RGB-D project.
Its purpose is to determine which encoder, depth strategy, 3D representation,
decoder, fusion mechanism, loss, and training mechanism are worth studying in
depth for indoor semantic segmentation.

The deterministic representation study remains the primary 3D analysis. These
26 versions are a controlled screening track, not 26 final research claims.

## Common protocol

- **Datasets:** NYUv2 and SUN RGB-D.
- **Input:** 480 x 480 training/evaluation tensors; RGB and depth remain aligned.
- **Training budget:** one full epoch, batch size 4, fixed seed 42.
- **Metrics:** mIoU, pixel accuracy, per-class IoU, parameter count, runtime,
  and a qualitative RGB/ground-truth/prediction/error grid.
- **Comparison plots:** metric heatmap, accuracy-versus-parameter Pareto plot,
  and fixed-sample qualitative comparison grids.
- **Fairness rule:** change one primary architectural factor at a time whenever
  possible; keep the split, image size, batch size, and evaluation code fixed.
- **Interpretation:** one epoch ranks candidates quickly; it is not a final
  accuracy benchmark.

## Version matrix

| Version | Primary change | Main components | Status |
| --- | --- | --- | --- |
| V1 | RGB control | SegFormer-B0, pretrained encoder/decoder | **Implemented; NYUv2 run recorded** |
| V2 | RGB encoder | ResNet-50 + DeepLabV3+ head | Planned |
| V3 | RGB encoder | Swin-T + UPerNet head | Planned |
| V4 | RGB foundation encoder | DINOv2 ViT-B + DPT decoder | Planned |
| V5 | RGB foundation encoder | DINOv3 ViT-B/16 + DPT decoder | Planned |
| V6 | RGB foundation encoder | DINOv3 ConvNeXt-B + UPerNet | Planned |
| V7 | Lightweight RGB encoder | ConvNeXt-T + FPN decoder | Planned |
| V8 | Vision-language encoder | CLIP ViT-B/16 + FPN decoder | Planned |
| V9 | Depth handling | RGB + normalized raw depth, early fusion | Planned |
| V10 | Depth handling | RGB + inverse-depth channel | Planned |
| V11 | Depth handling | RGB + HHA/geocentric depth encoding | Planned |
| V12 | Depth handling | RGB + depth-validity/confidence channel | Planned |
| V13 | Depth handling | RGB encoder + learned depth encoder, late fusion | Planned |
| V14 | Depth handling | RGB + monocular/pseudo-depth branch | Planned |
| V15 | 3D representation | Visible point-cloud branch | Planned |
| V16 | 3D representation | Surfel branch with normal and color attributes | Planned |
| V17 | 3D representation | Sparse voxel branch | Planned |
| V18 | 3D representation | Surface-TSDF branch | Planned |
| V19 | 3D representation | kNN graph + graph-attention branch | Planned |
| V20 | 3D representation | Adaptive octree branch | Planned |
| V21 | Decoder | Mask2Former-style mask decoder | Planned |
| V22 | Decoder | DPT-style multi-scale decoder | Planned |
| V23 | Decoder | UPerNet/FPN pyramid decoder | Planned |
| V24 | Fusion mechanism | Confidence-gated cross-modal fusion | Planned |
| V25 | Objective | Cross-entropy + Dice + boundary loss | Planned |
| V26 | Training mechanism | Uncertainty-weighted consistency/EMA training | Planned |

## V1 completion record

V1 is the RGB-only control: the aligned depth map is loaded only for sample
alignment and is not provided to the model. It is implemented in
`train_v1_rgb_segformer.py` with separate NYUv2 and SUN RGB-D configurations.
Each run writes a checkpoint, `metrics.json`, and a qualitative grid under
`outputs/segmentation/<experiment>/`. The checkpoint-only visualization utility
is `visualize_v1_segformer.py`.

The next version should be added only after its implementation, smoke test,
full one-epoch run, visualization, documentation, and separate Git commit are
complete.

> **Study observation:** V1 establishes how far RGB alone can go. A later depth
> or 3D version is meaningful only if it improves the matched V1 baseline under
> the same protocol and reports both quantitative and qualitative evidence.
