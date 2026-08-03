# Independent RGB-D Representation Study

This is a standalone research project for studying how RGB-D scene information
can be represented and evaluated on NYUv2 and SUN RGB-D. It is intentionally
separate from the other model-development repositories.

## Project map

The repository contains two clearly separated tracks:

| Track | Main question | Current status |
| --- | --- | --- |
| Deterministic 3D representation study | Which visible RGB-D representation gives the best geometry, coverage, boundary quality, and efficiency? | Main 10-image-per-dataset analysis completed |
| 26-version neural architecture screening | Which encoder, depth strategy, 3D representation, decoder, loss, and training mechanism is most useful for semantic segmentation? | V1 implemented; V2-V26 planned |

The neural track does not replace the representation study. It provides a
controlled semantic-segmentation benchmark for later representation variants.
The complete version matrix is documented in
[`docs/architecture_screening_plan.md`](docs/architecture_screening_plan.md).

## Contents

- [At a glance](#at-a-glance)
- [Neural segmentation screening track](#neural-segmentation-screening-track)
- [Deterministic 3D representation track](#deterministic-3d-representation-track)
  - [Mathematical and conceptual foundations](#mathematical-and-conceptual-foundations)
  - [Visual atlas](#visual-atlas-cat3d-and-rgb-d-grids)
  - [Extended comparison](#extended-comparison-quality-coverage-and-stability)
  - [Small-sample follow-up experiments](#small-sample-follow-up-experiments)
  - [Reproducibility and outputs](#reproducibility-and-outputs)
  - [Final recommendation](#final-recommendation)

## At a glance

- **Datasets:** 10 NYUv2 images and 10 SUN RGB-D images.
- **Representations:** point cloud, surfel, mesh, voxel, surface TSDF,
  superpoint, graph, octree, and local descriptor.
- **Future representation family:** neural occupancy, signed-distance, and
  radiance fields are documented mathematically but are not yet implemented or
  included in the deterministic ranking.
- **Evaluation:** full-label mIoU, valid-depth mIoU, coverage, boundary F1,
  purity, construction time, element count, and viewpoint stability.
- **Scope:** deterministic visible RGB-D geometry plus a clearly separated
  RGB-only segmentation pilot. The deterministic representation study remains
  the main analysis; the neural pilot is used only to screen architecture
  combinations.

> **Observation:** The study identifies which representation is useful for a
> particular quality, coverage, boundary, or efficiency goal; it does not claim
> that one representation is universally best.

## Neural segmentation screening track

### V1 RGB SegFormer baseline

V1 is the first of the 26 planned architecture configurations and the control
model for the comparison:

```text
RGB image → SegFormer-B0 encoder/decoder → semantic logits
```

It intentionally uses **RGB only**. The aligned depth map is available through
the common dataset adapter for sample bookkeeping, but no depth, point cloud,
voxel, mesh, or other 3D representation is passed to the network. This makes
later depth/3D variants attributable to their added representation rather than
to a changing baseline.

The implementation is [`train_v1_rgb_segformer.py`](train_v1_rgb_segformer.py)
with one configuration per dataset:

```bash
python train_v1_rgb_segformer.py --config configs/v1_rgb_segformer_nyuv2_5ep.yaml
python train_v1_rgb_segformer.py --config configs/v1_rgb_segformer_sunrgbd_5ep.yaml
```

For the complete two-dataset run, use the wrapper that trains both datasets,
verifies all expected artifacts, and writes a combined summary:

```bash
python run_v1_all.py
```

The current reported metrics are documented in
[`docs/v1_results.md`](docs/v1_results.md).

Each screening configuration runs five full training epochs at 480×480 with batch size 4,
then evaluates mIoU, pixel accuracy, and per-class IoU. The one-epoch
configuration is retained only as a historical quick baseline;
five epochs provide a more stable screening comparison. Use
`--no-pretrained` only for a local smoke test; the actual comparison should use
the configured pretrained `nvidia/segformer-b0-finetuned-ade-512-512`
checkpoint (the ADE classifier layer is replaced for the target class count).

Every completed run also writes a qualitative validation grid containing RGB,
ground truth, prediction, and correct/wrong/ignored error panels. For an
already-trained checkpoint, regenerate the grid without retraining:

```bash
python visualize_v1_segformer.py \
  --config configs/v1_rgb_segformer_nyuv2_5ep.yaml
python visualize_v1_segformer.py \
  --config configs/v1_rgb_segformer_sunrgbd_5ep.yaml
```

The PNG is stored beside the checkpoint under
`outputs/segmentation/<experiment>/`.

For the architecture study, summarize all completed versions with:

```bash
python plot_architecture_comparison.py
```

This writes a metric heatmap, an accuracy-versus-parameter Pareto plot, and
dataset-specific qualitative comparison grids under
`outputs/segmentation/architecture_plots/`. Only completed versions are shown;
missing V2-V26 results are never fabricated.

> **Observation:** V1 answers the control question “how far does RGB alone go?”;
> a 3D representation is useful only if a matched later version improves over
> this baseline under the same split and evaluation protocol.

## Deterministic 3D representation track

### Mathematical and conceptual foundations

This section explains every 3D representation used by the project. It
separates three things that are easy to confuse:

1. the canonical mathematical representation;
2. the concrete approximation implemented in this repository;
3. the consequence of that approximation for the reported experiment.

The equations are tied directly to
[`repstudy/geometry.py`](repstudy/geometry.py) and
[`repstudy/representations.py`](repstudy/representations.py); when the
canonical definition is richer than the implementation, that limitation is
stated explicitly.

The current representation flow is:

```text
RGB + depth + camera intrinsics
                |
                v
       visible 3D point set
       /    /    |    \       \       \       \        \
  surfel  mesh  voxel  TSDF  superpoint  graph  octree  descriptor

Neural fields are a future learned function-space branch, not a completed row.
```

Common notation:

| Symbol | Meaning |
| --- | --- |
| \((u,v)\) | Image pixel coordinates |
| \(z\) | Metric camera depth |
| \(\mathbf K\) | Camera intrinsic matrix |
| \(\mathbf p_i,\mathbf X\) | 3D point in camera or world coordinates |
| \(\mathbf n_i\) | Unit surface normal |
| \(s\) | Voxel side length |
| \(\mathbf k\) | Integer voxel index |
| \(\mathcal N_k(i)\) | k-nearest neighbors of element \(i\) |
| \(\theta\) | Learned neural-field parameters |

#### Common camera model: RGB-D back-projection

Let a pixel be \(\tilde{\mathbf p}=(u,v,1)^\top\), its measured metric depth be
\(z\), and the camera intrinsic matrix be

$$
\mathbf K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
$$

The visible camera-space point is

$$
\mathbf X_c=z\mathbf K^{-1}\tilde{\mathbf p}
=
\begin{bmatrix}
(u-c_x)z/f_x\\
(v-c_y)z/f_y\\
z
\end{bmatrix}.
$$

Projection performs the inverse mapping:

$$
u=f_xX/Z+c_x,\qquad v=f_yY/Z+c_y.
$$

If multiple calibrated cameras are used, a rigid transform maps camera points
to a shared world frame, \(\mathbf X_w=\mathbf R\mathbf X_c+\mathbf t\). The
current NYUv2 and SUN RGB-D pilot constructs each sample in its own camera
frame; it does not fuse a registered temporal sequence.

Surface normals are estimated from neighboring back-projected pixels. For
finite differences \(\Delta_u\mathbf X\) and \(\Delta_v\mathbf X\),

$$
\mathbf n=
\frac{\Delta_u\mathbf X\times\Delta_v\mathbf X}
{\|\Delta_u\mathbf X\times\Delta_v\mathbf X\|_2+\varepsilon}.
$$

Invalid depth neighborhoods receive a zero normal in the implementation.

> **Concept:** A depth image is not itself a full 3D model. Back-projection
> produces only the first visible surface intersected by each camera ray;
> geometry behind that surface remains unobserved.

#### 1. Point cloud

A point cloud is an unordered set

$$
\mathcal P=\{(\mathbf p_i,\mathbf a_i)\}_{i=1}^{N},
\qquad \mathbf p_i\in\mathbb R^3,
$$

where \(\mathbf a_i\) may contain RGB, a normal, a semantic feature, or another
attribute. Reordering the points must not change the represented geometry:

$$
\{\mathbf p_1,\ldots,\mathbf p_N\}
=\{\mathbf p_{\pi(1)},\ldots,\mathbf p_{\pi(N)}\}.
$$

This permutation invariance is why point networks use symmetric aggregation,
for example \(\mathbf h=\max_i\phi(\mathbf p_i)\), as formalized by
[PointNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html).
Local geometry is commonly defined by Euclidean distance and a neighborhood:

$$
d_{ij}=\|\mathbf p_i-\mathbf p_j\|_2,
\qquad
\mathcal N_k(i)=\operatorname{kNN}_k(\mathbf p_i,\mathcal P).
$$

**This project:** valid RGB-D pixels are back-projected and then uniformly
subsampled with a fixed random seed. The baseline stores 5,000 points with RGB
and finite-difference normals. It does not reconstruct occluded surfaces or
connect the points.

**Trade-off:** point clouds preserve measured coordinates without voxel
quantization, but a fixed point budget gives low image coverage in this pilot
and does not directly encode surface connectivity.

#### 2. Surfel

A surfel is an oriented surface element rather than a dimensionless point:

$$
\mathcal S_i=(\mathbf p_i,\mathbf n_i,r_i,\mathbf c_i),
$$

with center \(\mathbf p_i\), unit normal \(\mathbf n_i\), support radius \(r_i\),
and optional color \(\mathbf c_i\). Its tangent-plane support can be written as

$$
|\mathbf n_i^\top(\mathbf x-\mathbf p_i)|\approx0,
\qquad
\|(\mathbf I-\mathbf n_i\mathbf n_i^\top)
(\mathbf x-\mathbf p_i)\|_2\le r_i.
$$

During splatting, a smooth footprint can weight a projected location by

$$
w_i(\mathbf x)=
\exp\left(-\frac{\|\mathbf x_\perp-\mathbf p_i\|_2^2}{2r_i^2}\right).
$$

Here \(\mathbf x_\perp\) is the orthogonal projection of \(\mathbf x\) onto
the surfel's tangent plane.

Surfels were introduced as surface-element rendering primitives by
[Pfister et al.](https://doi.org/10.1145/344779.344936).

**This project:** the surfel centers and normals are the sampled point-cloud
values. The baseline radius is \(r_i=0.02\) m. The optional adaptive rule uses
\(r_i=\max(r_{\min},z_i/f_x)\), approximately one projected-pixel footprint.

**Trade-off:** orientation and area make surfels better surface primitives than
raw points, but holes remain when sampling is sparse or radii are too small;
large radii can blur boundaries and join unrelated surfaces.

#### 3. Triangle mesh and mesh generation

A triangle mesh is

$$
\mathcal M=(\mathcal V,\mathcal F),
\qquad
\mathcal V=\{\mathbf v_i\in\mathbb R^3\},
\quad
\mathcal F\subset\mathbb N^3.
$$

For a face \((i,j,k)\), every point inside the triangle has barycentric form

$$
\mathbf x=\alpha\mathbf v_i+\beta\mathbf v_j+\gamma\mathbf v_k,
\quad
\alpha,\beta,\gamma\ge0,
\quad
\alpha+\beta+\gamma=1.
$$

Its area and normal are

$$
A_{ijk}=\frac12
\|(\mathbf v_j-\mathbf v_i)\times(\mathbf v_k-\mathbf v_i)\|_2,
$$

$$
\mathbf n_{ijk}=
\frac{(\mathbf v_j-\mathbf v_i)\times(\mathbf v_k-\mathbf v_i)}
{\|(\mathbf v_j-\mathbf v_i)\times(\mathbf v_k-\mathbf v_i)\|_2}.
$$

There are two relevant mesh-generation routes:

- **Organized-depth triangulation, used here:** a valid 2x2 pixel cell is split
  into \((00,01,11)\) and \((00,11,10)\). A cell is rejected when its depth
  jump exceeds \(\tau_z\), an edge exceeds \(\tau_e\), or an optional normal
  disagreement exceeds \(\tau_n\).
- **Isosurface extraction, not used by the current mesh row:** for a scalar
  field \(f(\mathbf x)\), Marching Cubes extracts \(f(\mathbf x)=\tau\). If an
  edge joins values \(f_a\) and \(f_b\), the intersection is linearly placed at

$$
\mathbf x_e=\mathbf x_a+
\frac{\tau-f_a}{f_b-f_a}(\mathbf x_b-\mathbf x_a).
$$

The classical source is
[Marching Cubes](https://doi.org/10.1145/37401.37422).

**This project:** \(\tau_z=0.05\) m and \(\tau_e=0.25\) m in the baseline. The
result is a single-view partial mesh with duplicated per-triangle vertices, not
a watertight object or room reconstruction.

**Trade-off:** explicit faces preserve visible boundaries and make rendering
efficient, explaining the strong boundary F1 in this pilot. Depth noise can
create bad triangles, while conservative rejection creates holes.

#### 4. Sparse voxel grid

A voxel grid partitions space into cubes of side length \(s\). A point receives
integer voxel key

$$
\mathbf k_i=\left\lfloor\frac{\mathbf p_i}{s}\right\rfloor
\in\mathbb Z^3.
$$

The geometric center of voxel \(\mathbf k\) is

$$
\mathbf c_{\mathbf k}=s(\mathbf k+\tfrac12\mathbf 1),
$$

and binary occupancy is

$$
O(\mathbf k)=
\mathbb 1\left[\exists i:\mathbf k_i=\mathbf k\right].
$$

If the voxel center is used as the representative coordinate, the quantization
error is at most \(s/2\) per axis and \(\sqrt3s/2\) in Euclidean distance. A
dense grid over a cube of side \(L\) requires \(O((L/s)^3)\) cells, whereas a sparse map stores only
\(N_{\mathrm{occupied}}\) cells. Voxel learning commonly applies regular 3D
convolution to occupancy grids, as in
[VoxNet](https://publications.ri.cmu.edu/voxnet-a-3d-convolutional-neural-network-for-real-time-object-recognition).

**This project:** \(s=0.04\) m. Points with the same key are grouped, and the
stored representative position and RGB are their means rather than the voxel
center. This avoids center-snapping error in the representative coordinate,
but cell membership still has 4 cm resolution. Therefore this is a sparse
visible-surface voxelization, not a dense solid occupancy volume.

**Trade-off:** voxels provide regular indexing, neighborhood lookup and full
valid-depth support, which made them the strongest dense default in this
pilot. Smaller \(s\) preserves detail but increases memory and runtime roughly
cubically for a dense implementation.

#### 5. Truncated Signed Distance Field (TSDF)

For a closed surface \(\partial\Omega\), a signed distance field is

$$
d(\mathbf x)=
\begin{cases}
-\min_{\mathbf y\in\partial\Omega}\|\mathbf x-\mathbf y\|_2,
&\mathbf x\text{ inside},\\
\phantom{-}\min_{\mathbf y\in\partial\Omega}\|\mathbf x-\mathbf y\|_2,
&\mathbf x\text{ outside}.
\end{cases}
$$

The surface is its zero level set, \(\partial\Omega=\{\mathbf x:d(\mathbf
x)=0\}\). For a depth image, a common projective signed distance for voxel
\(\mathbf x\) is

$$
\delta_t(\mathbf x)=
z_t\!\left(\pi(\mathbf K\mathbf x)\right)-x_z,
$$

where positive values lie in observed free space in front of the measured
surface under this sign convention. With truncation distance \(\mu\),

$$
\phi_t(\mathbf x)=
\operatorname{clip}\left(\frac{\delta_t(\mathbf x)}{\mu},-1,1\right).
$$

Aligned frames are fused by a weighted running average:

$$
D_{t+1}(\mathbf x)=
\frac{W_t(\mathbf x)D_t(\mathbf x)+w_t(\mathbf x)\phi_t(\mathbf x)}
{W_t(\mathbf x)+w_t(\mathbf x)},
\qquad
W_{t+1}=W_t+w_t.
$$

This follows the cumulative weighted signed-distance formulation of
[Curless and Levoy](https://graphics.stanford.edu/papers/volrange/). A mesh can
then be extracted from \(D(\mathbf x)=0\) with Marching Cubes.

**This project:** the representation is intentionally a **single-view sparse
surface-TSDF proxy**. NYUv2 uses \(s=0.04\) m and \(\mu=4s=0.16\) m; SUN RGB-D
uses \(s=0.06\) m and \(\mu=3s=0.18\) m. It samples free cells along rays,
stores `+1` for free space and `0` for observed surface cells, and omits
unobserved space. It does **not** store continuous distances, negative
behind-surface values, or weighted multiview fusion. Its current mathematics
is therefore

$$
\phi_{\mathrm{pilot}}(\mathbf k)\in\{0,+1\},
$$

not a complete canonical TSDF in \([-1,1]\).

**Trade-off:** even this proxy distinguishes observed free space from surface
support, but it cannot perform true zero-crossing reconstruction or infer the
unseen side of a surface. Its results must not be generalized to a fused TSDF
reconstruction system.

#### 6. Superpoint regions

A superpoint representation partitions the point set into disjoint regions:

$$
\mathcal P=\bigcup_{m=1}^{M}\mathcal S_m,
\qquad
\mathcal S_m\cap\mathcal S_n=\varnothing\quad(m\ne n).
$$

Each region can be represented by a centroid and pooled feature,

$$
\bar{\mathbf p}_m=\frac1{|\mathcal S_m|}
\sum_{i\in\mathcal S_m}\mathbf p_i,
\qquad
\bar{\mathbf a}_m=\operatorname{AGG}_{i\in\mathcal S_m}\mathbf a_i.
$$

Canonical superpoints group geometrically homogeneous points and may become
nodes in a superpoint graph, as in
[Superpoint Graphs](https://openaccess.thecvf.com/content_cvpr_2018/html/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.html).

**This project:** the pilot uses a deterministic image-tile approximation. For
tile width \(t\),

$$
\mathcal S_{ab}=\{(v,u):\lfloor v/t\rfloor=a,
\lfloor u/t\rfloor=b,\ z(v,u)>0\}.
$$

The target is approximately 1,000 regions. Region centroids are computed in
3D, while edges join right/down neighboring image tiles. This is not learned
geometric partitioning and should be called a superpoint proxy.

**Trade-off:** strong compression and complete source-pixel bookkeeping make
the proxy useful for the pilot, but rectangular image tiles can merge points
across a true 3D or semantic boundary.

#### 7. k-nearest-neighbor graph

A geometric graph is \(\mathcal G=(\mathcal V,\mathcal E)\), with point nodes
\(\mathcal V=\{1,\ldots,N\}\) and undirected edges

$$
\mathcal E=\{(i,j):j\in\mathcal N_k(i)\},
\qquad
\mathcal N_k(i)=\operatorname{kNN}_k(\mathbf p_i,\mathcal P).
$$

A learned graph network would update node features through message passing:

$$
\mathbf h_i^{(\ell+1)}=
\Phi_\ell\left(
\mathbf h_i^{(\ell)},
\operatorname{AGG}_{j\in\mathcal N(i)}
\Psi_\ell(\mathbf h_i^{(\ell)},\mathbf h_j^{(\ell)},
\mathbf p_j-\mathbf p_i)
\right).
$$

The use of 3D neighborhood graphs for RGB-D segmentation is demonstrated by
[3D Graph Neural Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Qi_3D_Graph_Neural_ICCV_2017_paper.html).

**This project:** 2,500 seeded point samples are connected with \(k=8\). The
representation stores RGB, normals, node degree and undirected edges. No graph
neural network or message-passing parameters are trained in the deterministic
study.

**Trade-off:** graphs explicitly represent non-grid relationships and support
learned propagation, but kNN construction and storage add cost. Euclidean kNN
can also connect two different surfaces that happen to be spatially close.

#### 8. Octree

An octree recursively divides a cube into eight children. If the root side
length is \(L\), a node at level \(\ell\) has side length

$$
s_\ell=\frac{L}{2^\ell}.
$$

For node midpoint \(\mathbf m\), the three comparison bits

$$
b_x=\mathbb 1[x\ge m_x],\quad
b_y=\mathbb 1[y\ge m_y],\quad
b_z=\mathbb 1[z\ge m_z]
$$

select child index \(c=b_x+2b_y+4b_z\). Recursion stops when a node is simple
enough or reaches maximum depth. Unlike a dense depth-\(D\) grid with
\(8^D\) possible leaves, an adaptive tree stores only visited nodes and
occupied leaves. The classical hierarchical formulation is described by
[Meagher](https://doi.org/10.1016/0146-664X(82)90104-6).

**This project:** the root tightly bounds the visible cloud. A node becomes a
leaf when it contains at most 128 points or reaches depth 7. Each leaf stores
its bounds, level, point count, source-pixel members and center. Empty-space
children are not stored.

**Trade-off:** octrees allocate fine resolution only where needed, giving the
best construction-time/full-coverage combination in this pilot. Tree traversal
and irregular batching are more complicated than dense voxel convolution, and
large leaves can remove thin geometric detail.

#### 9. Local geometric descriptor

A descriptor is a feature attached to geometry; it is not a standalone surface
unless the supporting point locations are retained. For sampled point
\(\mathbf p_i\) and its kNN offsets
\(\mathbf q_{ij}=\mathbf p_j-\mathbf p_i\), the pilot computes

$$
\mathbf C_i=\frac1{k}\sum_{j\in\mathcal N_k(i)}
\mathbf q_{ij}\mathbf q_{ij}^{\top}.
$$

Let the normalized eigenvalues satisfy
\(\lambda_1\ge\lambda_2\ge\lambda_3\ge0\) and
\(\lambda_1+\lambda_2+\lambda_3=1\). The shape measures are

$$
L_i=\frac{\lambda_1-\lambda_2}{\lambda_1+\varepsilon},
\qquad
P_i=\frac{\lambda_2-\lambda_3}{\lambda_1+\varepsilon},
\qquad
S_i=\lambda_3.
$$

Local density is represented by inverse mean neighbor distance:

$$
\rho_i=
\left(
\frac1{k}\sum_{j\in\mathcal N_k(i)}
\|\mathbf p_j-\mathbf p_i\|_2
+\varepsilon
\right)^{-1}.
$$

Local-neighborhood encoding is also used by classical descriptors such as
[FPFH](https://doi.org/10.1109/ROBOT.2009.5152473), although FPFH uses angular
histograms while this pilot specifically uses PCA-eigenvalue shape measures.

**This project:** 5,000 points and \(k=16\) are used. The exact ten-dimensional
descriptor is

$$
[\lambda_1,\lambda_2,\lambda_3,L_i,P_i,S_i,\rho_i,R_i,G_i,B_i].
$$

It is visualized using \(\lambda_3\) as a scalar color and retains point
locations plus finite-difference normals.

**Trade-off:** descriptors expose local shape to classical or learned models
and can be rotation/scale normalized, but they depend strongly on neighborhood
size, sampling density and noise. The fixed 5,000-point support causes the same
coverage limitation as the sampled point cloud.

#### 10. Neural fields: documented future representation

A neural field stores a continuous function in learned parameters rather than
an explicit list of points, cells or faces:

$$
f_\theta:\mathbb R^3\rightarrow\mathcal Y.
$$

Important forms are:

- **Occupancy field:** \(f_\theta(\mathbf x)\in[0,1]\), with surface
  \(f_\theta(\mathbf x)=0.5\). See
  [Occupancy Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Mescheder_Occupancy_Networks_Learning_3D_Reconstruction_in_Function_Space_CVPR_2019_paper.html).
- **Neural SDF:** \(f_\theta(\mathbf x)\in\mathbb R\), with surface
  \(f_\theta(\mathbf x)=0\). See
  [DeepSDF](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html).
- **Radiance field:**
  \(f_\theta(\mathbf x,\mathbf d)=(\sigma,\mathbf c)\), where density and
  view-dependent color are rendered along a ray
  \(\mathbf r(t)=\mathbf o+t\mathbf d\). See
  [NeRF](https://arxiv.org/abs/2003.08934).

For a radiance field, continuous volume rendering is

$$
\hat{\mathbf C}(\mathbf r)=
\int_{t_n}^{t_f}T(t)\sigma(\mathbf r(t))
\mathbf c(\mathbf r(t),\mathbf d)\,dt,
$$

$$
T(t)=\exp\left(-\int_{t_n}^{t}\sigma(\mathbf r(s))\,ds\right).
$$

With discrete samples, \(\alpha_i=1-e^{-\sigma_i\Delta_i}\),
\(T_i=\prod_{j<i}(1-\alpha_j)\), and

$$
\hat{\mathbf C}(\mathbf r)=\sum_iT_i\alpha_i\mathbf c_i.
$$

An RGB photometric objective is
\(\sum_{\mathbf r}\|\hat{\mathbf C}(\mathbf r)-\mathbf C(\mathbf r)\|_2^2\).
A neural SDF can instead use metric depth/SDF supervision and the Eikonal
regularizer

$$
\mathcal L_{\mathrm{eik}}=
\mathbb E_{\mathbf x}
\left(\|\nabla_{\mathbf x}f_\theta(\mathbf x)\|_2-1\right)^2.
$$

**This project:** no neural field is currently trained, rendered, timed, or
ranked. A fair addition would require a new learned protocol, train/test scene
separation, camera poses or RGB-D field supervision, query resolution, training
time, rendering time, model size and reconstruction metrics. It must not be
inserted into the deterministic 10-image table without that protocol.

**Trade-off:** neural fields offer continuous queries, differentiable rendering
and potentially high fidelity, but training can be expensive, results depend
on scene coverage, and geometry may be difficult to edit or inspect compared
with explicit points, voxels or meshes.

#### Exact baseline settings and interpretation

| Representation | Mathematical domain | Baseline in this repository | What is actually stored |
| --- | --- | --- | --- |
| Point cloud | Unordered samples in \(\mathbb R^3\) | 5,000 seeded points | XYZ, RGB, normal, source pixel |
| Surfel | Oriented disks | 5,000 points, radius 0.02 m | Point attributes plus radius |
| Partial mesh | Triangle surface | depth jump 0.05 m, edge limit 0.25 m | Triangle vertices/faces and source pixels |
| Sparse voxel | Discrete \(\mathbb Z^3\) cells | voxel side 0.04 m | Occupied key, mean XYZ/RGB, source pixels |
| Surface-TSDF proxy | Sparse scalar volume | NYUv2: 0.04/0.16 m voxel/truncation; SUN: 0.06/0.18 m | `0` surface, `+1` free, unobserved omitted |
| Superpoint proxy | Partition of visible points | approximately 1,000 image tiles | Region centroid, members, tile adjacency |
| kNN graph | Nodes and edges | 2,500 nodes, \(k=8\) | XYZ/RGB/normal/degree and edges |
| Octree | Adaptive hierarchical cells | depth at most 7, 128 points/leaf | Leaf bounds, level, count, members |
| Descriptor | Local feature field on points | 5,000 points, \(k=16\) | 10D local feature, XYZ, normal |
| Neural field | Learned continuous function | Not implemented | No result or artifact yet |

RGB-D is the input modality, not a derived 3D representation. The original
Cat3D OBJ/CAD asset is already an explicit triangle mesh. Multiview and
projection panels are rendering/evaluation operations applied to a stored
representation; they do not create additional geometry by themselves. The
`original_2d` row is an identity projection control, not a competing 3D data
structure.

> **Observation:** These representations do not contain equivalent information.
> A point cloud and mesh primarily describe observed surfaces; a voxel grid
> discretizes space; a canonical TSDF stores signed proximity to a surface; a
> graph stores relationships; a descriptor stores local measurements; and a
> neural field stores a learned query function. Their metrics are meaningful
> only when resolution, coverage and construction/training budgets are reported.

### Visual atlas: Cat3D and RGB-D grids

#### Cat3D 3D representation atlas

<p align="center">
  <img src="docs/figures/cat3d_original_plus_9_grid.png" alt="Cat3D original model plus nine 3D representations" width="100%">
</p>

The atlas starts with the original Cat OBJ/CAD model and then shows its nine
derived 3D representations.

> **Observation:** The same object can be stored as a surface, volume,
> hierarchy, graph, or descriptor field; these are different data structures,
> not merely different visual styles.

#### Dataset visual overview

The main generated figures appear directly after the mathematical foundations.
The 2D grids show camera-space projections; the 3D grids show the raw geometric
structures.

| Dataset | 2D representation grid | 3D representation grid |
| --- | --- | --- |
| NYUv2 | <img src="docs/figures/nyuv2_2d_representation_grid.png" alt="NYUv2 2D representation grid" width="420"> | <img src="docs/figures/nyuv2_3d_representation_grid.png" alt="NYUv2 3D representation grid" width="420"> |
| SUN RGB-D | <img src="docs/figures/sunrgbd_2d_representation_grid.png" alt="SUN RGB-D 2D representation grid" width="420"> | <img src="docs/figures/sunrgbd_3d_representation_grid.png" alt="SUN RGB-D 3D representation grid" width="420"> |
| Cat3D OBJ | <img src="docs/figures/cat3d_2d_representation_grid.png" alt="Cat3D 2D representation grid" width="420"> | <img src="docs/figures/cat3d_original_plus_9_grid.png" alt="Cat3D original model plus nine representations" width="420"> |

The Cat3D figure contains the original OBJ/CAD model plus nine derived 3D
representations. The separate `catDog.png` RGB scene is intentionally not
included in the output atlas.

> **Observation:** A representation can look correct in 3D but still miss many
> labeled image pixels after projection, so the 2D and 3D grids must be read
> together.

### Baseline benchmark and Cat3D analysis

The pilot comparison below reports valid-depth mIoU, coverage, full-label mIoU,
and construction time for all nine representations on both RGB-D datasets.

<p align="center">
  <img src="docs/figures/representation_benchmark.png" alt="NYUv2 and SUN RGB-D representation benchmark" width="100%">
</p>

> **Observation:** Dense representations are the meaningful full-image
> comparison; sparse fixed-budget styles can have high local accuracy while
> covering only a small part of the image.

For Cat3D, the additional figures show orthographic and perspective
projections, depth and descriptor maps, surface normals, representation sizes,
and the local descriptor distribution.

| Cat3D projection analysis | Cat3D statistics |
| --- | --- |
| <img src="docs/figures/cat3d_projection_analysis.png" alt="Cat3D projection and geometric analysis" width="520"> | <img src="docs/figures/cat3d_statistics.png" alt="Cat3D representation statistics" width="520"> |

> **Observation:** The Cat3D atlas explains structure visually, while these
> projections and statistics expose scale and local geometric differences.

The study compares organized point clouds, surfels, partial meshes, sparse
voxels, single-view sparse TSDFs, superpoint regions, kNN graphs, adaptive
octrees, local geometric descriptors, and deterministic multiview projections. Ground truth is used only for oracle labeling and
evaluation -- not to construct the primary geometry.

### Extended comparison: quality, coverage, and stability

The extended analysis uses all nine representation styles, all ten pilot images
per dataset, and the four saved views (original, left oblique, right oblique,
and elevated). It separates dense semantic fidelity from coverage, boundary
quality, runtime, element count, image-to-image variation, and viewpoint
robustness.

<p align="center">
  <img src="docs/figures/representation_tradeoff_overview.png" alt="Representation quality, coverage, runtime, and size trade-offs" width="100%">
</p>

> **Observation:** Higher quality generally requires more geometry or more
> construction time; the best choice depends on the deployment objective.

<p align="center">
  <img src="docs/figures/representation_rank_heatmap.png" alt="Representation ranks across quality, coverage, purity, boundary quality, runtime, and size" width="100%">
</p>

> **Observation:** No single style dominates every criterion; the preferred
> representation changes when the objective changes from quality to boundaries,
> compactness, or speed.

| Decision objective | NYUv2 pilot | SUN RGB-D pilot |
| --- | --- | --- |
| Highest full-label mIoU with full coverage | Voxel - 96.4% | Voxel - 60.3% |
| Best boundary F1 | Mesh - 71.4% | Mesh - 43.4% |
| Smallest full-coverage style | Superpoint - 972 elements | Superpoint - 904 elements |
| Fastest full-coverage style | Octree - 0.46 s/image | Octree - 0.38 s/image |

The practical first choice for dense RGB-D geometry in this pilot is therefore
the voxel representation. Choose octrees when memory and construction time are
more important, and meshes when boundary preservation is the priority. The
point cloud, surfel, graph, and descriptor rows use fixed point budgets and
cover only about 1-2% of the labeled image; their near-perfect valid-depth mIoU
is therefore not a fair dense-image comparison. Full-label mIoU and coverage
must be read together.

> **Observation:** For this pilot, voxel is the safest dense default, mesh is
> strongest for boundaries, octree is the speed/size option, and superpoint is
> the most compact full-coverage style.

<p align="center">
  <img src="docs/figures/representation_sample_variability.png" alt="Per-image variability for representation quality and coverage" width="100%">
</p>

> **Observation:** Averages hide difficult frames; this plot shows which styles
> are consistently reliable and which are sensitive to scene content.

<p align="center">
  <img src="docs/figures/representation_multiview_robustness.png" alt="Multiview robustness for every representation style" width="100%">
</p>

> **Observation:** A representation that changes sharply across viewpoints is
> less suitable when downstream processing must be viewpoint-stable.

<p align="center">
  <img src="docs/figures/representation_case_study.png" alt="Coverage and oracle error maps for a median-quality case study in each dataset" width="100%">
</p>

> **Observation:** The case studies connect the numbers to pixels: missing
> coverage and boundary errors are spatially distinct failure modes.

The tables behind these plots are written to
`outputs/analysis/representation_extended_metrics.csv`,
`outputs/analysis/representation_multiview_metrics.csv`, and
`outputs/analysis/representation_ranks.csv`. Recreate the complete analysis
with:

```bash
python -m repstudy.plot_extended_analysis
```

The complete written interpretation is in
[`docs/representation_findings.md`](docs/representation_findings.md). These are deterministic oracle
measurements of visible RGB-D geometry on ten images per dataset, not a claim
that one representation is universally optimal or a replacement for a trained
segmentation benchmark.

> **Observation:** This is a deterministic oracle analysis of visible geometry,
> not a trained segmentation benchmark; it identifies candidates for later
> learned-system experiments.

### Small-sample follow-up experiments

The next experiments also use exactly ten NYUv2 and ten SUN RGB-D images. The
same image IDs are reused across all settings, so the comparisons are paired
and reproducible without running the full datasets.

<p align="center">
  <img src="docs/figures/rate_distortion.png" alt="Rate-distortion curves for all nine RGB-D representation styles" width="100%">
</p>

The rate-distortion pilot evaluates the baseline plus controlled coarse/fine
settings for all nine styles. For example, voxel mIoU changes from 92.1% to
96.4% to 98.6% on NYUv2 as the mean element count changes from 6.8k to 21.9k
to 62.6k. On SUN RGB-D the corresponding values are 58.9%, 60.3%, and 61.2%.
This shows that representation resolution is a real factor in the ranking.

> **Observation:** Representation resolution is an experimental factor; rankings
> should not be interpreted without reporting the geometry budget.

<p align="center">
  <img src="docs/figures/depth_corruption_robustness.png" alt="Depth dropout and Gaussian noise robustness for voxel, octree, and superpoint" width="100%">
</p>

The corruption pilot applies 10% and 30% random depth dropout plus 1 cm and
3 cm Gaussian depth noise. At 30% dropout, NYUv2 full-label mIoU falls to
67.8% for voxel, 65.6% for octree, and 60.6% for superpoints. The corresponding
SUN RGB-D values are 42.3%, 41.1%, and 38.5%. Gaussian noise is less damaging
than dropout in this pilot because the representations still retain complete
valid-depth support.

> **Observation:** Missing depth is more damaging than small metric noise in
> this pilot. The 100% dense-style coverage is relative to remaining valid
> support, not evidence that dropped measurements were recovered.

<p align="center">
  <img src="docs/figures/adaptive_hybrid_results.png" alt="Held-out adaptive hybrid comparison against voxel, mesh, and octree" width="100%">
</p>

The adaptive hybrid uses geometry-only depth/RGB edge scores: mesh near
structural edges, voxel in interiors, and octree as a fallback. The edge
quantile is selected on five images and evaluated on five held-out images. Its
held-out full-label mIoU is 98.2% on NYUv2 versus 96.8% for voxel, and 65.6% on
SUN RGB-D versus 64.9% for voxel. This is promising pilot evidence, not yet a
final claim because the held-out split contains only five images per dataset.

> **Observation:** Geometry-only routing is promising, but the five-image
> held-out split is too small for a universal claim.

<p align="center">
  <img src="docs/figures/representation_efficiency.png" alt="Runtime and XYZ geometry-storage efficiency for the nine baseline styles" width="100%">
</p>

The efficiency plot reports construction time and an explicitly labeled XYZ
geometry-storage lower bound. It is a storage proxy, not a measurement of the
complete serialized representation including adjacency and attributes.

> **Observation:** This proxy is useful for comparing scale and speed, but a
> deployment decision should measure the complete serialized representation.

The experiment runner and tables are:

```bash
python -m repstudy.run_small_sample_experiments
```

```text
outputs/experiments/rate_distortion_metrics.csv
outputs/experiments/depth_corruption_metrics.csv
outputs/experiments/adaptive_hybrid_metrics.csv
outputs/experiments/representation_efficiency.csv
outputs/experiments/small_sample_experiments.md
```

The full protocol and limitations are summarized in
[`docs/small_sample_experiments.md`](docs/small_sample_experiments.md). These
experiments are a 10+10-image pilot; full-dataset evaluation is still required
before making universal claims.

> **Observation:** These are pilot experiments for comparison and debugging;
> full-dataset evaluation is required before broad conclusions.

### Reproducibility and outputs

The first deterministic pipeline is implemented and smoke-tested on NYUv2:
backprojection, organized point clouds, surfels, partial meshes, sparse
voxels, single-view sparse TSDFs, superpoint regions, oracle labels,
z-buffer rendering, coverage-aware semantic metrics, boundary metrics, kNN
graphs, adaptive octrees, local descriptors, and virtual-camera projections.
The fixed pilot study uses 10 images from each
dataset, with no full-dataset training or sweep.

### Reproduce the main study

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

> **Observation:** The fixed smoke configurations make the study reproducible
> and affordable; they are not a substitute for a full-dataset benchmark.

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

> **Observation:** The 3D, 2D, and multiview notebooks answer different
> questions: geometric structure, image coverage, and viewpoint stability.

### Cat3D asset visualization

The local `data/cat3D/` folder contains a real Cat OBJ mesh with MTL and texture
files. The output atlas contains the original OBJ/CAD model plus the same nine
representations used by the main study: point cloud, surfel, mesh, voxel,
surface TSDF, superpoint regions, graph, octree, and descriptor.

```text
outputs/cat3d/cat3d_original_plus_9_grid.png
outputs/cat3d/cat3d_2d_representation_grid.png
outputs/cat3d/cat3d_projection_analysis.png
outputs/cat3d/cat3d_statistics.png
```

The separate `data/catDog.png` RGB scene is not included in this 3D output,
because it is not a calibrated geometry source.
Use `notebooks/06_catdog_asset_grids.ipynb` to view the Cat3D atlas.

> **Observation:** Cat3D is a controlled object-level sanity check; NYUv2 and
> SUN RGB-D remain the evidence for real RGB-D scene behavior.

### How to read the metrics

`valid_depth` mIoU evaluates only pixels with valid depth and representation
coverage. `full_label` mIoU counts uncovered labeled pixels as missing. Always
read mIoU together with `valid_depth_coverage` and `valid_depth_missing_rate`;
an identity-like point sample can have high covered-pixel accuracy while still
covering only a small fraction of the image.

> **Observation:** Full-label mIoU is the safest single dense-image summary,
> but coverage and boundary F1 explain why that score changes.

### Learning note and repository policy

Each representation keeps the source image pixels that contributed to each
3D element. That bookkeeping is what makes deterministic oracle labeling and
the missing-coverage measurement possible. The deterministic track uses no
neural network; the separate V1 screening track is the only learned baseline
currently included.

Generated data, datasets, credentials, and checkpoints will remain outside Git.

### Final recommendation

For dense RGB-D scene representation, **voxel is the recommended default** in
this pilot. It provides full image support and the highest full-label mIoU on
both datasets: 96.4% on NYUv2 and 60.3% on SUN RGB-D. Its regular spatial grid
also makes projection, batching, and downstream processing straightforward.

Choose another representation when the objective is different:

| Objective | Recommendation | Reason |
| --- | --- | --- |
| Best general dense representation | Voxel | Highest full-label mIoU with full coverage in this pilot |
| Best semantic boundaries | Mesh | Highest boundary F1 on both datasets |
| Lowest construction cost | Octree | Fastest full-coverage construction in the pilot |
| Smallest full-coverage representation | Superpoint | Fewest elements while retaining full support |
| Adaptive future direction | Hybrid mesh + voxel + octree | Promising held-out pilot result, but needs larger validation |

> **Final observation:** Use voxel as the primary RGB-D representation for the
> current study. Use mesh when boundary quality matters most, octree when speed
> or memory matters most, and superpoints when compactness matters most. The
> adaptive hybrid is a promising next experiment, not yet the final winner
> because it was evaluated on only five held-out images per dataset.
