"""Additional benchmark plots for the two RGB-D pilots and the Cat3D asset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .obj_utils import face_centers, face_normals, load_obj, normalize_vertices
from .plot_catdog_grids import (_build_asset, _project, _projected_mesh,
                                _equal_axes, _mesh_plot)


REPRESENTATIONS = ("pointcloud", "surfel", "mesh", "voxel", "tsdf", "superpoint",
                   "graph", "octree", "descriptor")


def _read_summary(output_root, dataset):
    path = Path(output_root) / dataset / f"{dataset}_representation_smoke" / "study_summary.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_representation_analysis(output_root="outputs", output=None):
    summaries = {dataset: _read_summary(output_root, dataset) for dataset in ("nyuv2", "sunrgbd")}
    metrics = {
        "valid-depth mIoU": lambda item: item["valid_depth_miou"]["mean"] * 100,
        "coverage": lambda item: item["valid_depth_coverage"]["mean"] * 100,
        "full-label mIoU": lambda item: item["full_label_miou"]["mean"] * 100,
        "construction time (s)": lambda item: item["construction_time_s"]["mean"],
    }
    colors = {"nyuv2": "#2878b5", "sunrgbd": "#d95f02"}
    figure, axes = plt.subplots(2, 2, figsize=(17, 10), squeeze=False)
    x = np.arange(len(REPRESENTATIONS))
    width = 0.36
    for axis, (title, getter) in zip(axes.flat, metrics.items()):
        for offset, dataset in zip((-width / 2, width / 2), ("nyuv2", "sunrgbd")):
            rows = summaries[dataset]["representations"]
            values = [getter(rows[name]) for name in REPRESENTATIONS]
            axis.bar(x + offset, values, width, label=dataset.upper() if dataset == "nyuv2" else "SUN RGB-D",
                     color=colors[dataset], alpha=0.88)
        axis.set_title(title)
        axis.set_xticks(x, [name.replace("superpoint", "super\npoint") for name in REPRESENTATIONS], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        if title != "construction time (s)":
            axis.set_ylim(0, 105)
            axis.set_ylabel("percent")
        else:
            axis.set_yscale("log")
            axis.set_ylabel("seconds, log scale")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Representation pilot comparison: NYUv2 versus SUN RGB-D", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    if output is None:
        output = Path(output_root) / "analysis" / "representation_benchmark.png"
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150); plt.close(figure)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "nyuv2_valid_depth_miou", "sunrgbd_valid_depth_miou",
                         "nyuv2_coverage", "sunrgbd_coverage", "nyuv2_full_label_miou",
                         "sunrgbd_full_label_miou", "nyuv2_time_s", "sunrgbd_time_s"])
        for name in REPRESENTATIONS:
            values = []
            for dataset in ("nyuv2", "sunrgbd"):
                row = summaries[dataset]["representations"][name]
                values.extend([row["valid_depth_miou"]["mean"], row["valid_depth_coverage"]["mean"],
                               row["full_label_miou"]["mean"], row["construction_time_s"]["mean"]])
            writer.writerow([name, values[0], values[4], values[1], values[5], values[2], values[6], values[3], values[7]])
    return output, csv_path


def make_cat3d_projection_analysis(obj_path, output):
    mesh = load_obj(obj_path)
    vertices, faces = normalize_vertices(mesh.vertices), mesh.faces
    asset = _build_asset(vertices, faces)
    figure, axes = plt.subplots(2, 4, figsize=(20, 10), squeeze=False)
    _projected_mesh(axes[0, 0], vertices, faces, "front", "Front projection")
    _projected_mesh(axes[0, 1], vertices, faces, "side", "Side projection")
    _projected_mesh(axes[0, 2], vertices, faces, "top", "Top projection")
    _projected_mesh(axes[0, 3], vertices, faces, "perspective", "Perspective projection")
    points = asset["points"]
    projected, depth = _project(points, "front")
    axes[1, 0].scatter(projected[:, 0], projected[:, 1], c=depth, s=2, cmap="magma", alpha=0.7)
    axes[1, 0].set_title("Depth-colored projection")
    axes[1, 1].scatter(projected[:, 0], projected[:, 1], c=asset["descriptors"][:, 5], s=2, cmap="turbo", alpha=0.75)
    axes[1, 1].set_title("Descriptor projection")
    axes[1, 2].scatter(projected[:, 0], projected[:, 1], s=2, c="black", alpha=0.65)
    axes[1, 2].set_title("Silhouette / occupancy")
    normal_ids = np.linspace(0, len(asset["normal_centers"]) - 1, min(1600, len(asset["normal_centers"])), dtype=int)
    centers = asset["normal_centers"][normal_ids]
    normals = asset["normal_vectors"][normal_ids]
    normal_xy, _ = _project(centers, "front")
    axes[1, 3].scatter(normal_xy[:, 0], normal_xy[:, 1], s=1.5, c="0.65", alpha=0.55)
    axes[1, 3].quiver(normal_xy[:, 0], normal_xy[:, 1], normals[:, 0], normals[:, 1],
                       color="teal", alpha=0.35, angles="xy", scale_units="xy", scale=5, width=0.0015)
    axes[1, 3].set_title("Projected surface normals")
    for axis in axes[1]:
        axis.set_aspect("equal")
        axis.axis("off")
    figure.suptitle("Cat3D projection and geometric analysis", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150); plt.close(figure)
    return output


def make_cat3d_statistics(obj_path, output):
    mesh = load_obj(obj_path)
    vertices, faces = normalize_vertices(mesh.vertices), mesh.faces
    asset = _build_asset(vertices, faces)
    names = ["original\nvertices", "pointcloud", "surfel", "mesh\ntriangles", "voxel", "TSDF",
             "superpoint", "graph\nnodes", "octree", "descriptor"]
    counts = [len(vertices), len(asset["points"]), len(asset["points"]), len(faces),
              len(asset["voxel_centers"]), len(asset["tsdf_centers"]), len(asset["superpoint_centers"]),
              len(asset["graph_points"]), len(asset["octree"]["centers"]), len(asset["points"])]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(np.arange(len(names)), counts, color="#6a51a3")
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axes[0].set_ylabel("elements (log scale)")
    axes[0].set_title("Cat3D representation sizes")
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].hist(asset["descriptors"][:, 5], bins=24, color="#1b9e77", alpha=0.85)
    axes[1].set_title("Local descriptor scattering")
    axes[1].set_xlabel("scattering component")
    axes[1].set_ylabel("sampled points")
    axes[1].grid(axis="y", alpha=0.22)
    figure.suptitle("Cat3D representation statistics", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150); plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot pilot and Cat3D representation analyses")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--cat-obj", default=None)
    args = parser.parse_args()
    root = Path(args.output_root)
    candidates = sorted(Path("data").rglob("*.obj")) if args.cat_obj is None else [Path(args.cat_obj)]
    if not candidates:
        raise FileNotFoundError("No Cat3D OBJ found under data/")
    obj_path = candidates[0]
    print(make_representation_analysis(root)[0])
    print(make_cat3d_projection_analysis(obj_path, root / "cat3d" / "cat3d_projection_analysis.png"))
    print(make_cat3d_statistics(obj_path, root / "cat3d" / "cat3d_statistics.png"))


if __name__ == "__main__":
    main()
