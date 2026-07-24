"""Render a raw-geometry grid for a dataset sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .datasets import build_adapter
from .oracle import assign_oracle_labels
from .pipeline import load_config
from .representations import build_representation


REPRESENTATION_ORDER = ("pointcloud", "surfel", "mesh", "voxel", "tsdf", "superpoint",
                        "graph", "octree", "descriptor")


def _subsample(values, limit):
    if len(values) <= limit:
        return np.arange(len(values), dtype=np.int64)
    return np.linspace(0, len(values) - 1, limit, dtype=np.int64)


def _set_equal_axes(axis, points):
    if len(points) == 0:
        return
    mins = points.min(axis=0); maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _plot_geometry(axis, result, max_points=12000, max_faces=12000):
    geometry = result.geometry[result.valid]
    labels = result.attributes.get("oracle_label", np.zeros(result.element_count, dtype=np.int32))[result.valid]
    if len(geometry) == 0:
        axis.set_title(f"{result.name} (empty)")
        return

    if result.name == "mesh" and "vertices" in result.attributes and "faces" in result.attributes:
        vertices = result.attributes["vertices"]
        faces = result.attributes["faces"]
        face_ids = _subsample(faces, max_faces)
        selected_faces = faces[face_ids]
        unique_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
        mesh_vertices = vertices[unique_vertices]
        mesh_faces = inverse.reshape(-1, 3)
        axis.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
                          triangles=mesh_faces, cmap="tab20", linewidth=0.05,
                          alpha=0.75, antialiased=False)
        _set_equal_axes(axis, mesh_vertices)
        axis.set_title(f"mesh ({len(faces):,} triangles)")
        return

    if result.name == "graph" and result.adjacency is not None:
        ids = _subsample(geometry, max_points)
        points = geometry[ids]
        labels_for_points = labels[ids]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels_for_points,
                     cmap="tab20", s=4, alpha=0.85)
        selected = set(ids.tolist())
        edges = result.adjacency
        edge_ids = _subsample(edges, min(len(edges), 8000))
        for source, target in edges[edge_ids]:
            if int(source) in selected and int(target) in selected:
                segment = geometry[[source, target]]
                axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="0.35", linewidth=0.25, alpha=0.16)
        _set_equal_axes(axis, points)
        axis.set_title(f"graph ({result.element_count:,} nodes / {len(edges):,} edges)")
        return

    if result.name == "octree" and "bounds_min" in result.attributes:
        ids = _subsample(geometry, min(len(geometry), 600))
        points = geometry[ids]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels[ids], cmap="tab20", s=5, alpha=0.8)
        bounds_min = result.attributes["bounds_min"][ids]
        bounds_max = result.attributes["bounds_max"][ids]
        for lower, upper in zip(bounds_min, bounds_max):
            corners = np.array([[lower[0], lower[1], lower[2]], [upper[0], lower[1], lower[2]],
                                [upper[0], upper[1], lower[2]], [lower[0], upper[1], lower[2]],
                                [lower[0], lower[1], upper[2]], [upper[0], lower[1], upper[2]],
                                [upper[0], upper[1], upper[2]], [lower[0], upper[1], upper[2]]])
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)):
                axis.plot(corners[[a, b], 0], corners[[a, b], 1], corners[[a, b], 2],
                          color="0.35", linewidth=0.22, alpha=0.12)
        _set_equal_axes(axis, points)
        axis.set_title(f"octree ({result.element_count:,} leaves)")
        return

    ids = _subsample(geometry, max_points)
    points = geometry[ids]
    colors = labels[ids]
    color_kwargs = {"c": colors, "cmap": "tab20"}
    if result.name == "descriptor" and "descriptor_scalar" in result.attributes:
        color_kwargs = {"c": result.attributes["descriptor_scalar"][result.valid][ids], "cmap": "viridis"}
    if result.name == "tsdf":
        states = result.attributes.get("state", np.asarray(["surface"] * result.element_count))[result.valid][ids]
        color = np.where(states == "surface", colors, -1)
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, cmap="tab20", s=2, alpha=0.65)
    else:
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], **color_kwargs, s=3, alpha=0.7)
    if result.name == "surfel" and "normals" in result.attributes:
        normals = result.attributes["normals"][result.valid][ids]
        axis.quiver(points[:, 0], points[:, 1], points[:, 2], normals[:, 0], normals[:, 1], normals[:, 2],
                    length=0.025, normalize=True, linewidth=0.25, alpha=0.18)
    _set_equal_axes(axis, points)
    suffix = " (local geometry descriptor)" if result.name == "descriptor" else ""
    axis.set_title(f"{result.name} ({result.element_count:,} elements){suffix}")


def make_3d_grid(config, sample_id=None, output=None):
    adapter = build_adapter(config["dataset"])
    sample_id = sample_id or adapter.sample_ids()[0]
    sample = adapter.load(sample_id)
    representations = [name for name in REPRESENTATION_ORDER
                       if config.get("representations", {}).get(name, {}).get("enabled", True)]
    ncols = 4
    nslots = len(representations) + 2
    nrows = int(np.ceil(nslots / ncols))
    figure = plt.figure(figsize=(22, 5.5 * nrows))
    rgb_axis = figure.add_subplot(nrows, ncols, 1)
    rgb_axis.imshow(sample.rgb)
    rgb_axis.set_title("RGB reference")
    rgb_axis.axis("off")
    gt_axis = figure.add_subplot(nrows, ncols, 2)
    gt_axis.imshow(sample.semantic_gt, cmap="tab20")
    gt_axis.set_title("Semantic ground truth")
    gt_axis.axis("off")
    for index, name in enumerate(representations, 1):
        axis = figure.add_subplot(nrows, ncols, index + 2, projection="3d")
        result = build_representation(name, sample, config.get("representations", {}).get(name, {}))
        result = assign_oracle_labels(result, sample)
        _plot_geometry(axis, result)
        axis.set_xlabel("X"); axis.set_ylabel("Y"); axis.set_zlabel("Z")
        axis.view_init(elev=22, azim=-62)
    figure.suptitle(f"{sample.dataset_name}: raw 3D representations", fontsize=16)
    figure.tight_layout()
    if output is None:
        output = Path(config["study"].get("output_root", "outputs")) / config["dataset"]["name"] / config["study"]["name"] / "3d_representation_grid.png"
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140); plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render a raw 3D representation grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = make_3d_grid(load_config(args.config), args.sample_id, args.output)
    print(path)


if __name__ == "__main__":
    main()
