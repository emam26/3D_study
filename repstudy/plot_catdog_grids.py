"""Create representation grids for the catDog RGB image and its OBJ asset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image

from .obj_utils import (face_centers, face_normals, knn_graph, load_obj,
                        local_descriptors, normalize_vertices, octree_leaves,
                        sample_indices, voxelize)


def _equal_axes(axis, points):
    points = np.asarray(points)
    if len(points) == 0:
        return
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _mesh_plot(axis, vertices, faces, max_faces=12000):
    ids = sample_indices(len(faces), max_faces, seed=42)
    selected = faces[ids]
    unique, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    points = vertices[unique]
    triangles = inverse.reshape(-1, 3)
    axis.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles,
                      cmap="Greys", linewidth=0.08, alpha=0.88, antialiased=True, shade=True)
    _equal_axes(axis, vertices)
    axis.set_title(f"Mesh / CAD ({len(faces):,} triangles)")


def _box(axis, lower, upper):
    corners = np.array([
        [lower[0], lower[1], lower[2]], [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]], [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]], [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]], [lower[0], upper[1], upper[2]],
    ])
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)):
        axis.plot(corners[[a, b], 0], corners[[a, b], 1], corners[[a, b], 2],
                  color="0.38", linewidth=0.3, alpha=0.16)


def _build_asset(vertices, faces):
    point_ids = sample_indices(len(vertices), 6000, seed=42)
    points = vertices[point_ids]
    voxel_centers, voxel_members = voxelize(vertices, voxel_size=0.045)
    octree = octree_leaves(vertices, max_depth=7, max_points_per_leaf=64)
    graph_ids = sample_indices(len(vertices), 1800, seed=7)
    graph_points = vertices[graph_ids]
    graph_edges = knn_graph(graph_points, k=7)
    descriptors = local_descriptors(points, k=16)
    centers = face_centers(vertices, faces)
    normals = face_normals(vertices, faces)
    return {
        "points": points,
        "voxel_centers": voxel_centers,
        "octree": octree,
        "graph_points": graph_points,
        "graph_edges": graph_edges,
        "descriptors": descriptors,
        "normal_centers": centers,
        "normal_vectors": normals,
    }


def make_3d_grid(image_path, obj_path, output):
    image = Image.open(image_path).convert("RGB")
    mesh = load_obj(obj_path)
    vertices = normalize_vertices(mesh.vertices)
    faces = mesh.faces
    asset = _build_asset(vertices, faces)

    figure = plt.figure(figsize=(22, 11))
    axes = []
    image_axis = figure.add_subplot(2, 4, 1)
    image_axis.imshow(image)
    image_axis.set_title("catDog RGB scene")
    image_axis.axis("off")

    axis = figure.add_subplot(2, 4, 2, projection="3d")
    _mesh_plot(axis, vertices, faces)
    axis.view_init(elev=20, azim=-65)
    axes.append(axis)

    axis = figure.add_subplot(2, 4, 3, projection="3d")
    axis.scatter(asset["points"][:, 0], asset["points"][:, 1], asset["points"][:, 2], s=2, alpha=0.55, c="royalblue")
    _equal_axes(axis, vertices); axis.set_title(f"Point cloud ({len(asset['points']):,} points)")
    axis.view_init(elev=20, azim=-65); axes.append(axis)

    axis = figure.add_subplot(2, 4, 4, projection="3d")
    axis.scatter(asset["voxel_centers"][:, 0], asset["voxel_centers"][:, 1], asset["voxel_centers"][:, 2],
                 s=5, alpha=0.55, c=asset["voxel_centers"][:, 2], cmap="gray")
    _equal_axes(axis, vertices); axis.set_title(f"Voxel ({len(asset['voxel_centers']):,} cells)")
    axis.view_init(elev=20, azim=-65); axes.append(axis)

    axis = figure.add_subplot(2, 4, 5, projection="3d")
    tree = asset["octree"]
    ids = sample_indices(len(tree["centers"]), 650, seed=42)
    axis.scatter(tree["centers"][ids, 0], tree["centers"][ids, 1], tree["centers"][ids, 2], s=5, c="darkgoldenrod", alpha=0.7)
    for index in ids:
        _box(axis, tree["bounds_min"][index], tree["bounds_max"][index])
    _equal_axes(axis, vertices); axis.set_title(f"Octree ({len(tree['centers']):,} leaves)")
    axis.view_init(elev=20, azim=-65); axes.append(axis)

    axis = figure.add_subplot(2, 4, 6, projection="3d")
    points, edges = asset["graph_points"], asset["graph_edges"]
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=3, c="black", alpha=0.7)
    for source, target in edges[sample_indices(len(edges), min(len(edges), 6000), seed=4)]:
        segment = points[[source, target]]
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="0.35", linewidth=0.22, alpha=0.18)
    _equal_axes(axis, vertices); axis.set_title(f"Graph ({len(points):,} nodes / {len(edges):,} edges)")
    axis.view_init(elev=20, azim=-65); axes.append(axis)

    axis = figure.add_subplot(2, 4, 7, projection="3d")
    scalar = asset["descriptors"][:, 5]
    axis.scatter(asset["points"][:, 0], asset["points"][:, 1], asset["points"][:, 2], s=3, c=scalar, cmap="turbo", alpha=0.8)
    _equal_axes(axis, vertices); axis.set_title("Descriptor (local scattering)")
    axis.view_init(elev=20, azim=-65); axes.append(axis)

    axis = figure.add_subplot(2, 4, 8, projection="3d")
    normal_ids = sample_indices(len(asset["normal_centers"]), 1800, seed=13)
    centers = asset["normal_centers"][normal_ids]
    normals = asset["normal_vectors"][normal_ids]
    axis.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=2, c="lightgray", alpha=0.5)
    axis.quiver(centers[:, 0], centers[:, 1], centers[:, 2], normals[:, 0], normals[:, 1], normals[:, 2],
                length=0.025, normalize=True, linewidth=0.25, color="teal", alpha=0.35)
    _equal_axes(axis, vertices); axis.set_title("Surface normals")
    axis.view_init(elev=20, azim=-65); axes.append(axis)

    for axis in axes:
        axis.set_xlabel("X"); axis.set_ylabel("Y"); axis.set_zlabel("Z")
    figure.suptitle("catDog scene with standalone Cat OBJ representations", fontsize=16)
    figure.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140); plt.close(figure)
    return output


def _project(vertices, view):
    if view == "front":
        return vertices[:, [0, 1]], vertices[:, 2]
    if view == "side":
        return vertices[:, [2, 1]], vertices[:, 0]
    if view == "top":
        return vertices[:, [0, 2]], vertices[:, 1]
    angle = np.deg2rad(28.0)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
    rotated = (rotation @ vertices.T).T
    return rotated[:, [0, 1]], rotated[:, 2]


def _projected_mesh(axis, vertices, faces, view, title):
    ids = sample_indices(len(faces), 9000, seed=11)
    triangles = vertices[faces[ids]]
    projected, depth = _project(triangles.reshape(-1, 3), view)
    polygons = projected.reshape(-1, 3, 2)
    collection = PolyCollection(polygons, array=depth.reshape(-1, 3).mean(axis=1), cmap="viridis",
                                edgecolors="none", alpha=0.9)
    axis.add_collection(collection)
    mins, maxs = projected.min(axis=0), projected.max(axis=0)
    axis.set_xlim(mins[0], maxs[0]); axis.set_ylim(mins[1], maxs[1]); axis.set_aspect("equal")
    axis.set_title(title); axis.axis("off")


def make_2d_grid(image_path, obj_path, output):
    image = Image.open(image_path).convert("RGB")
    mesh = load_obj(obj_path)
    vertices = normalize_vertices(mesh.vertices)
    faces = mesh.faces
    points = vertices[sample_indices(len(vertices), 10000, seed=42)]
    descriptors = local_descriptors(points, k=16)
    figure, axes = plt.subplots(2, 4, figsize=(22, 11), squeeze=False)
    axes[0, 0].imshow(image); axes[0, 0].set_title("catDog RGB scene"); axes[0, 0].axis("off")
    _projected_mesh(axes[0, 1], vertices, faces, "front", "OBJ front projection")
    _projected_mesh(axes[0, 2], vertices, faces, "side", "OBJ side projection")
    _projected_mesh(axes[0, 3], vertices, faces, "top", "OBJ top projection")
    _projected_mesh(axes[1, 0], vertices, faces, "perspective", "OBJ perspective projection")
    projected, depth = _project(points, "front")
    axes[1, 1].scatter(projected[:, 0], projected[:, 1], c=depth, s=2, cmap="magma", alpha=0.6)
    axes[1, 1].set_aspect("equal"); axes[1, 1].set_title("OBJ-derived depth"); axes[1, 1].axis("off")
    projected, _ = _project(points, "front")
    axes[1, 2].scatter(projected[:, 0], projected[:, 1], s=2, c="black", alpha=0.55)
    axes[1, 2].set_aspect("equal"); axes[1, 2].set_title("OBJ silhouette"); axes[1, 2].axis("off")
    axes[1, 3].scatter(projected[:, 0], projected[:, 1], c=descriptors[:, 5], s=2, cmap="turbo", alpha=0.7)
    axes[1, 3].set_aspect("equal"); axes[1, 3].set_title("Descriptor projection"); axes[1, 3].axis("off")
    figure.suptitle("catDog 2D projections (OBJ asset is not camera-aligned to the RGB scene)", fontsize=16)
    figure.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140); plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render catDog RGB and Cat OBJ representation grids")
    parser.add_argument("--image", default="data/catDog.png")
    parser.add_argument("--obj", default=None, help="OBJ path; defaults to the first OBJ under data/")
    parser.add_argument("--output-root", default="outputs/catdog")
    args = parser.parse_args()
    image_path = Path(args.image)
    if args.obj:
        obj_path = Path(args.obj)
    else:
        candidates = sorted(Path("data").rglob("*.obj"))
        if not candidates:
            raise FileNotFoundError("No OBJ found under data/")
        obj_path = candidates[0]
    root = Path(args.output_root)
    print(make_3d_grid(image_path, obj_path, root / "catdog_3d_representation_grid.png"))
    print(make_2d_grid(image_path, obj_path, root / "catdog_2d_representation_grid.png"))
    print(f"OBJ: {obj_path}")


if __name__ == "__main__":
    main()
