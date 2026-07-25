"""Small, dependency-light utilities for visualizing standalone OBJ assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OBJMesh:
    vertices: np.ndarray
    faces: np.ndarray


def load_obj(path) -> OBJMesh:
    """Load OBJ positions and triangulated faces (materials/UVs are optional)."""
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                indices = []
                for token in tokens:
                    raw = token.split("/")[0]
                    index = int(raw)
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for index in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[index], indices[index + 1]))
    mesh = OBJMesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32).reshape(-1, 3))
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"OBJ contains no renderable vertices/faces: {path}")
    return mesh


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    scale = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    return (vertices - center) / max(scale, 1e-6)


def sample_indices(count: int, budget: int, seed: int = 42) -> np.ndarray:
    if count <= budget:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, int(budget), replace=False))


def face_centers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces].mean(axis=1).astype(np.float32)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(lengths, 1e-8)).astype(np.float32)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-averaged vertex normals for surfel rendering."""
    normals = face_normals(vertices, faces)
    output = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(output, faces[:, corner], normals)
    lengths = np.linalg.norm(output, axis=1, keepdims=True)
    return (output / np.maximum(lengths, 1e-8)).astype(np.float32)


def knn_graph(points: np.ndarray, k: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 1:
        return np.zeros((0, 2), dtype=np.int32)
    k = min(max(int(k), 1), len(points) - 1)
    from scipy.spatial import cKDTree
    _, neighbours = cKDTree(points).query(points, k=k + 1)
    edges = set()
    for source, row in enumerate(np.asarray(neighbours)[:, 1:]):
        for target in row:
            edges.add(tuple(sorted((source, int(target)))))
    return np.asarray(sorted(edges), dtype=np.int32).reshape(-1, 2)


def voxelize(points: np.ndarray, voxel_size: float = 0.04) -> tuple[np.ndarray, list[np.ndarray]]:
    points = np.asarray(points, dtype=np.float32)
    keys = np.floor(points / float(voxel_size)).astype(np.int32)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, keys)):
        groups.setdefault(key, []).append(index)
    ordered = sorted(groups)
    centers = np.asarray([(np.asarray(key, dtype=np.float32) + 0.5) * voxel_size for key in ordered], dtype=np.float32)
    members = [np.asarray(groups[key], dtype=np.int32) for key in ordered]
    return centers, members


def octree_leaves(points: np.ndarray, max_depth: int = 7, max_points_per_leaf: int = 64) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0)
    side = max(float(np.max(points.max(axis=0) - mins)), 1e-4)
    pad = max(side * 1e-5, 1e-5)
    root_min = mins - pad
    root_side = side + 2 * pad
    stack = [(np.arange(len(points), dtype=np.int32), root_min, root_side, 0)]
    leaves = []
    while stack:
        indices, origin, cell_side, depth = stack.pop()
        if len(indices) <= max_points_per_leaf or depth >= max_depth:
            leaves.append((indices, origin.copy(), float(cell_side), depth))
            continue
        half = cell_side / 2.0
        midpoint = origin + half
        bits = (points[indices] >= midpoint).astype(np.int8)
        child_ids = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
        for child in range(7, -1, -1):
            mask = child_ids == child
            if np.any(mask):
                offset = bits[mask][0].astype(np.float32) * half
                stack.append((indices[mask], origin + offset, half, depth + 1))
    leaves.sort(key=lambda item: (item[3], float(item[1][0]), float(item[1][1]), float(item[1][2])))
    return {
        "centers": np.asarray([origin + cell_side / 2 for _, origin, cell_side, _ in leaves], dtype=np.float32),
        "members": [indices for indices, _, _, _ in leaves],
        "bounds_min": np.asarray([origin for _, origin, _, _ in leaves], dtype=np.float32),
        "bounds_max": np.asarray([origin + cell_side for _, origin, cell_side, _ in leaves], dtype=np.float32),
        "levels": np.asarray([depth for _, _, _, depth in leaves], dtype=np.int32),
    }


def local_descriptors(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Return compact covariance descriptors: eigenvalues, shape, density."""
    points = np.asarray(points, dtype=np.float32)
    descriptors = np.zeros((len(points), 7), dtype=np.float32)
    if len(points) <= 1:
        return descriptors
    from scipy.spatial import cKDTree
    k = min(max(int(k), 2), len(points) - 1)
    _, neighbours = cKDTree(points).query(points, k=k + 1)
    for index, row in enumerate(np.asarray(neighbours)[:, 1:]):
        local = points[row] - points[index]
        covariance = (local.T @ local) / max(len(local), 1)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0))[::-1]
        trace = float(eigenvalues.sum())
        l1, l2, l3 = eigenvalues / max(trace, 1e-8)
        mean_distance = float(np.linalg.norm(local, axis=1).mean())
        descriptors[index] = [l1, l2, l3, (l1 - l2) / max(l1, 1e-8),
                              (l2 - l3) / max(l1, 1e-8), l3, 1.0 / max(mean_distance, 1e-4)]
    return descriptors
