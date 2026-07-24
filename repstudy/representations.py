"""Deterministic visible-surface representation builders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np

from .geometry import backproject, geometry_attributes
from .types import RGBDSample, RepresentationResult


class BaseRepresentation:
    name = "base"

    def build(self, sample: RGBDSample, config: dict[str, Any] | None = None) -> RepresentationResult:
        raise NotImplementedError


def _sample_indices(count: int, budget: int | None, seed: int) -> np.ndarray:
    if budget is None or budget >= count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, budget, replace=False))


def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Return deterministic k-nearest-neighbour indices for each point."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 1:
        return np.zeros((len(points), 0), dtype=np.int32)
    k = min(max(int(k), 1), max(len(points) - 1, 1))
    try:
        from scipy.spatial import cKDTree
        _, indices = cKDTree(points).query(points, k=k + 1)
        return np.asarray(indices[:, 1:], dtype=np.int32)
    except Exception:
        # The pilot requirements include SciPy. This fallback keeps the core
        # representation usable in minimal environments and small tests.
        distances = np.sum((points[:, None] - points[None, :]) ** 2, axis=-1)
        np.fill_diagonal(distances, np.inf)
        return np.argpartition(distances, kth=min(k - 1, len(points) - 1), axis=1)[:, :k].astype(np.int32)


class Original2DControl(BaseRepresentation):
    name = "original_2d"

    def build(self, sample, config=None):
        xyz, pixels = backproject(sample)
        return RepresentationResult(
            self.name, xyz, [p[None] for p in pixels], np.ones(len(xyz), dtype=bool),
            {"rgb": sample.rgb[pixels[:, 0], pixels[:, 1]]}, {}, {"identity": True},
        )


class OrganizedPointCloud(BaseRepresentation):
    name = "pointcloud"

    def build(self, sample, config=None):
        config = config or {}
        xyz, pixels = backproject(sample)
        indices = _sample_indices(len(xyz), config.get("budget"), config.get("seed", 42))
        attrs = geometry_attributes(sample)
        normals = attrs["normals_image"][pixels[:, 0], pixels[:, 1]][indices]
        rgb = sample.rgb[pixels[:, 0], pixels[:, 1]][indices]
        return RepresentationResult(
            self.name, xyz[indices], [p[None] for p in pixels[indices]],
            np.ones(len(indices), dtype=bool), {"rgb": rgb, "normals": normals},
            None, {"budget": config.get("budget"), "sampling": config.get("sampling", "random")},
        )


class Surfels(BaseRepresentation):
    name = "surfel"

    def build(self, sample, config=None):
        config = config or {}
        cloud = OrganizedPointCloud().build(sample, config)
        radius = config.get("radius_m", 0.02)
        if config.get("adaptive", False):
            radius = np.maximum(radius, cloud.geometry[:, 2] / sample.intrinsics[0, 0])
        cloud.name = self.name
        cloud.construction_parameters = {**cloud.construction_parameters, "radius_m": radius}
        cloud.attributes["radius_m"] = np.asarray(radius, dtype=np.float32)
        return cloud


class PartialMesh(BaseRepresentation):
    name = "mesh"

    def build(self, sample, config=None):
        config = config or {}
        xyz_image = geometry_attributes(sample)["xyz_image"]
        valid = sample.valid_depth_mask
        max_jump = config.get("max_depth_jump_m", 0.05)
        max_edge = config.get("max_edge_length_m", np.inf)
        angle = config.get("max_normal_angle_deg")
        normals = geometry_attributes(sample)["normals_image"]
        vertices, members, faces = [], [], []
        h, w = valid.shape
        for v in range(h - 1):
            for u in range(w - 1):
                cell = [(v, u), (v, u + 1), (v + 1, u), (v + 1, u + 1)]
                if not all(valid[a, b] for a, b in cell):
                    continue
                depths = [xyz_image[a, b, 2] for a, b in cell]
                if max(depths) - min(depths) > max_jump:
                    continue
                if max_edge < np.inf:
                    edges = [np.linalg.norm(xyz_image[cell[i]] - xyz_image[cell[j]])
                             for i, j in ((0, 1), (0, 2), (1, 3), (2, 3))]
                    if max(edges) > max_edge:
                        continue
                if angle is not None:
                    ns = [normals[a, b] for a, b in cell]
                    if min(np.linalg.norm(n) for n in ns) == 0:
                        continue
                    cosine = min(float(np.dot(ns[i], ns[j])) for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)))
                    if np.degrees(np.arccos(np.clip(cosine, -1, 1))) > angle:
                        continue
                for tri in ((0, 1, 3), (0, 3, 2)):
                    points = np.asarray([cell[index] for index in tri], dtype=np.int32)
                    faces.append(tuple(range(len(vertices), len(vertices) + 3)))
                    vertices.extend(xyz_image[points[:, 0], points[:, 1]])
                    members.append(points)
        if not vertices:
            return RepresentationResult(self.name, np.zeros((0, 3), np.float32), [], np.zeros(0, bool))
        vertices = np.asarray(vertices, dtype=np.float32)
        geometry = vertices.reshape(-1, 3, 3).mean(1)
        return RepresentationResult(
            self.name, geometry, members, np.ones(len(geometry), bool),
            {"vertices": vertices, "faces": np.asarray(faces, dtype=np.int32)}, None,
            {"max_depth_jump_m": max_jump, "max_edge_length_m": max_edge, "max_normal_angle_deg": angle},
        )


class SparseVoxels(BaseRepresentation):
    name = "voxel"

    def build(self, sample, config=None):
        config = config or {}
        size = float(config.get("voxel_size_m", 0.04))
        xyz, pixels = backproject(sample)
        keys = np.floor(xyz / size).astype(np.int32)
        groups = defaultdict(list)
        for index, key in enumerate(map(tuple, keys)):
            groups[key].append(index)
        geometry, members, colors, purity = [], [], [], []
        for key in sorted(groups):
            indices = np.asarray(groups[key], dtype=np.int64)
            geometry.append(xyz[indices].mean(0)); members.append(pixels[indices])
            colors.append(sample.rgb[pixels[indices, 0], pixels[indices, 1]].mean(0))
        return RepresentationResult(
            self.name, np.asarray(geometry, np.float32), members, np.ones(len(geometry), bool),
            {"voxel_key": np.asarray(sorted(groups), np.int32), "rgb": np.asarray(colors, np.float32)},
            None, {"voxel_size_m": size},
        )


class SparseTSDF(BaseRepresentation):
    name = "tsdf"

    def build(self, sample, config=None):
        config = config or {}
        size = float(config.get("voxel_size_m", 0.04))
        truncation = float(config.get("truncation_multiplier", 4.0)) * size
        ray_stride = max(1, int(config.get("ray_stride", 1)))
        xyz, pixels = backproject(sample)
        cells: dict[tuple[int, int, int], dict[str, Any]] = {}
        # Free-space integration is sampled deterministically for speed. Every
        # observed surface point is still retained below, so this does not
        # change the visible surface support or oracle coverage. Key generation
        # is batched across rays instead of constructing one linspace per pixel.
        ray_points = xyz[::ray_stride]
        distances = np.linalg.norm(ray_points, axis=1)
        ray_steps = np.maximum(1, ((np.maximum(distances - truncation, 0) / size)).astype(np.int32))
        free_keys = []
        for step in range(int(ray_steps.max(initial=1))):
            active = ray_steps > step
            if not np.any(active):
                continue
            alpha = (step / ray_steps[active]).astype(np.float32)
            locations = ray_points[active] * alpha[:, None]
            free_keys.append(np.floor(locations / size).astype(np.int32))
        if free_keys:
            for key in np.unique(np.concatenate(free_keys, axis=0), axis=0):
                cells[tuple(key)] = {"state": "free", "members": []}
        for point, pixel in zip(xyz, pixels):
            key = tuple(np.floor(point / size).astype(np.int32))
            cell = cells.setdefault(key, {"state": "surface", "members": []})
            cell["state"] = "surface"; cell["members"].append(pixel)
        geometry, members, states, tsdf = [], [], [], []
        for key in sorted(cells):
            cell = cells[key]; geometry.append((np.asarray(key) + 0.5) * size)
            members.append(np.asarray(cell["members"], np.int32)); states.append(cell["state"])
            tsdf.append(0.0 if cell["state"] == "surface" else 1.0)
        return RepresentationResult(
            self.name, np.asarray(geometry, np.float32), members,
            np.ones(len(geometry), bool),
            {"state": np.asarray(states), "tsdf": np.asarray(tsdf, np.float32)}, None,
            {"voxel_size_m": size, "truncation_m": truncation, "ray_stride": ray_stride,
             "unobserved_is_omitted": True},
        )


class SuperpointRegions(BaseRepresentation):
    name = "superpoint"

    def build(self, sample, config=None):
        config = config or {}
        target = int(config.get("target_regions", 1000))
        valid_pixels = np.argwhere(sample.valid_depth_mask)
        if len(valid_pixels) == 0:
            return RepresentationResult(self.name, np.zeros((0, 3), np.float32), [], np.zeros(0, bool))
        h, w = sample.depth_m.shape
        tile = max(1, int(np.ceil(np.sqrt((h * w) / max(target, 1)))))
        groups = defaultdict(list)
        for pixel in valid_pixels:
            groups[(int(pixel[0] // tile), int(pixel[1] // tile))].append(pixel)
        xyz_image = geometry_attributes(sample)["xyz_image"]
        geometry = [xyz_image[np.asarray(p)[:, 0], np.asarray(p)[:, 1]].mean(0) for p in groups.values()]
        members = [np.asarray(p, np.int32) for p in groups.values()]
        keys = list(groups)
        edges = []
        index = {key: i for i, key in enumerate(keys)}
        for key, i in index.items():
            for neighbor in ((key[0] + 1, key[1]), (key[0], key[1] + 1)):
                if neighbor in index: edges.append((i, index[neighbor]))
        return RepresentationResult(
            self.name, np.asarray(geometry, np.float32), members, np.ones(len(geometry), bool),
            {"region_key": np.asarray(keys, np.int32)}, np.asarray(edges, np.int32),
            {"target_regions": target, "tile_size_px": tile},
        )


class PointGraph(BaseRepresentation):
    """A kNN graph over a deterministic point-cloud sample."""

    name = "graph"

    def build(self, sample, config=None):
        config = config or {}
        xyz, pixels = backproject(sample)
        indices = _sample_indices(len(xyz), config.get("budget", 2500), config.get("seed", 42))
        points = xyz[indices]
        members = [p[None] for p in pixels[indices]]
        k = int(config.get("k", 8))
        neighbours = _knn_indices(points, k)
        edge_set = set()
        for source, row in enumerate(neighbours):
            for target in row:
                if source == int(target):
                    continue
                edge_set.add(tuple(sorted((source, int(target)))))
        edges = np.asarray(sorted(edge_set), dtype=np.int32).reshape(-1, 2)
        attrs = geometry_attributes(sample)
        normals = attrs["normals_image"][pixels[indices, 0], pixels[indices, 1]]
        rgb = sample.rgb[pixels[indices, 0], pixels[indices, 1]]
        degree = np.bincount(edges.reshape(-1), minlength=len(points)).astype(np.float32) if len(edges) else np.zeros(len(points), np.float32)
        return RepresentationResult(
            self.name, points, members, np.ones(len(points), bool),
            {"rgb": rgb, "normals": normals, "degree": degree}, edges,
            {"budget": config.get("budget", 2500), "k": k, "sampling": "seeded_uniform"},
        )


class OctreeLeaves(BaseRepresentation):
    """Adaptive octree leaves over the complete visible point cloud."""

    name = "octree"

    def build(self, sample, config=None):
        config = config or {}
        xyz, pixels = backproject(sample)
        if len(xyz) == 0:
            return RepresentationResult(self.name, np.zeros((0, 3), np.float32), [], np.zeros(0, bool))
        max_depth = max(1, int(config.get("max_depth", 7)))
        max_points = max(1, int(config.get("max_points_per_leaf", 128)))
        mins = xyz.min(axis=0).astype(np.float32)
        side = float(np.max(xyz.max(axis=0) - mins))
        side = max(side, 1e-3)
        # A tiny pad keeps points on the maximum boundary inside the root.
        pad = max(side * 1e-5, 1e-5)
        root_min = mins - pad
        root_side = side + 2 * pad
        stack = [(np.arange(len(xyz), dtype=np.int32), root_min, root_side, 0)]
        leaves = []
        while stack:
            idx, origin, cell_side, depth = stack.pop()
            if len(idx) <= max_points or depth >= max_depth:
                leaves.append((idx, origin.copy(), float(cell_side), depth))
                continue
            half = cell_side / 2.0
            midpoint = origin + half
            bits = (xyz[idx] >= midpoint).astype(np.int8)
            child_ids = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
            for child in range(7, -1, -1):
                child_mask = child_ids == child
                if np.any(child_mask):
                    offset = bits[child_mask][0].astype(np.float32) * half
                    stack.append((idx[child_mask], origin + offset, half, depth + 1))
        # Stable order makes summaries and visual comparisons reproducible.
        leaves.sort(key=lambda item: (item[3], float(item[1][0]), float(item[1][1]), float(item[1][2])))
        geometry = [origin + side_len / 2.0 for _, origin, side_len, _ in leaves]
        members = [pixels[idx] for idx, _, _, _ in leaves]
        bounds_min = np.asarray([origin for _, origin, _, _ in leaves], dtype=np.float32)
        bounds_max = np.asarray([origin + side_len for _, origin, side_len, _ in leaves], dtype=np.float32)
        levels = np.asarray([depth for _, _, _, depth in leaves], dtype=np.int32)
        counts = np.asarray([len(idx) for idx, _, _, _ in leaves], dtype=np.int32)
        return RepresentationResult(
            self.name, np.asarray(geometry, np.float32), members, np.ones(len(geometry), bool),
            {"bounds_min": bounds_min, "bounds_max": bounds_max, "level": levels, "point_count": counts}, None,
            {"max_depth": max_depth, "max_points_per_leaf": max_points, "root_min": root_min, "root_side_m": root_side},
        )


class GeometricDescriptors(BaseRepresentation):
    """Local geometry descriptors attached to sampled visible-surface points."""

    name = "descriptor"

    def build(self, sample, config=None):
        config = config or {}
        xyz, pixels = backproject(sample)
        indices = _sample_indices(len(xyz), config.get("budget", 5000), config.get("seed", 42))
        points = xyz[indices]
        members = [p[None] for p in pixels[indices]]
        if len(points) <= 1:
            descriptor = np.zeros((len(points), 10), dtype=np.float32)
        else:
            neighbours = _knn_indices(points, int(config.get("k", 16)))
            descriptor_rows = []
            rgb_pixels = sample.rgb[pixels[indices, 0], pixels[indices, 1]].astype(np.float32)
            rgb_pixels /= 255.0 if rgb_pixels.max(initial=0) > 1.5 else 1.0
            for point_index, neighbour_ids in enumerate(neighbours):
                local = points[neighbour_ids] - points[point_index]
                covariance = (local.T @ local) / max(len(local), 1)
                eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0))[::-1]
                trace = float(eigenvalues.sum())
                normalized = eigenvalues / max(trace, 1e-8)
                l1, l2, l3 = normalized
                mean_distance = float(np.linalg.norm(local, axis=1).mean())
                density = 1.0 / max(mean_distance, 1e-4)
                descriptor_rows.append(np.concatenate([
                    normalized, [((l1 - l2) / max(l1, 1e-8)), ((l2 - l3) / max(l1, 1e-8)),
                    l3, density, *rgb_pixels[point_index]],
                ]))
            descriptor = np.asarray(descriptor_rows, dtype=np.float32)
        attrs = geometry_attributes(sample)
        normals = attrs["normals_image"][pixels[indices, 0], pixels[indices, 1]]
        scalar = descriptor[:, 2] if len(descriptor) else np.zeros(0, np.float32)
        return RepresentationResult(
            self.name, points, members, np.ones(len(points), bool),
            {"descriptor": descriptor, "descriptor_scalar": scalar, "normals": normals}, None,
            {"budget": config.get("budget", 5000), "k": int(config.get("k", 16)),
             "descriptor_layout": ["lambda1", "lambda2", "lambda3", "linearity", "planarity", "scattering", "density", "r", "g", "b"]},
        )


REPRESENTATIONS = {
    "original_2d": Original2DControl,
    "pointcloud": OrganizedPointCloud,
    "surfel": Surfels,
    "mesh": PartialMesh,
    "voxel": SparseVoxels,
    "tsdf": SparseTSDF,
    "superpoint": SuperpointRegions,
    "graph": PointGraph,
    "octree": OctreeLeaves,
    "descriptor": GeometricDescriptors,
}


def build_representation(name: str, sample: RGBDSample, config=None) -> RepresentationResult:
    if name not in REPRESENTATIONS:
        raise KeyError(f"Unknown representation {name}; choose from {sorted(REPRESENTATIONS)}")
    return REPRESENTATIONS[name]().build(sample, config)
