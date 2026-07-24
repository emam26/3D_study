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
            {"faces": np.asarray(faces, dtype=np.int32)}, None,
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
        xyz, pixels = backproject(sample)
        cells: dict[tuple[int, int, int], dict[str, Any]] = {}
        origin = np.zeros(3, dtype=np.float32)
        for point, pixel in zip(xyz, pixels):
            distance = np.linalg.norm(point)
            steps = max(1, int(max(distance - truncation, 0) / size))
            for alpha in np.linspace(0, 1, steps, endpoint=False):
                location = point * alpha
                key = tuple(np.floor(location / size).astype(np.int32))
                cells.setdefault(key, {"state": "free", "members": []})
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
            {"voxel_size_m": size, "truncation_m": truncation, "unobserved_is_omitted": True},
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


REPRESENTATIONS = {
    "original_2d": Original2DControl,
    "pointcloud": OrganizedPointCloud,
    "surfel": Surfels,
    "mesh": PartialMesh,
    "voxel": SparseVoxels,
    "tsdf": SparseTSDF,
    "superpoint": SuperpointRegions,
}


def build_representation(name: str, sample: RGBDSample, config=None) -> RepresentationResult:
    if name not in REPRESENTATIONS:
        raise KeyError(f"Unknown representation {name}; choose from {sorted(REPRESENTATIONS)}")
    return REPRESENTATIONS[name]().build(sample, config)


