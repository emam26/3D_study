"""Canonical RGB-D geometry preprocessing shared by all representations."""

from __future__ import annotations

import numpy as np

from .types import RGBDSample


def backproject(sample: RGBDSample) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid depth pixels to camera XYZ and return `(xyz, pixels)`."""
    valid = sample.valid_depth_mask
    v, u = np.nonzero(valid)
    z = sample.depth_m[v, u].astype(np.float32)
    fx, fy = sample.intrinsics[0, 0], sample.intrinsics[1, 1]
    cx, cy = sample.intrinsics[0, 2], sample.intrinsics[1, 2]
    xyz = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    pixels = np.stack((v, u), axis=1).astype(np.int32)
    return xyz, pixels


def normals_from_xyz(xyz_image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Finite-difference camera normals; invalid neighborhoods return zero."""
    h, w, _ = xyz_image.shape
    normals = np.zeros_like(xyz_image, dtype=np.float32)
    if h < 2 or w < 2:
        return normals
    dx = xyz_image[:, 1:] - xyz_image[:, :-1]
    dy = xyz_image[1:] - xyz_image[:-1]
    cross = np.cross(dx[:-1], dy[:, :-1])
    support = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
    norm = np.linalg.norm(cross, axis=-1, keepdims=True).clip(1e-8)
    normals[:-1, :-1] = np.where(support[..., None], cross / norm, 0)
    return normals


def xyz_image(sample: RGBDSample) -> np.ndarray:
    h, w = sample.depth_m.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    xyz, pixels = backproject(sample)
    image[pixels[:, 0], pixels[:, 1]] = xyz
    return image


def geometry_attributes(sample: RGBDSample) -> dict[str, np.ndarray]:
    xyz = xyz_image(sample)
    normals = normals_from_xyz(xyz, sample.valid_depth_mask)
    depth = sample.depth_m.astype(np.float32)
    gx = np.zeros_like(depth); gy = np.zeros_like(depth)
    gx[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
    gy[1:] = np.abs(depth[1:] - depth[:-1])
    depth_edge = ((gx + gy) / np.maximum(depth, 1e-3)).clip(0, 1)
    rgb = sample.rgb.astype(np.float32) / (255.0 if sample.rgb.max() > 1.5 else 1.0)
    gray = rgb.mean(axis=-1)
    rgx = np.zeros_like(gray); rgy = np.zeros_like(gray)
    rgx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    rgy[1:] = np.abs(gray[1:] - gray[:-1])
    return {
        "xyz_image": xyz,
        "normals_image": normals,
        "depth_edge": (depth_edge * sample.valid_depth_mask).astype(np.float32),
        "rgb_edge": np.clip(rgx + rgy, 0, 1).astype(np.float32),
        "valid": sample.valid_depth_mask.astype(np.uint8),
    }


