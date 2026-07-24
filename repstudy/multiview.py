"""Deterministic virtual-camera projections of visible geometry."""

from __future__ import annotations

import numpy as np

from .types import RGBDSample, RenderResult, RepresentationResult


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def render_virtual_view(result: RepresentationResult, sample: RGBDSample, view: str) -> RenderResult:
    if view == "original":
        from .oracle import render_original_view
        return render_original_view(result, sample)
    angles = {"left_oblique": -0.35, "right_oblique": 0.35, "elevated": 0.30}
    if view not in angles:
        raise KeyError(f"Unknown virtual view {view}")
    rotation = _rotation_y(angles[view])
    xyz = (rotation @ result.geometry.T).T
    valid = result.valid & (xyz[:, 2] > 1e-5)
    h, w = sample.depth_m.shape
    u = np.rint(sample.intrinsics[0, 0] * xyz[:, 0] / np.maximum(xyz[:, 2], 1e-6) + sample.intrinsics[0, 2]).astype(int)
    v = np.rint(sample.intrinsics[1, 1] * xyz[:, 1] / np.maximum(xyz[:, 2], 1e-6) + sample.intrinsics[1, 2]).astype(int)
    element_id = np.full((h, w), -1, np.int32); depth = np.full((h, w), np.inf, np.float32); collision = np.zeros((h, w), np.int32)
    labels = result.attributes.get("oracle_label", np.zeros(result.element_count, np.int32))
    for index in np.flatnonzero(valid):
        if not (0 <= u[index] < w and 0 <= v[index] < h): continue
        collision[v[index], u[index]] += 1
        if xyz[index, 2] < depth[v[index], u[index]]:
            depth[v[index], u[index]] = xyz[index, 2]; element_id[v[index], u[index]] = index
    coverage = element_id >= 0; semantic = np.zeros((h, w), np.int32); semantic[coverage] = labels[element_id[coverage]]; depth[~coverage] = 0
    return RenderResult(semantic, coverage, collision, element_id, depth, {"view": view, "representation": result.name})


def render_views(result, sample, views=None):
    views = views or ["original", "left_oblique", "right_oblique", "elevated"]
    return {view: render_virtual_view(result, sample, view) for view in views}


