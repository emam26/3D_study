"""Deterministic oracle labels and original-camera z-buffer rendering."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .types import RGBDSample, RenderResult, RepresentationResult


def majority_label(labels: np.ndarray, valid: np.ndarray | None = None) -> tuple[int, float, float]:
    values = np.asarray(labels).reshape(-1)
    if valid is not None:
        values = values[np.asarray(valid).reshape(-1)]
    values = values[values > 0]
    if values.size == 0:
        return 0, 0.0, 0.0
    counts = Counter(values.tolist())
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / values.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return int(label), float(count / values.size), entropy


def assign_oracle_labels(result: RepresentationResult, sample: RGBDSample) -> RepresentationResult:
    labels, purity, entropy, crossing = [], [], [], []
    for members in result.original_pixel_members:
        pixels = np.asarray(members, dtype=np.int32)
        if pixels.size == 0:
            labels.append(0); purity.append(0.0); entropy.append(0.0); crossing.append(False); continue
        values = sample.semantic_gt[pixels[:, 0], pixels[:, 1]]
        label, score, ent = majority_label(values, sample.valid_label_mask[pixels[:, 0], pixels[:, 1]])
        labels.append(label); purity.append(score); entropy.append(ent)
        crossing.append(np.unique(values[values > 0]).size > 1)
    result.attributes["oracle_label"] = np.asarray(labels, dtype=np.int32)
    result.attributes["purity"] = np.asarray(purity, dtype=np.float32)
    result.attributes["entropy"] = np.asarray(entropy, dtype=np.float32)
    result.attributes["boundary_crossing"] = np.asarray(crossing, dtype=bool)
    return result


def render_original_view(result: RepresentationResult, sample: RGBDSample) -> RenderResult:
    h, w = sample.depth_m.shape
    element_id = np.full((h, w), -1, dtype=np.int32)
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    collisions = np.zeros((h, w), dtype=np.int32)
    for index, (xyz, members) in enumerate(zip(result.geometry, result.original_pixel_members)):
        if not result.valid[index]:
            continue
        for v, u in np.asarray(members, dtype=np.int32):
            collisions[v, u] += 1
            if xyz[2] < depth_map[v, u]:
                depth_map[v, u] = xyz[2]; element_id[v, u] = index
    labels = result.attributes.get("oracle_label", np.zeros(result.element_count, dtype=np.int32))
    semantic = np.zeros((h, w), dtype=np.int32)
    covered = element_id >= 0
    semantic[covered] = labels[element_id[covered]]
    depth_map[~covered] = 0
    return RenderResult(
        semantic, covered, collisions, element_id, depth_map,
        {"representation": result.name},
    )


