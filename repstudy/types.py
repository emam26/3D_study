from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RGBDSample:
    sample_id: str
    dataset_name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    semantic_gt: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray | None = None
    valid_depth_mask: np.ndarray | None = None
    valid_label_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape HxWx3")
        h, w = self.rgb.shape[:2]
        if self.depth_m.shape != (h, w) or self.semantic_gt.shape != (h, w):
            raise ValueError("rgb, depth_m, and semantic_gt must be spatially aligned")
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        if self.intrinsics.shape != (3, 3) or abs(np.linalg.det(self.intrinsics)) < 1e-8:
            raise ValueError("intrinsics must be a non-singular 3x3 matrix")
        if self.valid_depth_mask is None:
            self.valid_depth_mask = np.isfinite(self.depth_m) & (self.depth_m > 0)
        if self.valid_label_mask is None:
            self.valid_label_mask = np.isfinite(self.semantic_gt) & (self.semantic_gt > 0)
        self.valid_depth_mask = np.asarray(self.valid_depth_mask, dtype=bool)
        self.valid_label_mask = np.asarray(self.valid_label_mask, dtype=bool)


@dataclass
class RepresentationResult:
    name: str
    geometry: np.ndarray
    original_pixel_members: list[np.ndarray]
    valid: np.ndarray
    attributes: dict[str, np.ndarray] = field(default_factory=dict)
    adjacency: np.ndarray | None = None
    construction_parameters: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, float] = field(default_factory=dict)

    @property
    def element_count(self) -> int:
        return int(self.geometry.shape[0])


@dataclass
class RenderResult:
    semantic_map: np.ndarray
    coverage_mask: np.ndarray
    collision_map: np.ndarray
    element_id_map: np.ndarray
    depth_map: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


