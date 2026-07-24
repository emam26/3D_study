"""Dataset adapters with the common RGBDSample contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .types import RGBDSample


class NYUv2Adapter:
    name = "nyuv2"

    def __init__(self, root: str | Path, split="test", intrinsics=None, ignore_labels=(0, 255)):
        self.root = Path(root); self.split = split
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.depth_dir = self.root / "depth" / split
        self.ignore_labels = tuple(ignore_labels)
        self.intrinsics = np.asarray(intrinsics or [518.857901, 519.469611, 325.582449, 253.736166], dtype=np.float32)
        if self.intrinsics.size == 4:
            fx, fy, cx, cy = self.intrinsics.tolist()
            self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        self.image_files = sorted(self.image_dir.glob("*.png"))
        if not self.image_files or not self.label_dir.exists():
            raise FileNotFoundError(f"NYUv2 prepared layout missing under {self.root}")

    def __len__(self): return len(self.image_files)

    def sample_ids(self): return [path.stem for path in self.image_files]

    def load(self, sample_id: str) -> RGBDSample:
        path = self.image_dir / f"{sample_id}.png"
        label_path = self.label_dir / path.name
        depth_path = self.depth_dir / path.name
        if not path.exists() or not label_path.exists() or not depth_path.exists():
            raise FileNotFoundError(f"Missing aligned NYUv2 files for {sample_id}")
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
        valid_label = np.ones_like(label, dtype=bool)
        for value in self.ignore_labels: valid_label &= label != value
        return RGBDSample(
            sample_id, self.name, rgb, depth, label, self.intrinsics.copy(),
            valid_depth_mask=np.isfinite(depth) & (depth > 0), valid_label_mask=valid_label,
            metadata={"split": self.split, "depth_scale": "millimetres_to_metres"},
        )


class SUNRGBDAdapter:
    """Reserved common-contract adapter; requires an explicit prepared layout."""

    name = "sunrgbd"

    def __init__(self, root, *args, **kwargs):
        raise NotImplementedError(
            "SUN RGB-D adapter is intentionally gated until its per-sample calibration layout is configured"
        )


def build_adapter(config):
    if config["name"].lower() in ("nyu", "nyuv2", "nyu_depth_v2"):
        return NYUv2Adapter(
            config["root"], config.get("split", "test"),
            config.get("intrinsics"), config.get("ignore_labels", [0, 255]),
        )
    if config["name"].lower() in ("sun", "sunrgbd", "sun_rgbd"):
        return SUNRGBDAdapter(config["root"])
    raise ValueError(f"Unsupported dataset: {config['name']}")


