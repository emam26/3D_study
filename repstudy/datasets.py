"""Dataset adapters with the common RGBDSample contract."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat

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
    """Adapter for the official SUN RGB-D release and its 37-class masks.

    The toolbox stores paths and segmentation masks in MATLAB files. The
    adapter converts those records into the same aligned RGB-D contract used
    by NYUv2; it does not create geometry or use labels during construction.
    """

    name = "sunrgbd"

    def __init__(self, root, split="all", label_set="37", ignore_labels=(0,), max_depth_m=8.0):
        self.root = Path(root)
        self.dataset_root = self.root / "SUNRGBD" if (self.root / "SUNRGBD").exists() else self.root
        toolbox = self.root / "SUNRGBDtoolbox" / "Metadata"
        if not toolbox.exists():
            toolbox = self.dataset_root.parent / "SUNRGBDtoolbox" / "Metadata"
        self.metadata_path = toolbox / "SUNRGBDMeta.mat"
        self.segmentation_path = toolbox / "SUNRGBD2Dseg.mat"
        if not self.metadata_path.exists() or not self.segmentation_path.exists():
            raise FileNotFoundError(
                "SUN RGB-D metadata missing; expected SUNRGBDMeta.mat and SUNRGBD2Dseg.mat "
                f"under {toolbox}"
            )
        if label_set not in ("37", 37):
            raise ValueError("This adapter currently supports the official 37-class masks only")
        self.split = split
        self.ignore_labels = tuple(ignore_labels)
        self.max_depth_m = float(max_depth_m)
        self.metadata = loadmat(self.metadata_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta"]
        names = loadmat(toolbox / "seg37list.mat", squeeze_me=True, struct_as_record=False)["seg37list"]
        self.class_names = [str(value) for value in np.asarray(names).reshape(-1)]
        self.records = []
        for index, item in enumerate(np.asarray(self.metadata).reshape(-1)):
            if not int(getattr(item, "valid", 1)):
                continue
            sequence = Path(str(item.sequenceName))
            if sequence.parts and sequence.parts[0].lower() == "sunrgbd":
                sequence = Path(*sequence.parts[1:])
            sample_root = self.dataset_root / sequence
            rgb_path = sample_root / "image" / str(item.rgbname)
            depth_path = sample_root / "depth" / str(item.depthname)
            if rgb_path.exists() and depth_path.exists():
                self.records.append({"index": index, "item": item, "root": sample_root,
                                     "rgb": rgb_path, "depth": depth_path})
        if not self.records:
            raise FileNotFoundError(f"No valid SUN RGB-D image/depth records found under {self.dataset_root}")

    def __len__(self):
        return len(self.records)

    def sample_ids(self):
        return [f"{record['index']:05d}" for record in self.records]

    def _record(self, sample_id):
        index = int(sample_id)
        for record in self.records:
            if record["index"] == index:
                return record
        raise KeyError(f"Unknown SUN RGB-D sample ID: {sample_id}")

    def load(self, sample_id: str) -> RGBDSample:
        record = self._record(sample_id)
        item = record["item"]
        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.uint8)
        raw_depth = np.asarray(Image.open(record["depth"]), dtype=np.uint16)
        # Official SUN RGB-D MATLAB toolbox decoding:
        # bitor(bitshift(depthVis,-3), bitshift(depthVis,13))/1000.
        depth_code = np.bitwise_or(np.right_shift(raw_depth, 3),
                                   np.left_shift(raw_depth.astype(np.uint32), 13).astype(np.uint16))
        decoded_depth = depth_code.astype(np.float32) / 1000.0
        valid_depth = np.isfinite(decoded_depth) & (decoded_depth > 0) & (decoded_depth <= self.max_depth_m)
        depth = np.minimum(decoded_depth, self.max_depth_m)
        with h5py.File(self.segmentation_path, "r") as handle:
            reference = handle["SUNRGBD2Dseg/seglabel"][record["index"], 0]
            label = np.asarray(handle[reference], dtype=np.int32).T
        if label.shape != depth.shape or rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"SUN RGB-D alignment mismatch for {sample_id}: rgb={rgb.shape[:2]}, "
                f"depth={depth.shape}, labels={label.shape}"
            )
        valid_label = np.ones_like(label, dtype=bool)
        for value in self.ignore_labels:
            valid_label &= label != value
        intrinsics = np.asarray(item.K, dtype=np.float32)
        return RGBDSample(
            sample_id, self.name, rgb, depth, label, intrinsics,
            valid_depth_mask=valid_depth, valid_label_mask=valid_label,
            metadata={"split": self.split, "sensor_type": str(item.sensorType),
                      "depth_scale": "sunrgbd_packed_13bit_to_metres", "class_names": self.class_names,
                      "rgb_path": str(record["rgb"]), "depth_path": str(record["depth"]),
                      "max_depth_m": self.max_depth_m},
        )


def build_adapter(config):
    if config["name"].lower() in ("nyu", "nyuv2", "nyu_depth_v2"):
        return NYUv2Adapter(
            config["root"], config.get("split", "test"),
            config.get("intrinsics"), config.get("ignore_labels", [0, 255]),
        )
    if config["name"].lower() in ("sun", "sunrgbd", "sun_rgbd"):
        return SUNRGBDAdapter(config["root"], config.get("split", "all"),
                              config.get("label_set", "37"), config.get("ignore_labels", [0]),
                              config.get("max_depth_m", 8.0))
    raise ValueError(f"Unsupported dataset: {config['name']}")
