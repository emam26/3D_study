"""Generate V1 qualitative grids from an existing checkpoint without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from repstudy.datasets import build_adapter
from train_v1_rgb_segformer import (
    RGBSegmentationDataset,
    make_model,
    save_qualitative_grid,
    seed_everything,
    split_ids,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    evaluation_cfg = config.get("evaluation", {})
    seed_everything(int(training_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = dataset_cfg["name"].lower()
    adapter_cfg = dict(dataset_cfg)
    adapter_cfg["split"] = dataset_cfg.get("train_split", dataset_cfg.get("split", "all"))
    train_adapter = build_adapter(adapter_cfg)
    if dataset_name == "nyuv2":
        val_cfg = dict(dataset_cfg)
        val_cfg["split"] = dataset_cfg.get("val_split", "test")
        val_adapter = build_adapter(val_cfg)
        val_ids = val_adapter.sample_ids()
    else:
        _, val_ids = split_ids(
            train_adapter, dataset_name, int(training_cfg.get("seed", 42)),
            float(evaluation_cfg.get("val_fraction", 0.2)),
        )
        val_adapter = train_adapter

    image_size = tuple(training_cfg.get("image_size", [480, 480]))
    val_set = RGBSegmentationDataset(
        val_adapter, val_ids, image_size,
        int(dataset_cfg["num_classes"]), int(dataset_cfg.get("ignore_label", 0)),
    )
    loader = DataLoader(
        val_set, batch_size=int(evaluation_cfg.get("batch_size", 4)),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = Path(args.checkpoint or config["paths"]["output_root"]) / "v1_rgb_segformer.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    # The checkpoint contains all model weights, so no Hugging Face download is needed.
    model = make_model(config, device, force_random=True)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    output_path = checkpoint.parent / f"{dataset_name}_qualitative_grid.png"
    saved = save_qualitative_grid(
        model, loader, device, int(dataset_cfg["num_classes"]),
        output_path, max_samples=args.num_samples,
    )
    print(f"device={device} checkpoint={checkpoint}")
    print(f"visualization={saved}")


if __name__ == "__main__":
    main()
