"""V1 RGB-only semantic-segmentation baseline for the 3D study.

This is deliberately a control model: SegFormer-B0 receives RGB only. The
depth map is loaded by the common RGB-D adapter for sample alignment, but it is
not passed to the model. Later versions can add one 3D branch at a time.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from repstudy.datasets import build_adapter


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_sample(sample, image_size):
    height, width = image_size
    rgb = np.asarray(
        Image.fromarray(sample.rgb).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    ).astype(np.float32) / 255.0
    label = np.asarray(
        Image.fromarray(sample.semantic_gt.astype(np.int32), mode="I").resize(
            (width, height), Image.Resampling.NEAREST
        ),
        dtype=np.int32,
    )
    return rgb, label


class RGBSegmentationDataset(Dataset):
    def __init__(self, adapter, sample_ids, image_size, num_classes, ignore_label):
        self.adapter = adapter
        self.sample_ids = list(sample_ids)
        self.image_size = tuple(image_size)
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample = self.adapter.load(sample_id)
        rgb, label = resize_sample(sample, self.image_size)
        rgb = (rgb - MEAN) / STD

        # Hugging Face segmentation models use 255 as the ignore target.
        target = np.full(label.shape, 255, dtype=np.int64)
        foreground = (label != self.ignore_label) & (label > 0)
        class_ids = label[foreground] - 1
        valid_class = (class_ids >= 0) & (class_ids < self.num_classes)
        target_values = np.full(class_ids.shape, 255, dtype=np.int64)
        target_values[valid_class] = class_ids[valid_class]
        target[foreground] = target_values

        return {
            "pixel_values": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
            "labels": torch.from_numpy(target),
            "name": sample_id,
        }


def split_ids(adapter, dataset_name, seed, val_fraction):
    ids = list(adapter.sample_ids())
    if dataset_name == "nyuv2":
        return ids, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    val_count = max(1, int(round(len(ids) * val_fraction)))
    val_indices = set(order[:val_count].tolist())
    train_ids = [sample_id for index, sample_id in enumerate(ids) if index not in val_indices]
    val_ids = [sample_id for index, sample_id in enumerate(ids) if index in val_indices]
    return train_ids, val_ids


def make_model(config, device, force_random=False):
    model_cfg = config["model"]
    num_classes = int(config["dataset"]["num_classes"])
    pretrained = bool(model_cfg.get("pretrained", True)) and not force_random
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_cfg["name"], num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        model_config = SegformerConfig(num_labels=num_classes)
        model = SegformerForSemanticSegmentation(model_config)
    model.config.semantic_loss_ignore_index = 255
    return model.to(device)


def update_confusion(confusion, logits, labels, num_classes):
    prediction = F.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    ).argmax(1)
    valid = labels != 255
    truth = labels[valid].detach().cpu().numpy()
    predicted = prediction[valid].detach().cpu().numpy()
    if truth.size:
        values = truth * num_classes + predicted
        confusion += np.bincount(
            values, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)


def metric_summary(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    class_present = union > 0
    return {
        "miou": float(iou[class_present].mean()) if class_present.any() else 0.0,
        "pixel_accuracy": float(intersection.sum() / max(confusion.sum(), 1)),
        "class_iou": iou.tolist(),
        "classes_present": int(class_present.sum()),
    }


def _colorize_labels(labels, num_classes, colormap):
    """Render contiguous class IDs as an RGB image; 255 is shown as black."""
    labels = np.asarray(labels)
    image = np.zeros((*labels.shape, 3), dtype=np.float32)
    valid = (labels >= 0) & (labels < num_classes)
    if valid.any():
        image[valid] = colormap(labels[valid])[:, :3]
    return image


@torch.no_grad()
def save_qualitative_grid(model, loader, device, num_classes, output_path, max_samples=8):
    """Save RGB/GT/prediction/error rows for a small deterministic validation subset."""
    if loader is None or max_samples <= 0:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    colormap = plt.get_cmap("turbo", num_classes)
    rows = []
    collected = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        output = model(pixel_values=pixel_values)
        predictions = F.interpolate(
            output.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        ).argmax(1)
        rgb_batch = pixel_values.detach().cpu().numpy().transpose(0, 2, 3, 1)
        rgb_batch = np.clip(rgb_batch * STD + MEAN, 0.0, 1.0)
        labels_np = labels.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        for rgb, truth, prediction in zip(rgb_batch, labels_np, predictions_np):
            valid = truth != 255
            error = np.zeros((*truth.shape, 3), dtype=np.float32)
            error[~valid] = (0.25, 0.25, 0.25)
            error[valid & (prediction == truth)] = (0.15, 0.75, 0.25)
            error[valid & (prediction != truth)] = (0.90, 0.15, 0.10)
            rows.append((rgb, truth, prediction, error))
            collected += 1
            if collected >= max_samples:
                break
        if collected >= max_samples:
            break
    model.train()
    if not rows:
        return None

    columns = ("RGB", "Ground truth", "Prediction", "Error map")
    figure, axes = plt.subplots(
        len(rows), len(columns), figsize=(12, max(3.0 * len(rows), 3.0)), squeeze=False
    )
    for row_index, (rgb, truth, prediction, error) in enumerate(rows):
        images = (
            rgb,
            _colorize_labels(truth, num_classes, colormap),
            _colorize_labels(prediction, num_classes, colormap),
            error,
        )
        for column_index, (axis, image) in enumerate(zip(axes[row_index], images)):
            axis.imshow(image)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(columns[column_index])
    figure.tight_layout(pad=0.8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        output = model(pixel_values=pixel_values)
        update_confusion(confusion, output.logits, labels, num_classes)
    model.train()
    return metric_summary(confusion)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override DataLoader workers (use 0 for notebook/Windows CPU smoke tests).",
    )
    parser.add_argument("--num-visual-samples", type=int, default=8)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    evaluation_cfg = config.get("evaluation", {})
    seed = int(training_cfg.get("seed", 42))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataset_name = dataset_cfg["name"].lower()
    adapter_cfg = dict(dataset_cfg)
    adapter_cfg["split"] = dataset_cfg.get("train_split", dataset_cfg.get("split", "all"))
    train_adapter = build_adapter(adapter_cfg)

    if dataset_name == "nyuv2":
        val_cfg = dict(dataset_cfg)
        val_cfg["split"] = dataset_cfg.get("val_split", "test")
        val_adapter = build_adapter(val_cfg)
        train_ids = train_adapter.sample_ids()
        val_ids = val_adapter.sample_ids()
    else:
        train_ids, val_ids = split_ids(
            train_adapter, dataset_name, seed, float(evaluation_cfg.get("val_fraction", 0.2))
        )
        val_adapter = train_adapter

    if args.max_train_samples is not None:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples is not None and val_ids is not None:
        val_ids = val_ids[: args.max_val_samples]

    image_size = tuple(args.image_size or training_cfg.get("image_size", [480, 480]))
    num_classes = int(dataset_cfg["num_classes"])
    ignore_label = int(dataset_cfg.get("ignore_label", 0))
    train_set = RGBSegmentationDataset(
        train_adapter, train_ids, image_size, num_classes, ignore_label
    )
    val_set = RGBSegmentationDataset(
        val_adapter, val_ids, image_size, num_classes, ignore_label
    ) if val_ids else None
    workers = int(
        args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0)
    )
    common_loader = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_set, batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=True, drop_last=False, **common_loader
    )
    val_loader = DataLoader(
        val_set, batch_size=int(evaluation_cfg.get("batch_size", 4)),
        shuffle=False, **common_loader
    ) if val_set else None

    model = make_model(config, device, force_random=args.no_pretrained)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_cfg.get("learning_rate", 6e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01))
    )
    use_amp = bool(training_cfg.get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"V1 RGB SegFormer-B0 | dataset={dataset_name} | device={device}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"train={len(train_set)} val={len(val_set) if val_set else 0} image_size={image_size}")
    print(f"parameters={parameter_count:,}")

    started = time.perf_counter()
    model.train()
    train_loss = []
    history = []
    best_epoch = 0
    best_metrics = {}
    best_miou = -float("inf")
    best_state = None
    evaluation_seconds = 0.0
    evaluate_each_epoch = bool(training_cfg.get("evaluate_each_epoch", True))
    epochs = int(training_cfg.get("epochs", 1))
    for epoch in range(epochs):
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(pixel_values=pixel_values, labels=labels)
                loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite V1 loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach().cpu())
            train_loss.append(loss_value)
            epoch_losses.append(loss_value)
        epoch_metrics = {}
        if val_loader and evaluate_each_epoch:
            evaluation_started = time.perf_counter()
            epoch_metrics = evaluate(model, val_loader, device, num_classes)
            evaluation_seconds += time.perf_counter() - evaluation_started
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "metrics": epoch_metrics,
        })
        if epoch_metrics:
            print(
                f"epoch={epoch + 1} train_loss={np.mean(epoch_losses):.6f} "
                f"val_mIoU={epoch_metrics['miou']:.6f} "
                f"pixel_accuracy={epoch_metrics['pixel_accuracy']:.6f}"
            )
            if epoch_metrics["miou"] > best_miou:
                best_miou = epoch_metrics["miou"]
                best_epoch = epoch + 1
                best_metrics = epoch_metrics
                best_state = deepcopy(model.state_dict())
        else:
            print(f"epoch={epoch + 1} train_loss={np.mean(epoch_losses):.6f}")

    if not evaluate_each_epoch and val_loader:
        evaluation_started = time.perf_counter()
        best_metrics = evaluate(model, val_loader, device, num_classes)
        evaluation_seconds += time.perf_counter() - evaluation_started
        best_epoch = epochs
        best_miou = best_metrics.get("miou", -float("inf"))
        best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = best_metrics
    if metrics:
        print(f"best_epoch={best_epoch} best_val_mIoU={metrics['miou']:.6f}")

    output_root = Path(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "history": history,
    }
    torch.save(checkpoint, output_root / "v1_rgb_segformer.pth")
    with open(output_root / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    visualization_path = None
    if not args.no_visualization:
        visualization_path = save_qualitative_grid(
            model, val_loader, device, num_classes,
            output_root / f"{dataset_name}_qualitative_grid.png",
            max_samples=args.num_visual_samples,
        )
        if visualization_path:
            print(f"visualization={visualization_path}")
    with open(output_root / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"dataset": dataset_name, "train_samples": len(train_set),
                   "val_samples": len(val_set) if val_set else 0,
                   "mean_train_loss": float(np.mean(train_loss)), "metrics": metrics,
                   "visualization": visualization_path,
                   "parameters": int(parameter_count),
                   "epochs": epochs,
                   "best_epoch": int(best_epoch),
                   "history": history,
                   "evaluation_seconds": round(evaluation_seconds, 3),
                   "total_runtime_seconds": round(time.perf_counter() - started, 3)},
                  handle, indent=2)
    print(f"saved={output_root}")


if __name__ == "__main__":
    main()
