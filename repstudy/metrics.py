"""Semantic, boundary, component, coverage, and efficiency metrics."""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage


def _valid_mask(gt, valid_label=None):
    return (gt > 0) if valid_label is None else valid_label.astype(bool) & (gt > 0)


def semantic_metrics(prediction, ground_truth, coverage=None, valid_label=None, valid_depth=None, num_classes=None):
    prediction = np.asarray(prediction); ground_truth = np.asarray(ground_truth)
    if num_classes is None:
        prediction_max = int(prediction.max()) if prediction.size else 0
        ground_truth_max = int(ground_truth.max()) if ground_truth.size else 0
        num_classes = max(prediction_max, ground_truth_max, 0)
    base = _valid_mask(ground_truth, valid_label)
    depth = np.ones_like(base, bool) if valid_depth is None else valid_depth.astype(bool)
    covered = np.ones_like(base, bool) if coverage is None else coverage.astype(bool)
    tracks = {"valid_depth": base & depth & covered, "full_label": base}
    output = {}
    for name, mask in tracks.items():
        track_base = base & depth if name == "valid_depth" else base
        confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
        p = prediction.copy(); p[~covered] = 0
        codes = ground_truth[mask] * (num_classes + 1) + p[mask]
        if codes.size: confusion += np.bincount(codes, minlength=(num_classes + 1) ** 2).reshape(confusion.shape)
        tp = np.diag(confusion)[1:]
        union = confusion[1:, :].sum(1) + confusion[:, 1:].sum(0) - tp
        iou = np.divide(tp, union, out=np.full(num_classes, np.nan), where=union > 0)
        total = confusion[1:].sum()
        output[name] = {
            "miou": float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0,
            "pixel_accuracy": float(tp.sum() / max(total, 1)),
            "mean_accuracy": float(np.nanmean(np.divide(tp, confusion[1:].sum(1), out=np.full(num_classes, np.nan), where=confusion[1:].sum(1) > 0))),
            "class_iou": iou.tolist(), "confusion_matrix": confusion,
            "coverage": float((track_base & covered).sum() / max(track_base.sum(), 1)),
            "missing_rate": float((track_base & ~covered).sum() / max(track_base.sum(), 1)),
        }
    return output


def boundary_map(labels):
    labels = np.asarray(labels)
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[1:] |= labels[:-1] != labels[1:]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary & (labels > 0)


def boundary_metrics(prediction, ground_truth, valid_mask=None, tolerances=(1, 2, 4)):
    valid = np.ones_like(ground_truth, bool) if valid_mask is None else valid_mask.astype(bool)
    target = boundary_map(ground_truth) & valid; predicted = boundary_map(prediction) & valid
    output = {}
    for tolerance in tolerances:
        structure = ndimage.generate_binary_structure(2, 1)
        dilated_target = ndimage.binary_dilation(target, structure, iterations=tolerance)
        dilated_prediction = ndimage.binary_dilation(predicted, structure, iterations=tolerance)
        precision = (predicted & dilated_target).sum() / max(predicted.sum(), 1)
        recall = (target & dilated_prediction).sum() / max(target.sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        iou = (predicted & dilated_target).sum() / max((predicted | target).sum(), 1)
        output[str(tolerance)] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou)}
    return output


def component_metrics(prediction, ground_truth, bins=(0, 64, 256, 1024, np.inf)):
    target = ground_truth > 0
    component_map, count = ndimage.label(target)
    output = {f"{bins[i]}-{bins[i + 1]}": {"count": 0, "recall": 0.0, "disappearance_rate": 0.0} for i in range(len(bins) - 1)}
    for component in range(1, count + 1):
        mask = component_map == component; size = mask.sum()
        index = next((i for i in range(len(bins) - 1) if bins[i] <= size < bins[i + 1]), len(bins) - 2)
        key = f"{bins[index]}-{bins[index + 1]}"; output[key]["count"] += 1
        recall = float((prediction[mask] == ground_truth[mask]).sum() / max(size, 1))
        output[key]["recall"] += recall
        output[key]["disappearance_rate"] += float(recall == 0)
    for value in output.values():
        if value["count"]: value["recall"] /= value["count"]; value["disappearance_rate"] /= value["count"]
    return output


class Timer:
    def __init__(self): self.start_time = None
    def __enter__(self): self.start_time = time.perf_counter(); return self
    def __exit__(self, *_): self.elapsed = time.perf_counter() - self.start_time
