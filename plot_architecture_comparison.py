"""Create the three scientific comparison plots for completed versions.

The script discovers version metrics under ``outputs/segmentation``. It never
invents missing results: incomplete versions remain absent or are marked NaN.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VERSION_RE = re.compile(r"^(v\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/segmentation")
    parser.add_argument("--save-dir", default="outputs/segmentation/architecture_plots")
    return parser.parse_args()


def discover_records(output_root):
    records = []
    for metrics_path in sorted(Path(output_root).rglob("metrics.json")):
        parent = metrics_path.parent
        match = VERSION_RE.match(parent.name)
        if not match:
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        metrics = data.get("metrics", {})
        dataset = str(data.get("dataset", "unknown"))
        records.append({
            "version": match.group(1).lower(),
            "experiment": parent.name,
            "dataset": dataset,
            "miou": float(metrics.get("miou", np.nan)),
            "pixel_accuracy": float(metrics.get("pixel_accuracy", np.nan)),
            "parameters": float(data.get("parameters", np.nan)),
            "runtime_seconds": float(data.get("total_runtime_seconds", np.nan)),
            "metrics_path": str(metrics_path),
            "visualization": data.get("visualization"),
            "visualization_path": parent / f"{dataset}_qualitative_grid.png",
        })
    return records


def write_summary(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["version", "experiment", "dataset", "miou", "pixel_accuracy",
              "parameters", "runtime_seconds", "metrics_path"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def plot_metric_heatmap(records, output_path):
    versions = sorted({record["version"] for record in records}, key=lambda x: int(x[1:]))
    datasets = sorted({record["dataset"] for record in records})
    columns = [f"{dataset} mIoU" for dataset in datasets] + ["Mean mIoU", "Mean pixel accuracy"]
    matrix = np.full((len(versions), len(columns)), np.nan, dtype=np.float32)
    for row, version in enumerate(versions):
        version_records = [record for record in records if record["version"] == version]
        for column, dataset in enumerate(datasets):
            values = [record["miou"] for record in version_records
                      if record["dataset"] == dataset and np.isfinite(record["miou"])]
            if values:
                matrix[row, column] = np.mean(values)
        miou_values = [record["miou"] for record in version_records if np.isfinite(record["miou"])]
        pa_values = [record["pixel_accuracy"] for record in version_records
                     if np.isfinite(record["pixel_accuracy"])]
        if miou_values:
            matrix[row, len(datasets)] = np.mean(miou_values)
        if pa_values:
            matrix[row, len(datasets) + 1] = np.mean(pa_values)

    figure, axis = plt.subplots(figsize=(max(8, len(columns) * 1.5), max(3, len(versions) * 0.55)))
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    axis.set_yticks(range(len(versions)), [version.upper() for version in versions])
    axis.set_title("Architecture screening: metric heatmap")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="Score")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_pareto(records, output_path):
    grouped = defaultdict(list)
    for record in records:
        if np.isfinite(record["parameters"]) and np.isfinite(record["miou"]):
            grouped[record["version"]].append(record)
    points = []
    for version, version_records in grouped.items():
        points.append((version, np.mean([r["parameters"] for r in version_records]) / 1e6,
                       np.mean([r["miou"] for r in version_records])))

    figure, axis = plt.subplots(figsize=(8, 5.5))
    if points:
        points.sort(key=lambda item: item[1])
        x = np.asarray([item[1] for item in points])
        y = np.asarray([item[2] for item in points])
        axis.scatter(x, y, s=75, color="#2878b5")
        best_so_far = -np.inf
        pareto_x, pareto_y = [], []
        for version, params, miou in points:
            axis.annotate(version.upper(), (params, miou), xytext=(5, 5), textcoords="offset points")
            if miou >= best_so_far:
                pareto_x.append(params); pareto_y.append(miou); best_so_far = miou
        axis.plot(pareto_x, pareto_y, "--", color="#d95f02", label="Pareto frontier")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Parameter metadata unavailable", ha="center", va="center")
    axis.set_xlabel("Parameters (millions)")
    axis.set_ylabel("Mean validation mIoU")
    axis.set_title("Accuracy-efficiency Pareto comparison")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_qualitative_comparisons(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    by_dataset = defaultdict(list)
    for record in records:
        path = record["visualization_path"]
        if record.get("visualization"):
            candidate = Path(record["visualization"])
            if candidate.exists():
                path = candidate
        if path.exists():
            by_dataset[record["dataset"]].append((record["version"], path))
    for dataset, entries in sorted(by_dataset.items()):
        entries.sort(key=lambda item: int(item[0][1:]))
        figure, axes = plt.subplots(len(entries), 1, figsize=(12, 4 * len(entries)), squeeze=False)
        for axis, (version, path) in zip(axes[:, 0], entries):
            axis.imshow(plt.imread(path))
            axis.set_title(f"{version.upper()} - {dataset}")
            axis.axis("off")
        figure.tight_layout()
        path = output_dir / f"architecture_qualitative_{dataset}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)
    return saved


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    save_dir = Path(args.save_dir)
    records = discover_records(output_root)
    if not records:
        raise FileNotFoundError(f"No version metrics found under {output_root}")
    save_dir.mkdir(parents=True, exist_ok=True)
    write_summary(records, save_dir / "architecture_screening_summary.csv")
    plot_metric_heatmap(records, save_dir / "architecture_metric_heatmap.png")
    plot_pareto(records, save_dir / "architecture_pareto_miou_params.png")
    qualitative = plot_qualitative_comparisons(records, save_dir)
    print(f"records={len(records)}")
    print(f"heatmap={save_dir / 'architecture_metric_heatmap.png'}")
    print(f"pareto={save_dir / 'architecture_pareto_miou_params.png'}")
    for path in qualitative:
        print(f"qualitative={path}")


if __name__ == "__main__":
    main()
