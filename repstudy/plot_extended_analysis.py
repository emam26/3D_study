"""Extended evidence plots for choosing RGB-D 3D representations.

This module does not invent new labels or train a model.  It consumes the
existing ten-image NYUv2 and SUN RGB-D pilot outputs, plus the stored virtual
view render maps, and exposes the quality/coverage/efficiency trade-offs that
are easy to miss in a single mIoU number.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .datasets import build_adapter
from .metrics import boundary_metrics, semantic_metrics
from .pipeline import load_config


DATASETS = ("nyuv2", "sunrgbd")
REPRESENTATIONS = (
    "pointcloud", "surfel", "mesh", "voxel", "tsdf", "superpoint",
    "graph", "octree", "descriptor",
)
VIEWS = ("original", "left_oblique", "right_oblique", "elevated")
DISPLAY = {
    "pointcloud": "Point cloud", "surfel": "Surfel", "mesh": "Mesh",
    "voxel": "Voxel", "tsdf": "Surface TSDF", "superpoint": "Superpoint",
    "graph": "Graph", "octree": "Octree", "descriptor": "Descriptor",
}
COLORS = {"nyuv2": "#2878b5", "sunrgbd": "#d95f02"}


def _base(output_root: str | Path, dataset: str) -> Path:
    return Path(output_root) / dataset / f"{dataset}_representation_smoke"


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sample_records(output_root: str | Path, dataset: str) -> list[dict]:
    """Load one row per image and representation from saved aggregate JSON."""
    rows = []
    root = _base(output_root, dataset)
    for sample_dir in sorted(root.iterdir() if root.exists() else []):
        path = sample_dir / "aggregate_metrics.json"
        if not path.exists():
            continue
        sample = _read_json(path)
        for name, details in sample.get("representations", {}).items():
            metrics = details.get("metrics", {})
            valid = metrics.get("valid_depth", {})
            full = metrics.get("full_label", {})
            boundary = details.get("boundary", {}).get("2", {})
            rows.append({
                "dataset": dataset,
                "sample": str(sample.get("sample_id", sample_dir.name)),
                "representation": name,
                "element_count": float(details.get("element_count", np.nan)),
                "construction_time_s": float(details.get("construction_time_s", np.nan)),
                "element_purity": float(details.get("element_purity", np.nan)),
                "boundary_crossing_rate": float(details.get("boundary_crossing_rate", np.nan)),
                "valid_depth_miou": float(valid.get("miou", np.nan)),
                "valid_depth_coverage": float(valid.get("coverage", np.nan)),
                "full_label_miou": float(full.get("miou", np.nan)),
                "boundary_f1": float(boundary.get("f1", np.nan)),
            })
    return rows


def _mean_rows(rows: list[dict], dataset: str) -> dict[str, dict]:
    output = {}
    for name in REPRESENTATIONS:
        selected = [row for row in rows if row["dataset"] == dataset and row["representation"] == name]
        if not selected:
            continue
        values = {}
        for key in ("element_count", "construction_time_s", "element_purity",
                    "boundary_crossing_rate", "valid_depth_miou",
                    "valid_depth_coverage", "full_label_miou", "boundary_f1"):
            array = np.asarray([row[key] for row in selected], dtype=np.float64)
            values[key] = float(np.nanmean(array))
            values[f"{key}_std"] = float(np.nanstd(array))
        values["sample_count"] = len(selected)
        values["coverage_optimism_gap"] = values["valid_depth_miou"] - values["full_label_miou"]
        values["miou_per_second"] = values["full_label_miou"] / max(values["construction_time_s"], 1e-8)
        values["miou_per_log10_elements"] = values["full_label_miou"] / max(np.log10(values["element_count"]), 1.0)
        output[name] = values
    return output


def _annotate_points(axis, means, x_key, y_key, scale_x=1.0, scale_y=100.0, min_y=5.0):
    offsets = {
        "mesh": (4, 13), "voxel": (4, 8), "tsdf": (4, -13),
        "superpoint": (4, 4), "octree": (4, -12),
    }
    groups = {}
    for name in REPRESENTATIONS:
        if name not in means:
            continue
        x = means[name][x_key] * scale_x
        y = means[name][y_key] * scale_y
        if y < min_y:
            continue
        groups.setdefault((round(float(x), 2), round(float(y), 2)), []).append((name, x, y))
    for items in groups.values():
        name, x, y = items[0]
        label = " / ".join(DISPLAY[item[0]] for item in items)
        axis.annotate(label, (x, y), xycoords="data", fontsize=8, color="0.2",
                      xytext=offsets.get(name, (4, 4)), textcoords="offset points")


def _pareto_front(means: dict[str, dict], x_key: str, y_key: str,
                  x_lower_is_better: bool = True) -> list[str]:
    names = [name for name in REPRESENTATIONS if name in means]
    front = []
    for name in names:
        x, y = means[name][x_key], means[name][y_key]
        dominated = False
        for other in names:
            if other == name:
                continue
            ox, oy = means[other][x_key], means[other][y_key]
            no_worse_x = ox <= x if x_lower_is_better else ox >= x
            no_worse_y = oy >= y
            strictly_better = (ox < x if x_lower_is_better else ox > x) or oy > y
            if no_worse_x and no_worse_y and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(name)
    return front


def make_tradeoff_overview(rows: list[dict], output: str | Path) -> Path:
    """Show quality versus coverage, runtime, memory proxy, and optimism gap."""
    means = {dataset: _mean_rows(rows, dataset) for dataset in DATASETS}
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), squeeze=False)

    for dataset in DATASETS:
        color = COLORS[dataset]
        summary = means[dataset]
        x = np.asarray([summary[name]["valid_depth_coverage"] * 100 for name in REPRESENTATIONS if name in summary])
        y = np.asarray([summary[name]["full_label_miou"] * 100 for name in REPRESENTATIONS if name in summary])
        size = np.asarray([35 + 18 * np.log10(max(summary[name]["construction_time_s"], 1e-3))
                           for name in REPRESENTATIONS if name in summary])
        axes[0, 0].scatter(x, y, s=np.maximum(size, 20), color=color, alpha=0.82,
                           label=dataset.upper() if dataset == "nyuv2" else "SUN RGB-D")
        _annotate_points(axes[0, 0], summary, "valid_depth_coverage", "full_label_miou", scale_x=100.0)

        x = np.asarray([summary[name]["construction_time_s"] for name in REPRESENTATIONS if name in summary])
        axes[0, 1].scatter(x, y, s=42, color=color, alpha=0.82)
        _annotate_points(axes[0, 1], summary, "construction_time_s", "full_label_miou")

        x = np.asarray([summary[name]["element_count"] for name in REPRESENTATIONS if name in summary])
        axes[1, 0].scatter(x, y, s=42, color=color, alpha=0.82)
        _annotate_points(axes[1, 0], summary, "element_count", "full_label_miou")

        gaps = [summary[name]["coverage_optimism_gap"] * 100 for name in REPRESENTATIONS if name in summary]
        positions = np.arange(len(gaps)) + (-0.19 if dataset == "nyuv2" else 0.19)
        axes[1, 1].bar(positions, gaps, width=0.36, color=color, alpha=0.82,
                        label=dataset.upper() if dataset == "nyuv2" else "SUN RGB-D")

    axes[0, 0].set_xlabel("valid-depth coverage (%)")
    axes[0, 0].set_ylabel("full-label mIoU (%)")
    axes[0, 0].set_title("Coverage-adjusted semantic quality")
    axes[0, 0].set_xlim(-3, 105); axes[0, 0].set_ylim(-2, 102)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("construction time (s, log scale)")
    axes[0, 1].set_ylabel("full-label mIoU (%)")
    axes[0, 1].set_title("Quality versus construction time")
    axes[0, 1].set_ylim(-2, 102)

    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("representation elements (log scale)")
    axes[1, 0].set_ylabel("full-label mIoU (%)")
    axes[1, 0].set_title("Quality versus representation size")
    axes[1, 0].set_ylim(-2, 102)

    x = np.arange(len(REPRESENTATIONS))
    axes[1, 1].set_xticks(x, [DISPLAY[name] for name in REPRESENTATIONS], rotation=38, ha="right")
    axes[1, 1].set_ylabel("valid-depth mIoU − full-label mIoU (percentage points)")
    axes[1, 1].set_title("Coverage optimism gap: why valid-only mIoU can mislead")
    axes[1, 1].axhline(0, color="0.3", linewidth=0.8)
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("RGB-D representation trade-offs on the fixed 10-image pilots", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return output


def _rank(values: dict[str, float], higher_is_better: bool) -> dict[str, int]:
    ordered = sorted(values, key=values.get, reverse=higher_is_better)
    return {name: index + 1 for index, name in enumerate(ordered)}


def make_rank_heatmap(rows: list[dict], output: str | Path) -> tuple[Path, list[dict]]:
    """Rank every representation on six complementary criteria."""
    metric_spec = [
        ("full_label_miou", "full-label mIoU", True),
        ("valid_depth_coverage", "coverage", True),
        ("element_purity", "purity", True),
        ("boundary_f1", "boundary F1", True),
        ("construction_time_s", "runtime", False),
        ("element_count", "size", False),
    ]
    means = {dataset: _mean_rows(rows, dataset) for dataset in DATASETS}
    ranks = []
    figure, axes = plt.subplots(1, 2, figsize=(17, 8), squeeze=False)
    for axis, dataset in zip(axes.flat, DATASETS):
        matrix = []
        rank_map = {}
        for key, label, high in metric_spec:
            values = {name: means[dataset][name][key] for name in REPRESENTATIONS if name in means[dataset]}
            current = _rank(values, high)
            rank_map[label] = current
            matrix.append([current.get(name, np.nan) for name in REPRESENTATIONS])
        matrix = np.asarray(matrix, dtype=float).T
        image = axis.imshow(matrix, cmap="RdYlGn_r", vmin=1, vmax=len(REPRESENTATIONS), aspect="auto")
        axis.set_xticks(np.arange(len(metric_spec)), [item[1] for item in metric_spec], rotation=35, ha="right")
        axis.set_yticks(np.arange(len(REPRESENTATIONS)), [DISPLAY[name] for name in REPRESENTATIONS])
        axis.set_title(dataset.upper() if dataset == "nyuv2" else "SUN RGB-D")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=9)
        for name in REPRESENTATIONS:
            if name not in means[dataset]:
                continue
            row = {"dataset": dataset, "representation": name}
            row.update({f"rank_{key}": rank_map[label][name] for key, label, _ in metric_spec})
            row["mean_rank"] = float(np.mean(list(row.values())[2:]))
            ranks.append(row)
    figure.subplots_adjust(left=0.10, right=0.88, bottom=0.24, top=0.86, wspace=0.38)
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03,
                    label="rank (1 = best)")
    figure.suptitle("No single representation wins every objective", fontsize=16)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return output, ranks


def make_sample_variability(rows: list[dict], output: str | Path) -> Path:
    """Show whether a representation is stable across the ten images."""
    figure, axes = plt.subplots(2, 2, figsize=(18, 11), squeeze=False)
    for column, dataset in enumerate(DATASETS):
        selected = [row for row in rows if row["dataset"] == dataset]
        for row_index, (metric, ylabel, scale) in enumerate((
            ("full_label_miou", "full-label mIoU (%)", 100),
            ("valid_depth_coverage", "valid-depth coverage (%)", 100),
        )):
            data = [[row[metric] * scale for row in selected if row["representation"] == name]
                    for name in REPRESENTATIONS]
            axis = axes[row_index, column]
            axis.boxplot(data, tick_labels=[DISPLAY[name] for name in REPRESENTATIONS],
                         patch_artist=True, showmeans=True,
                         boxprops={"facecolor": COLORS[dataset], "alpha": 0.20},
                         medianprops={"color": COLORS[dataset], "linewidth": 1.4},
                         meanprops={"marker": "D", "markerfacecolor": COLORS[dataset], "markeredgecolor": "none", "markersize": 4})
            axis.set_title(f"{dataset.upper() if dataset == 'nyuv2' else 'SUN RGB-D'} — {ylabel}")
            axis.set_xticklabels([DISPLAY[name] for name in REPRESENTATIONS], rotation=38, ha="right")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.2)
            if metric == "full_label_miou":
                axis.set_ylim(-2, 102)
            else:
                axis.set_ylim(-2, 102)
    figure.suptitle("Image-to-image variability in the ten-sample pilots", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return output


def _load_adapters(config_root: str | Path) -> dict:
    adapters = {}
    for dataset in DATASETS:
        config = load_config(Path(config_root) / f"{dataset}_smoke.yaml")
        adapters[dataset] = build_adapter(config["dataset"])
    return adapters


def compute_multiview_records(output_root: str | Path, config_root: str | Path) -> list[dict]:
    """Evaluate the already-saved virtual-view maps against the real labels."""
    adapters = _load_adapters(config_root)
    records = []
    for dataset in DATASETS:
        root = _base(output_root, dataset)
        for sample_dir in sorted(root.iterdir() if root.exists() else []):
            aggregate_path = sample_dir / "aggregate_metrics.json"
            if not aggregate_path.exists():
                continue
            sample_id = sample_dir.name
            try:
                sample = adapters[dataset].load(sample_id)
            except Exception as exc:
                print(f"Skipping {dataset}/{sample_id} multiview metrics: {exc}")
                continue
            for name in REPRESENTATIONS:
                view_dir = sample_dir / name / "multiview"
                for view in VIEWS:
                    path = view_dir / f"{view}.npz"
                    if not path.exists():
                        continue
                    with np.load(path) as maps:
                        prediction = maps["semantic_map"]
                        coverage = maps["coverage_mask"]
                        collision = maps["collision_map"]
                    semantic = semantic_metrics(prediction, sample.semantic_gt, coverage,
                                                 sample.valid_label_mask, sample.valid_depth_mask)
                    boundary = boundary_metrics(prediction, sample.semantic_gt, sample.valid_label_mask)
                    records.append({
                        "dataset": dataset, "sample": sample_id, "representation": name, "view": view,
                        "full_label_miou": semantic["full_label"]["miou"],
                        "valid_depth_miou": semantic["valid_depth"]["miou"],
                        "coverage": semantic["valid_depth"]["coverage"],
                        "boundary_f1": boundary["2"]["f1"],
                        "collision_rate": float(np.mean(collision > 1)),
                    })
    return records


def _heatmap(axis, matrix, row_labels, column_labels, title, fmt="{:.1f}"):
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                axis.text(j, i, fmt.format(value), ha="center", va="center", color="white", fontsize=8)
    return image


def make_multiview_robustness(records: list[dict], output: str | Path) -> Path:
    """Compare original and oblique/elevated projections for every style."""
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), squeeze=False)
    image = None
    for column, dataset in enumerate(DATASETS):
        for row_index, metric in enumerate(("full_label_miou", "coverage")):
            matrix = np.full((len(REPRESENTATIONS), len(VIEWS)), np.nan, dtype=float)
            for i, name in enumerate(REPRESENTATIONS):
                for j, view in enumerate(VIEWS):
                    values = [row[metric] for row in records
                              if row["dataset"] == dataset and row["representation"] == name and row["view"] == view]
                    if values:
                        matrix[i, j] = np.mean(values) * 100
            image = _heatmap(axes[row_index, column], matrix, [DISPLAY[name] for name in REPRESENTATIONS],
                             ["Original", "Left", "Right", "Elevated"],
                             f"{dataset.upper() if dataset == 'nyuv2' else 'SUN RGB-D'} — {metric.replace('_', ' ')}")
            axes[row_index, column].set_ylabel("representation")
    if image is not None:
        figure.subplots_adjust(left=0.10, right=0.88, bottom=0.18, top=0.86, wspace=0.38, hspace=0.36)
        figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03, label="percent")
    figure.suptitle("Viewpoint robustness of RGB-D representations", fontsize=16)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return output


def make_case_study_grid(output_root: str | Path, config_root: str | Path,
                         rows: list[dict], output: str | Path) -> Path:
    """Render coverage/error maps for a median-quality scene in each dataset."""
    adapters = _load_adapters(config_root)
    chosen = {}
    for dataset in DATASETS:
        candidates = defaultdict(list)
        for row in rows:
            if row["dataset"] == dataset:
                candidates[row["sample"]].append(row["full_label_miou"])
        if not candidates:
            continue
        med = float(np.median([np.mean(values) for values in candidates.values()]))
        chosen[dataset] = min(candidates, key=lambda sample: abs(np.mean(candidates[sample]) - med))

    cmap = ListedColormap(["#2ca25f", "#de2d26", "#fdae6b", "#bdbdbd"])
    figure, axes = plt.subplots(4, len(REPRESENTATIONS), figsize=(22, 10), squeeze=False)
    for column, dataset in enumerate(DATASETS):
        if dataset not in chosen:
            continue
        sample_id = chosen[dataset]
        sample = adapters[dataset].load(sample_id)
        base = _base(output_root, dataset) / sample_id
        for i, name in enumerate(REPRESENTATIONS):
            path = base / name / "render_maps.npz"
            if not path.exists():
                continue
            with np.load(path) as maps:
                coverage = maps["coverage_mask"]
                prediction = maps["semantic_map"]
            error = np.full(sample.semantic_gt.shape, 3, dtype=np.int32)
            error[coverage & (prediction == sample.semantic_gt)] = 0
            error[coverage & (prediction != sample.semantic_gt)] = 1
            error[sample.valid_label_mask & ~coverage] = 2
            # Dataset rows are arranged in pairs: NYU coverage/error, SUN coverage/error.
            offset = 0 if dataset == "nyuv2" else 2
            axes[offset, i].imshow(coverage, cmap="gray", vmin=0, vmax=1)
            axes[offset + 1, i].imshow(error, cmap=cmap, vmin=0, vmax=3)
            axes[offset, i].axis("off"); axes[offset + 1, i].axis("off")
            axes[offset, i].set_title(DISPLAY[name], fontsize=9)
        axes[offset, 0].set_ylabel(f"{dataset.upper()}\n{sample_id}\ncoverage", fontsize=9)
        axes[offset + 1, 0].set_ylabel("error map\n0 correct · 1 wrong\n2 missing", fontsize=8)
    figure.suptitle("Case-study maps: coverage and oracle reconstruction error", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return output


def write_tables(rows: list[dict], multiview: list[dict], ranks: list[dict], output_root: str | Path):
    analysis_root = Path(output_root) / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    means = {dataset: _mean_rows(rows, dataset) for dataset in DATASETS}
    fields = ["dataset", "representation", "sample_count", "full_label_miou_mean", "full_label_miou_std",
              "valid_depth_miou_mean", "valid_depth_coverage_mean", "coverage_optimism_gap",
              "element_purity_mean", "boundary_f1_mean", "construction_time_s_mean", "element_count_mean",
              "miou_per_second", "miou_per_log10_elements"]
    with (analysis_root / "representation_extended_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for dataset in DATASETS:
            for name in REPRESENTATIONS:
                if name not in means[dataset]:
                    continue
                value = means[dataset][name]
                writer.writerow({"dataset": dataset, "representation": name,
                                 "sample_count": value["sample_count"],
                                 "full_label_miou_mean": value["full_label_miou"],
                                 "full_label_miou_std": value["full_label_miou_std"],
                                 "valid_depth_miou_mean": value["valid_depth_miou"],
                                 "valid_depth_coverage_mean": value["valid_depth_coverage"],
                                 "coverage_optimism_gap": value["coverage_optimism_gap"],
                                 "element_purity_mean": value["element_purity"],
                                 "boundary_f1_mean": value["boundary_f1"],
                                 "construction_time_s_mean": value["construction_time_s"],
                                 "element_count_mean": value["element_count"],
                                 "miou_per_second": value["miou_per_second"],
                                 "miou_per_log10_elements": value["miou_per_log10_elements"]})

    if multiview:
        fields = ["dataset", "representation", "view", "full_label_miou_mean", "valid_depth_miou_mean",
                  "coverage_mean", "boundary_f1_mean", "collision_rate_mean"]
        with (analysis_root / "representation_multiview_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for dataset in DATASETS:
                for name in REPRESENTATIONS:
                    for view in VIEWS:
                        selected = [row for row in multiview if row["dataset"] == dataset and row["representation"] == name and row["view"] == view]
                        if not selected:
                            continue
                        writer.writerow({"dataset": dataset, "representation": name, "view": view,
                                         "full_label_miou_mean": np.mean([r["full_label_miou"] for r in selected]),
                                         "valid_depth_miou_mean": np.mean([r["valid_depth_miou"] for r in selected]),
                                         "coverage_mean": np.mean([r["coverage"] for r in selected]),
                                         "boundary_f1_mean": np.mean([r["boundary_f1"] for r in selected]),
                                         "collision_rate_mean": np.mean([r["collision_rate"] for r in selected])})
    with (analysis_root / "representation_ranks.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in ranks for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(ranks)

    lines = [
        "# Extended representation findings",
        "",
        "These findings summarize the fixed ten-image pilots. They are an oracle study of visible RGB-D geometry, not a trained segmentation benchmark.",
        "",
    ]
    for dataset in DATASETS:
        summary = means[dataset]
        if not summary:
            continue
        quality = max(summary, key=lambda name: summary[name]["full_label_miou"])
        boundary = max(summary, key=lambda name: summary[name]["boundary_f1"])
        compact = min((name for name in summary if summary[name]["valid_depth_coverage"] >= 0.95),
                      key=lambda name: summary[name]["element_count"])
        fast = min((name for name in summary if summary[name]["valid_depth_coverage"] >= 0.95),
                   key=lambda name: summary[name]["construction_time_s"])
        lines.extend([
            f"## {dataset.upper() if dataset == 'nyuv2' else 'SUN RGB-D'}",
            "",
            f"- Highest full-label mIoU: **{DISPLAY[quality]}** ({summary[quality]['full_label_miou'] * 100:.1f}%).",
            f"- Highest boundary F1: **{DISPLAY[boundary]}** ({summary[boundary]['boundary_f1'] * 100:.1f}%).",
            f"- Smallest full-coverage representation: **{DISPLAY[compact]}** ({summary[compact]['element_count']:.0f} elements).",
            f"- Fastest full-coverage representation: **{DISPLAY[fast]}** ({summary[fast]['construction_time_s']:.2f} seconds/image).",
            "",
        ])
    lines.extend([
        "## How to interpret the low-coverage styles",
        "",
        "Point clouds, surfels, graphs, and descriptors use fixed point budgets in this pilot. Their valid-depth mIoU can look perfect because it is measured only on the few pixels they cover. The coverage-optimism plot and full-label mIoU expose that limitation.",
        "",
        "The most defensible first choice for dense RGB-D geometry in this pilot is the voxel representation: it reaches full coverage with the highest full-label mIoU on both datasets. Octrees are the compact alternative; meshes preserve boundaries best but are much more expensive. These are pilot findings, not universal rankings.",
        "",
    ])
    (analysis_root / "representation_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Extended RGB-D representation quality and efficiency analysis")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--docs-figures", default="docs/figures")
    args = parser.parse_args()

    rows = []
    for dataset in DATASETS:
        rows.extend(_sample_records(args.output_root, dataset))
    if not rows:
        raise FileNotFoundError("No aggregate_metrics.json files found; run the two ten-image pilots first.")

    output_root = Path(args.output_root)
    docs = Path(args.docs_figures); docs.mkdir(parents=True, exist_ok=True)
    print(make_tradeoff_overview(rows, output_root / "analysis" / "representation_tradeoff_overview.png"))
    rank_path, ranks = make_rank_heatmap(rows, output_root / "analysis" / "representation_rank_heatmap.png")
    print(rank_path)
    print(make_sample_variability(rows, output_root / "analysis" / "representation_sample_variability.png"))
    multiview = compute_multiview_records(args.output_root, args.config_root)
    if multiview:
        print(make_multiview_robustness(multiview, output_root / "analysis" / "representation_multiview_robustness.png"))
    print(make_case_study_grid(args.output_root, args.config_root, rows,
                               output_root / "analysis" / "representation_case_study.png"))
    write_tables(rows, multiview, ranks, args.output_root)

    # Keep README-renderable copies under version control while leaving the
    # large per-sample output tree ignored.
    for name in ("representation_tradeoff_overview.png", "representation_rank_heatmap.png",
                 "representation_sample_variability.png", "representation_multiview_robustness.png",
                 "representation_case_study.png"):
        source = output_root / "analysis" / name
        if source.exists():
            target = docs / name
            target.write_bytes(source.read_bytes())
            print(target)


if __name__ == "__main__":
    main()
