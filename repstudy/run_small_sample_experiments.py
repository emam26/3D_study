"""Small-sample follow-up experiments for the RGB-D representation study.

The runner intentionally uses the same ten NYUv2 and ten SUN RGB-D images as
the main pilot.  It adds controlled rate--distortion, depth-corruption, and
geometry-only hybrid experiments without training a model or using labels to
construct a representation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .datasets import build_adapter
from .geometry import geometry_attributes
from .metrics import boundary_metrics, semantic_metrics
from .oracle import assign_oracle_labels, render_original_view
from .pipeline import load_config
from .plot_extended_analysis import DISPLAY, _mean_rows, _sample_records
from .representations import build_representation
from .types import RGBDSample


DATASETS = ("nyuv2", "sunrgbd")
REPRESENTATIONS = (
    "pointcloud", "surfel", "mesh", "voxel", "tsdf", "superpoint",
    "graph", "octree", "descriptor",
)
HYBRID_REPRESENTATIONS = ("mesh", "voxel", "octree")
ROBUST_REPRESENTATIONS = ("voxel", "octree", "superpoint")
COLORS = {
    "pointcloud": "#4c78a8", "surfel": "#72b7b2", "mesh": "#e45756",
    "voxel": "#f2cf5b", "tsdf": "#b279a2", "superpoint": "#54a24b",
    "graph": "#ff9da6", "octree": "#9d755d", "descriptor": "#bab0ab",
    "hybrid": "#1f9e89",
}


# The baseline point is already available in the ten-image pilot.  Only the
# additional settings below are recomputed, which keeps the small study
# reproducible without duplicating the baseline work.
RATE_OVERRIDES = {
    "pointcloud": [("low", {"budget": 1000}), ("high", {"budget": 10000})],
    "surfel": [("low", {"budget": 1000}), ("high", {"budget": 10000})],
    "mesh": [("coarse", {"max_depth_jump_m": 0.10})],
    "voxel": [("coarse", {"voxel_size_m": 0.08}), ("fine", {"voxel_size_m": 0.02})],
    "tsdf": [("coarse", {"voxel_size_m": 0.08}), ("fine", {"voxel_size_m": 0.04})],
    "superpoint": [("coarse", {"target_regions": 500}), ("fine", {"target_regions": 2000})],
    "graph": [("low", {"budget": 1000}), ("high", {"budget": 5000})],
    "octree": [("coarse", {"max_points_per_leaf": 256}), ("fine", {"max_points_per_leaf": 64})],
    "descriptor": [("low", {"budget": 1000}), ("high", {"budget": 10000})],
}

CORRUPTION_CONDITIONS = (
    ("clean", "clean", 0.0),
    ("dropout10", "dropout", 0.10),
    ("dropout30", "dropout", 0.30),
    ("noise1cm", "gaussian", 0.01),
    ("noise3cm", "gaussian", 0.03),
)


def _base(output_root: str | Path, dataset: str) -> Path:
    return Path(output_root) / dataset / f"{dataset}_representation_smoke"


def _sample_ids(output_root: str | Path, dataset: str) -> list[str]:
    root = _base(output_root, dataset)
    ids = sorted(path.name for path in root.iterdir()
                 if path.is_dir() and (path / "aggregate_metrics.json").exists())
    if len(ids) < 10:
        raise RuntimeError(f"Expected ten pilot samples for {dataset}, found {len(ids)}")
    return ids[:10]


def _representation_config(config: dict, name: str, overrides: dict | None = None) -> dict:
    values = {key: value for key, value in config["representations"][name].items() if key != "enabled"}
    values.update(overrides or {})
    return values


def _evaluate(sample: RGBDSample, name: str, config: dict) -> dict:
    started = time.perf_counter()
    result = build_representation(name, sample, config)
    result = assign_oracle_labels(result, sample)
    render = render_original_view(result, sample)
    semantic = semantic_metrics(render.semantic_map, sample.semantic_gt, render.coverage_mask,
                                sample.valid_label_mask, sample.valid_depth_mask)
    boundary = boundary_metrics(render.semantic_map, sample.semantic_gt, sample.valid_label_mask)
    elapsed = time.perf_counter() - started
    return {
        "element_count": result.element_count,
        "construction_time_s": elapsed,
        "element_purity": float(result.attributes.get("purity", np.zeros(1)).mean()) if result.element_count else 0.0,
        "boundary_crossing_rate": float(result.attributes.get("boundary_crossing", np.zeros(1)).mean()) if result.element_count else 0.0,
        "valid_depth_miou": semantic["valid_depth"]["miou"],
        "valid_depth_coverage": semantic["valid_depth"]["coverage"],
        "full_label_miou": semantic["full_label"]["miou"],
        "boundary_f1": boundary["2"]["f1"],
    }


def _sample_cache(adapters: dict, output_root: str | Path) -> dict[tuple[str, str], RGBDSample]:
    cache = {}
    for dataset in DATASETS:
        for sample_id in _sample_ids(output_root, dataset):
            cache[(dataset, sample_id)] = adapters[dataset].load(sample_id)
    return cache


def _row(dataset: str, sample: str, representation: str, variant: str, metrics: dict, **extra) -> dict:
    value = {"dataset": dataset, "sample": sample, "representation": representation,
             "variant": variant}
    value.update(metrics)
    value.update(extra)
    return value


METRIC_FIELDS = (
    "element_count", "construction_time_s", "element_purity", "boundary_crossing_rate",
    "valid_depth_miou", "valid_depth_coverage", "full_label_miou", "boundary_f1",
)


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_rate_distortion(output_root: str | Path, configs: dict, cache: dict) -> list[dict]:
    """Evaluate baseline plus controlled resolution changes for all nine styles."""
    rows = []
    existing = {dataset: _sample_records(output_root, dataset) for dataset in DATASETS}
    for dataset in DATASETS:
        for source in existing[dataset]:
            if source["representation"] not in REPRESENTATIONS:
                continue
            rows.append(_row(dataset, source["sample"], source["representation"], "baseline",
                             {key: source[key] for key in METRIC_FIELDS}))
        for name in REPRESENTATIONS:
            for variant, overrides in RATE_OVERRIDES[name]:
                print(f"rate-distortion {dataset} {name}/{variant}")
                for sample_id in _sample_ids(output_root, dataset):
                    metrics = _evaluate(cache[(dataset, sample_id)], name,
                                        _representation_config(configs[dataset], name, overrides))
                    rows.append(_row(dataset, sample_id, name, variant, metrics))
    return rows


def _corrupt_sample(sample: RGBDSample, kind: str, level: float, seed: int) -> RGBDSample:
    depth = sample.depth_m.astype(np.float32).copy()
    valid = sample.valid_depth_mask.copy()
    if kind == "dropout":
        rng = np.random.default_rng(seed)
        drop = (rng.random(depth.shape) < level) & valid
        depth[drop] = 0.0
        valid[drop] = False
    elif kind == "gaussian":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, level, depth.shape).astype(np.float32)
        depth[valid] = np.maximum(depth[valid] + noise[valid], 1e-4)
    elif kind != "clean":
        raise ValueError(f"Unknown corruption kind: {kind}")
    return RGBDSample(
        sample.sample_id, sample.dataset_name, sample.rgb, depth, sample.semantic_gt,
        sample.intrinsics, sample.extrinsics, valid, sample.valid_label_mask,
        {**sample.metadata, "corruption": kind, "corruption_level": level},
    )


def run_corruption(output_root: str | Path, configs: dict, cache: dict) -> list[dict]:
    """Measure controlled depth dropout/noise for compact dense representations."""
    rows = []
    existing = {dataset: _sample_records(output_root, dataset) for dataset in DATASETS}
    for dataset in DATASETS:
        # All nine clean baselines remain visible in the table.  Corrupted
        # inputs are evaluated on compact dense styles to keep this pilot small.
        for source in existing[dataset]:
            if source["representation"] not in REPRESENTATIONS:
                continue
            rows.append(_row(dataset, source["sample"], source["representation"], "clean",
                             {key: source[key] for key in METRIC_FIELDS}, condition="clean"))
        for condition, kind, level in CORRUPTION_CONDITIONS:
            if condition == "clean":
                continue
            for name in ROBUST_REPRESENTATIONS:
                print(f"corruption {dataset} {name}/{condition}")
                for sample_id in _sample_ids(output_root, dataset):
                    dataset_seed = 0 if dataset == "nyuv2" else 1000
                    corrupted = _corrupt_sample(cache[(dataset, sample_id)], kind, level,
                                                seed=42 + int(sample_id) + dataset_seed)
                    metrics = _evaluate(corrupted, name, _representation_config(configs[dataset], name))
                    rows.append(_row(dataset, sample_id, name, condition, metrics, condition=condition))
    return rows


def _render_maps(output_root: str | Path, dataset: str, sample_id: str, name: str) -> dict:
    path = _base(output_root, dataset) / sample_id / name / "render_maps.npz"
    with np.load(path) as maps:
        return {key: maps[key] for key in ("semantic_map", "coverage_mask")}


def _hybrid_prediction(sample: RGBDSample, maps: dict[str, dict], quantile: float) -> tuple[np.ndarray, np.ndarray, dict]:
    attributes = geometry_attributes(sample)
    score = 0.65 * attributes["depth_edge"] + 0.35 * attributes["rgb_edge"]
    valid_scores = score[sample.valid_depth_mask]
    threshold = float(np.quantile(valid_scores, quantile)) if valid_scores.size else np.inf
    boundary = score >= threshold

    prediction = np.zeros_like(sample.semantic_gt, dtype=np.int32)
    coverage = np.zeros_like(sample.valid_depth_mask, dtype=bool)
    # Geometry-only policy: use the mesh near structural edges, voxels in
    # interiors, and octrees only as a compact fallback for missing coverage.
    mesh = maps["mesh"]; voxel = maps["voxel"]; octree = maps["octree"]
    choose = boundary & mesh["coverage_mask"]
    prediction[choose] = mesh["semantic_map"][choose]; coverage[choose] = True
    choose = (~boundary) & voxel["coverage_mask"]
    prediction[choose] = voxel["semantic_map"][choose]; coverage[choose] = True
    for fallback in (octree, voxel, mesh):
        choose = (~coverage) & fallback["coverage_mask"]
        prediction[choose] = fallback["semantic_map"][choose]; coverage[choose] = True
    selected_mesh = float(np.mean(boundary & mesh["coverage_mask"]))
    selected_voxel = float(np.mean((~boundary) & voxel["coverage_mask"]))
    fallback_fraction = float(np.mean(coverage & ~(boundary & mesh["coverage_mask"]) & ~(~boundary & voxel["coverage_mask"])))
    return prediction, coverage, {
        "threshold_quantile": quantile, "mesh_selection_fraction": selected_mesh,
        "voxel_selection_fraction": selected_voxel, "fallback_fraction": fallback_fraction,
    }


def run_adaptive_hybrid(output_root: str | Path, configs: dict, cache: dict) -> list[dict]:
    """Tune a geometry-only boundary quantile on five images and test five held out."""
    rows = []
    quantiles = (0.60, 0.75, 0.90)
    for dataset in DATASETS:
        sample_ids = _sample_ids(output_root, dataset)
        tune_ids, test_ids = sample_ids[:5], sample_ids[5:]
        selected_quantile = None
        tune_scores = {}
        for quantile in quantiles:
            values = []
            for sample_id in tune_ids:
                maps = {name: _render_maps(output_root, dataset, sample_id, name)
                        for name in HYBRID_REPRESENTATIONS}
                prediction, coverage, _ = _hybrid_prediction(cache[(dataset, sample_id)], maps, quantile)
                metrics = semantic_metrics(prediction, cache[(dataset, sample_id)].semantic_gt, coverage,
                                            cache[(dataset, sample_id)].valid_label_mask,
                                            cache[(dataset, sample_id)].valid_depth_mask)
                values.append(metrics["full_label"]["miou"])
            tune_scores[quantile] = float(np.mean(values))
        selected_quantile = max(quantiles, key=lambda value: (tune_scores[value], -value))
        for split, ids in (("tune", tune_ids), ("heldout", test_ids)):
            for sample_id in ids:
                sample = cache[(dataset, sample_id)]
                maps = {name: _render_maps(output_root, dataset, sample_id, name)
                        for name in HYBRID_REPRESENTATIONS}
                prediction, coverage, details = _hybrid_prediction(sample, maps, selected_quantile)
                metrics = semantic_metrics(prediction, sample.semantic_gt, coverage,
                                            sample.valid_label_mask, sample.valid_depth_mask)
                boundary = boundary_metrics(prediction, sample.semantic_gt, sample.valid_label_mask)
                rows.append({"dataset": dataset, "sample": sample_id, "split": split,
                             "representation": "hybrid", "variant": "mesh-boundary+voxel-interior+octree-fallback",
                             "full_label_miou": metrics["full_label"]["miou"],
                             "valid_depth_miou": metrics["valid_depth"]["miou"],
                             "valid_depth_coverage": metrics["valid_depth"]["coverage"],
                             "boundary_f1": boundary["2"]["f1"], **details,
                             "tune_score": tune_scores[selected_quantile]})
        print(f"adaptive hybrid {dataset}: q={selected_quantile:.2f}, tune={tune_scores[selected_quantile]:.4f}")
    return rows


def _plot_rate(rows: list[dict], output: Path):
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), squeeze=False)
    for column, dataset in enumerate(DATASETS):
        selected = [row for row in rows if row["dataset"] == dataset]
        for row_index, x_key in enumerate(("element_count", "construction_time_s")):
            axis = axes[row_index, column]
            for name in REPRESENTATIONS:
                points = []
                for variant in sorted({row["variant"] for row in selected
                                       if row["representation"] == name}):
                    group = [row for row in selected if row["representation"] == name
                             and row["variant"] == variant]
                    if not group:
                        continue
                    points.append({
                        "x": float(np.mean([row[x_key] for row in group])),
                        "y": float(np.mean([row["full_label_miou"] for row in group])) * 100,
                        "y_std": float(np.std([row["full_label_miou"] for row in group])) * 100,
                    })
                points.sort(key=lambda row: row["x"])
                axis.errorbar([row["x"] for row in points], [row["y"] for row in points],
                              yerr=[row["y_std"] for row in points], marker="o",
                              linewidth=1.4, markersize=4, capsize=2,
                              color=COLORS[name], label=DISPLAY[name])
            axis.set_xscale("log"); axis.set_ylim(-2, 102)
            axis.set_xlabel("elements (log scale)" if x_key == "element_count" else "construction time (s, log scale)")
            axis.set_ylabel("full-label mIoU (%)")
            axis.set_title(f"{dataset.upper() if dataset == 'nyuv2' else 'SUN RGB-D'} - quality versus {x_key.replace('_', ' ')}")
            axis.grid(axis="y", alpha=0.2); axis.set_axisbelow(True)
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Rate-distortion pilot: controlled representation resolution changes", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=160); plt.close(figure)


def _plot_corruption(rows: list[dict], output: Path):
    conditions = [item[0] for item in CORRUPTION_CONDITIONS]
    figure, axes = plt.subplots(2, 2, figsize=(18, 11), squeeze=False)
    for column, dataset in enumerate(DATASETS):
        selected = [row for row in rows if row["dataset"] == dataset]
        for row_index, metric in enumerate(("full_label_miou", "valid_depth_coverage")):
            axis = axes[row_index, column]
            for name in ROBUST_REPRESENTATIONS:
                means = []
                for condition in conditions:
                    values = [row[metric] for row in selected
                              if row["representation"] == name and row["condition"] == condition]
                    means.append(float(np.mean(values)) * 100 if values else np.nan)
                axis.plot(np.arange(len(conditions)), means, marker="o", linewidth=1.8,
                          color=COLORS[name], label=DISPLAY[name])
            axis.set_xticks(np.arange(len(conditions)), conditions, rotation=25, ha="right")
            axis.set_ylim(-2, 102); axis.set_ylabel("percent")
            axis.set_title(f"{dataset.upper() if dataset == 'nyuv2' else 'SUN RGB-D'} - {metric.replace('_', ' ')}")
            axis.grid(axis="y", alpha=0.2); axis.set_axisbelow(True)
    axes[0, 1].legend(frameon=False)
    figure.suptitle("Depth-corruption robustness on the fixed ten-image samples", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=160); plt.close(figure)


def _plot_hybrid(rows: list[dict], output: Path, output_root: str | Path):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)
    for axis, metric, title in zip(axes.flat, ("full_label_miou", "valid_depth_coverage"),
                                  ("held-out full-label mIoU", "held-out coverage")):
        positions = np.arange(4)
        width = 0.34
        for index, dataset in enumerate(DATASETS):
            ids = [row["sample"] for row in rows if row["dataset"] == dataset and row["split"] == "heldout"]
            values_by_rep = []
            for name in ("voxel", "mesh", "octree", "hybrid"):
                values = []
                if name == "hybrid":
                    values = [row[metric] for row in rows if row["dataset"] == dataset and row["split"] == "heldout"]
                else:
                    for sample_id in ids:
                        # The per-representation directory stores maps, while
                        # baseline metrics are recovered from the sample JSON.
                        sample_path = _base(output_root, dataset) / sample_id / "aggregate_metrics.json"
                        sample_json = json.loads(sample_path.read_text(encoding="utf-8"))
                        detail = sample_json["representations"][name]
                        values.append(detail["metrics"]["full_label" if metric == "full_label_miou" else "valid_depth"]["miou" if metric == "full_label_miou" else "coverage"])
                values_by_rep.append(np.asarray(values) * 100)
            for rep_index, values in enumerate(values_by_rep):
                if values.size:
                    axis.bar(positions[rep_index] + (index - 0.5) * width,
                             values.mean(), width * 0.92,
                             color="#2878b5" if dataset == "nyuv2" else "#d95f02",
                             alpha=0.82,
                             label=(dataset.upper() if dataset == "nyuv2" else "SUN RGB-D") if rep_index == 0 else None)
        axis.set_xticks(positions, ["Voxel", "Mesh", "Octree", "Adaptive hybrid"])
        axis.set_ylabel("percent"); axis.set_title(title); axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2); axis.set_axisbelow(True); axis.legend(frameon=False)
    figure.suptitle("Geometry-only adaptive hybrid versus fixed representations", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=160); plt.close(figure)


def _write_efficiency(output_root: str | Path, output_dir: Path, docs: Path):
    rows = []
    for dataset in DATASETS:
        means = _mean_rows(_sample_records(output_root, dataset), dataset)
        for name, value in means.items():
            geometry_mb = value["element_count"] * 3 * 4 / 1_000_000
            rows.append({"dataset": dataset, "representation": name,
                         "full_label_miou": value["full_label_miou"],
                         "coverage": value["valid_depth_coverage"],
                         "runtime_s": value["construction_time_s"],
                         "element_count": value["element_count"],
                         "xyz_geometry_mb_lower_bound": geometry_mb,
                         "mIoU_per_second": value["miou_per_second"]})
    _write_csv(output_dir / "representation_efficiency.csv", rows)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    for axis, x_key, xlabel in zip(axes.flat, ("runtime_s", "xyz_geometry_mb_lower_bound"),
                                   ("construction time (s, log scale)", "XYZ geometry lower bound (MB, log scale)")):
        for dataset in DATASETS:
            selected = [row for row in rows if row["dataset"] == dataset]
            axis.scatter([row[x_key] for row in selected], [row["full_label_miou"] * 100 for row in selected],
                         color="#2878b5" if dataset == "nyuv2" else "#d95f02", label=dataset.upper() if dataset == "nyuv2" else "SUN RGB-D")
            for row in selected:
                if row["full_label_miou"] > 0.05:
                    axis.annotate(DISPLAY[row["representation"]], (row[x_key], row["full_label_miou"] * 100),
                                  xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xscale("log"); axis.set_ylim(-2, 102); axis.set_xlabel(xlabel); axis.set_ylabel("full-label mIoU (%)")
        axis.grid(axis="y", alpha=0.2); axis.set_axisbelow(True); axis.legend(frameon=False)
    figure.suptitle("Runtime and geometry-storage efficiency on the ten-image pilots", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "representation_efficiency.png"; figure.savefig(path, dpi=160); plt.close(figure)
    docs.mkdir(parents=True, exist_ok=True); (docs / path.name).write_bytes(path.read_bytes())


def _write_report(output_dir: Path):
    lines = [
        "# Small-sample RGB-D experiments",
        "",
        "All experiments use exactly ten NYUv2 and ten SUN RGB-D images. The same image IDs are reused across settings.",
        "",
        "## Experiments",
        "",
        "1. **Rate-distortion:** baseline plus controlled lower/higher representation resolutions for all nine styles.",
        "2. **Depth corruption:** clean, 10%/30% random depth dropout, and 1 cm/3 cm Gaussian noise for voxel, octree, and superpoint.",
        "3. **Adaptive hybrid:** mesh at geometry-only depth/RGB edges, voxel in interiors, and octree as fallback; the edge quantile is tuned on five images and evaluated on five held-out images.",
        "4. **Runtime/memory:** construction time and an XYZ-only geometry-storage lower bound for every baseline style.",
        "5. **Variability/coverage:** the existing extended analysis reports per-image distributions and coverage-optimism gaps.",
        "",
        "These are pilot experiments. They support comparison and debugging, not universal claims about all indoor RGB-D data.",
    ]
    (output_dir / "small_sample_experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the ten-image RGB-D follow-up experiments")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--docs-figures", default="docs/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / "experiments"
    docs = Path(args.docs_figures)
    configs = {dataset: load_config(Path(args.config_root) / f"{dataset}_smoke.yaml") for dataset in DATASETS}
    adapters = {dataset: build_adapter(configs[dataset]["dataset"]) for dataset in DATASETS}
    cache = _sample_cache(adapters, args.output_root)

    rate_rows = run_rate_distortion(args.output_root, configs, cache)
    _write_csv(output_dir / "rate_distortion_metrics.csv", rate_rows,
               ("dataset", "sample", "representation", "variant", *METRIC_FIELDS))
    _plot_rate(rate_rows, output_dir / "rate_distortion.png")
    docs.mkdir(parents=True, exist_ok=True); (docs / "rate_distortion.png").write_bytes((output_dir / "rate_distortion.png").read_bytes())

    corruption_rows = run_corruption(args.output_root, configs, cache)
    _write_csv(output_dir / "depth_corruption_metrics.csv", corruption_rows,
               ("dataset", "sample", "representation", "variant", "condition", *METRIC_FIELDS))
    _plot_corruption(corruption_rows, output_dir / "depth_corruption_robustness.png")
    (docs / "depth_corruption_robustness.png").write_bytes((output_dir / "depth_corruption_robustness.png").read_bytes())

    hybrid_rows = run_adaptive_hybrid(args.output_root, configs, cache)
    _write_csv(output_dir / "adaptive_hybrid_metrics.csv", hybrid_rows)
    _plot_hybrid(hybrid_rows, output_dir / "adaptive_hybrid_results.png", args.output_root)
    (docs / "adaptive_hybrid_results.png").write_bytes((output_dir / "adaptive_hybrid_results.png").read_bytes())

    _write_efficiency(args.output_root, output_dir, docs)
    _write_report(output_dir)
    print(f"Wrote small-sample experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
