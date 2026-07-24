"""Sample/study orchestration and reproducible output contract."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .datasets import build_adapter
from .metrics import boundary_metrics, component_metrics, semantic_metrics
from .multiview import render_views
from .oracle import assign_oracle_labels, render_original_view
from .representations import build_representation


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.integer): return value.item()
    if isinstance(value, np.floating):
        scalar = value.item()
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def save_source_panel(sample, output):
    output.mkdir(parents=True, exist_ok=True)
    rgb = sample.rgb.astype(np.float32) / (255 if sample.rgb.max() > 1.5 else 1)
    depth = sample.depth_m
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb); axes[0].set_title("RGB")
    axes[1].imshow(depth, cmap="magma"); axes[1].set_title("Depth (m)")
    axes[2].imshow(sample.semantic_gt, cmap="tab20"); axes[2].set_title("Ground truth")
    axes[3].imshow(sample.valid_depth_mask, cmap="gray"); axes[3].set_title("Valid depth")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(output / "source_panel.png", dpi=120); plt.close(fig)


def save_representation_panel(sample, result, render, output):
    output.mkdir(parents=True, exist_ok=True)
    rgb = sample.rgb.astype(np.float32) / (255 if sample.rgb.max() > 1.5 else 1)
    error = np.full(sample.semantic_gt.shape, 3, np.int32)
    covered = render.coverage_mask
    error[covered & (render.semantic_map == sample.semantic_gt)] = 0
    error[covered & (render.semantic_map != sample.semantic_gt)] = 1
    error[sample.valid_label_mask & ~covered] = 2
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(rgb); axes[0].set_title(result.name)
    axes[1].imshow(render.coverage_mask, cmap="gray"); axes[1].set_title("Coverage")
    axes[2].imshow(sample.semantic_gt, cmap="tab20"); axes[2].set_title("Ground truth")
    axes[3].imshow(render.semantic_map, cmap="tab20"); axes[3].set_title("Oracle reconstruction")
    axes[4].imshow(error, cmap="viridis", vmin=0, vmax=3); axes[4].set_title("0 correct / 1 wrong / 2 missing")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(output / "representation_panel.png", dpi=120); plt.close(fig)


def run_sample(sample, representations, output_root, representation_configs=None, views=None):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    save_source_panel(sample, output_root / "source")
    summary = {"sample_id": sample.sample_id, "dataset": sample.dataset_name, "representations": {}}
    representation_configs = representation_configs or {}
    for name in representations:
        started = time.perf_counter()
        result = build_representation(name, sample, representation_configs.get(name, {}))
        result = assign_oracle_labels(result, sample)
        render = render_original_view(result, sample)
        elapsed = time.perf_counter() - started
        semantic = semantic_metrics(render.semantic_map, sample.semantic_gt, render.coverage_mask, sample.valid_label_mask, sample.valid_depth_mask)
        boundary = boundary_metrics(render.semantic_map, sample.semantic_gt, sample.valid_label_mask)
        components = component_metrics(render.semantic_map, sample.semantic_gt)
        rep_dir = output_root / name
        save_representation_panel(sample, result, render, rep_dir)
        np.savez_compressed(rep_dir / "render_maps.npz", semantic_map=render.semantic_map, coverage_mask=render.coverage_mask, collision_map=render.collision_map, element_id_map=render.element_id_map)
        views_dir = rep_dir / "multiview"
        views_dir.mkdir(parents=True, exist_ok=True)
        for view, view_result in render_views(result, sample, views).items():
            np.savez_compressed(views_dir / f"{view}.npz", semantic_map=view_result.semantic_map, coverage_mask=view_result.coverage_mask, element_id_map=view_result.element_id_map, collision_map=view_result.collision_map)
        summary["representations"][name] = _jsonable({
            "element_count": result.element_count, "construction_time_s": elapsed,
            "element_purity": float(result.attributes.get("purity", np.zeros(1)).mean()) if result.element_count else 0.0,
            "boundary_crossing_rate": float(result.attributes.get("boundary_crossing", np.zeros(1)).mean()) if result.element_count else 0.0,
            "metrics": semantic, "boundary": boundary, "components": components,
            "parameters": result.construction_parameters,
        })
    with open(output_root / "aggregate_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def load_config(path):
    with open(path, encoding="utf-8") as handle: return yaml.safe_load(handle)


def run_from_config(config, sample_id=None):
    adapter = build_adapter(config["dataset"])
    sample_id = sample_id or adapter.sample_ids()[0]
    sample = adapter.load(sample_id)
    reps = [name for name, value in config.get("representations", {}).items() if value.get("enabled", True)]
    rep_configs = {name: value for name, value in config.get("representations", {}).items()}
    output = Path(config["study"].get("output_root", "outputs")) / config["dataset"]["name"] / config["study"]["name"] / sample_id
    return run_sample(sample, reps, output, rep_configs, config.get("views"))
