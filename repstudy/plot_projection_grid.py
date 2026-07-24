"""Render camera, oblique, and elevated oracle projections for key representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .datasets import build_adapter
from .multiview import render_views
from .oracle import assign_oracle_labels
from .pipeline import load_config
from .representations import build_representation


REPRESENTATIONS = ("pointcloud", "graph", "octree", "descriptor")
VIEWS = ("original", "left_oblique", "right_oblique", "elevated")


def make_projection_grid(config, sample_id=None, output=None):
    adapter = build_adapter(config["dataset"])
    sample_id = sample_id or adapter.sample_ids()[0]
    sample = adapter.load(sample_id)
    reps = [name for name in REPRESENTATIONS
            if config.get("representations", {}).get(name, {}).get("enabled", True)]
    rows = len(reps) + 1
    figure, axes = plt.subplots(rows, len(VIEWS), figsize=(22, 5.0 * rows), squeeze=False)

    rgb = sample.rgb.astype(np.float32) / (255 if sample.rgb.max() > 1.5 else 1)
    axes[0][0].imshow(rgb); axes[0][0].set_title("RGB reference")
    axes[0][1].imshow(sample.semantic_gt, cmap="tab20"); axes[0][1].set_title("Semantic ground truth")
    axes[0][2].axis("off"); axes[0][3].axis("off")
    for axis in axes[0]:
        axis.axis("off")

    for row, name in enumerate(reps, 1):
        result = build_representation(name, sample, config.get("representations", {}).get(name, {}))
        result = assign_oracle_labels(result, sample)
        renders = render_views(result, sample, VIEWS)
        for column, view in enumerate(VIEWS):
            rendered = renders[view]
            display = rendered.semantic_map.astype(np.float32)
            display[~rendered.coverage_mask] = np.nan
            axis = axes[row][column]
            axis.imshow(display, cmap="tab20", vmin=0, vmax=max(int(sample.semantic_gt.max()), 1))
            coverage = 100.0 * float(rendered.coverage_mask.mean())
            axis.set_title(f"{name} / {view} ({coverage:.1f}% covered)")
            axis.axis("off")

    figure.suptitle(f"{sample.dataset_name}: multiview projections", fontsize=16)
    figure.tight_layout()
    if output is None:
        output = (Path(config["study"].get("output_root", "outputs")) /
                  config["dataset"]["name"] / config["study"]["name"] /
                  "projection_grid.png")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140); plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render multiview projection grids")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = make_projection_grid(load_config(args.config), args.sample_id, args.output)
    print(path)


if __name__ == "__main__":
    main()
