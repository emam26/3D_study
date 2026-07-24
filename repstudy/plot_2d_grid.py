"""Render one two-dimensional oracle-reconstruction grid per dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .datasets import build_adapter
from .oracle import assign_oracle_labels, render_original_view
from .pipeline import load_config
from .representations import build_representation


REPRESENTATION_ORDER = ("pointcloud", "surfel", "mesh", "voxel", "tsdf", "superpoint",
                        "graph", "octree", "descriptor")


def make_2d_grid(config, sample_id=None, output=None):
    adapter = build_adapter(config["dataset"])
    sample_id = sample_id or adapter.sample_ids()[0]
    sample = adapter.load(sample_id)
    representations = [name for name in REPRESENTATION_ORDER
                       if config.get("representations", {}).get(name, {}).get("enabled", True)]
    ncols = 4
    nslots = len(representations) + 2
    nrows = int(np.ceil(nslots / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(22, 5.5 * nrows), squeeze=False)
    rgb = sample.rgb.astype(np.float32) / (255 if sample.rgb.max() > 1.5 else 1)
    axes[0][0].imshow(rgb); axes[0][0].set_title("RGB reference"); axes[0][0].axis("off")
    axes[0][1].imshow(sample.semantic_gt, cmap="tab20"); axes[0][1].set_title("Semantic ground truth"); axes[0][1].axis("off")
    for index, name in enumerate(representations):
        result = build_representation(name, sample, config.get("representations", {}).get(name, {}))
        result = assign_oracle_labels(result, sample)
        rendered = render_original_view(result, sample)
        axis = axes[(index + 2) // 4][(index + 2) % 4]
        display = rendered.semantic_map.astype(np.float32)
        display[~rendered.coverage_mask] = np.nan
        axis.imshow(display, cmap="tab20", vmin=0, vmax=max(int(sample.semantic_gt.max()), 1))
        coverage = 100.0 * float(rendered.coverage_mask.mean())
        axis.set_title(f"{name} oracle projection ({coverage:.1f}% covered)")
        axis.axis("off")
    for index in range(len(representations) + 2, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    figure.suptitle(f"{sample.dataset_name}: 2D representation projections", fontsize=16)
    figure.tight_layout()
    if output is None:
        output = Path(config["study"].get("output_root", "outputs")) / config["dataset"]["name"] / config["study"]["name"] / "2d_representation_grid.png"
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140); plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render a 2D representation projection grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = make_2d_grid(load_config(args.config), args.sample_id, args.output)
    print(path)


if __name__ == "__main__":
    main()
