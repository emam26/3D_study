"""Create contact sheets from already-generated representation panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def make_overviews(output_root="outputs", datasets=("nyuv2", "sunrgbd"),
                   studies=None, representations=None):
    output_root = Path(output_root)
    studies = studies or {"nyuv2": "nyuv2_representation_smoke",
                          "sunrgbd": "sunrgbd_representation_smoke"}
    made = []
    for dataset in datasets:
        study_dir = output_root / dataset / studies[dataset]
        if not study_dir.exists():
            continue
        sample_dirs = sorted(path for path in study_dir.iterdir() if path.is_dir())
        for sample_dir in sample_dirs:
            panel_paths = sorted(sample_dir.glob("*/representation_panel.png"))
            if representations:
                panel_paths = [sample_dir / name / "representation_panel.png" for name in representations]
            panel_paths = [path for path in panel_paths if path.exists()]
            if not panel_paths:
                continue
            columns = 2
            rows = (len(panel_paths) + columns - 1) // columns
            figure, axes = plt.subplots(rows, columns, figsize=(14, 5 * rows), squeeze=False)
            for index, path in enumerate(panel_paths):
                axis = axes[index // columns][index % columns]
                axis.imshow(Image.open(path))
                axis.set_title(path.parent.name)
                axis.axis("off")
            for index in range(len(panel_paths), rows * columns):
                axes[index // columns][index % columns].axis("off")
            figure.suptitle(f"{dataset} / sample {sample_dir.name}", fontsize=15)
            figure.tight_layout()
            output = sample_dir / "all_representations_overview.png"
            figure.savefig(output, dpi=110)
            plt.close(figure)
            made.append(output)
    return made


def main():
    parser = argparse.ArgumentParser(description="Create batch representation contact sheets")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    paths = make_overviews(args.output_root)
    print(f"Created {len(paths)} overview images")


if __name__ == "__main__":
    main()
