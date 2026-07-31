"""Run the complete V1 screening pipeline on NYUv2 and SUN RGB-D.

This wrapper keeps the two dataset runs consistent and verifies that each run
produced a checkpoint, metrics JSON, and qualitative grid. Use ``--smoke`` only
for a fast local pipeline check; the default is the full one-epoch protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CONFIGS = (
    "configs/v1_rgb_segformer_nyuv2.yaml",
    "configs/v1_rgb_segformer_sunrgbd.yaml",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run two samples at 32x32 with random weights for pipeline testing.",
    )
    parser.add_argument("--num-visual-samples", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    env = None
    summary = {"protocol": "v1_smoke" if args.smoke else "v1_full", "datasets": {}}

    for config in CONFIGS:
        command = [
            sys.executable,
            "train_v1_rgb_segformer.py",
            "--config",
            config,
            "--num-visual-samples",
            str(args.num_visual_samples),
        ]
        if args.smoke:
            command += [
                "--no-pretrained",
                "--max-train-samples", "2",
                "--max-val-samples", "2",
                "--image-size", "32", "32",
                "--num-workers", "0",
            ]
        started = time.perf_counter()
        subprocess.run(command, cwd=root, env=env, check=True)
        elapsed = time.perf_counter() - started

        dataset_name = "nyuv2" if "nyuv2" in config else "sunrgbd"
        output_root = root / "outputs" / "segmentation" / f"v1_rgb_segformer_{dataset_name}"
        metrics_path = output_root / "metrics.json"
        checkpoint_path = output_root / "v1_rgb_segformer.pth"
        visualization_path = output_root / f"{dataset_name}_qualitative_grid.png"
        for path in (metrics_path, checkpoint_path, visualization_path):
            if not path.exists():
                raise FileNotFoundError(f"V1 run did not produce expected output: {path}")
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        summary["datasets"][dataset_name] = {
            "elapsed_seconds": round(elapsed, 3),
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "visualization": str(visualization_path),
        }

    subprocess.run(
        [sys.executable, "plot_architecture_comparison.py"],
        cwd=root, env=env, check=True,
    )
    summary_path = root / "outputs" / "segmentation" / "v1_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"V1 complete: {summary_path}")


if __name__ == "__main__":
    main()
