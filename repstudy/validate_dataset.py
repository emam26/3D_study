from __future__ import annotations

import argparse
import json

from .datasets import build_adapter
from .pipeline import load_config


def main():
    parser = argparse.ArgumentParser(description="Validate RGB-D dataset alignment and calibration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    adapter = build_adapter(config["dataset"])
    maximum = config.get("study", {}).get("max_samples", 5)
    report = {"dataset": config["dataset"]["name"], "split": config["dataset"].get("split", "test"), "samples": len(adapter), "checked": 0, "failures": []}
    for sample_id in adapter.sample_ids()[:maximum]:
        try:
            sample = adapter.load(sample_id)
            report["checked"] += 1
            if sample.valid_depth_mask.shape != sample.semantic_gt.shape:
                raise ValueError("depth/label shape mismatch")
        except Exception as error:
            report["failures"].append({"sample_id": sample_id, "error": str(error)})
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__": main()
