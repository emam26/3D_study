"""Run a reproducible multi-sample representation oracle study."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .pipeline import load_config, run_from_config
from .datasets import build_adapter


def _finite_values(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return np.asarray(values, dtype=np.float64)


def _summary(values):
    values = _finite_values(values)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def aggregate_summaries(summaries):
    """Aggregate per-image JSON summaries while retaining coverage statistics."""
    rows = {}
    for sample in summaries:
        for name, details in sample.get("representations", {}).items():
            row = rows.setdefault(name, {"element_count": [], "construction_time_s": [],
                                         "element_purity": [], "boundary_crossing_rate": [],
                                         "valid_depth_miou": [], "valid_depth_pixel_accuracy": [],
                                         "valid_depth_coverage": [], "valid_depth_missing_rate": [],
                                         "full_label_miou": [], "full_label_pixel_accuracy": [],
                                         "boundary_f1_tolerance_2": []})
            row["element_count"].append(details.get("element_count"))
            row["construction_time_s"].append(details.get("construction_time_s"))
            row["element_purity"].append(details.get("element_purity"))
            row["boundary_crossing_rate"].append(details.get("boundary_crossing_rate"))
            valid = details.get("metrics", {}).get("valid_depth", {})
            full = details.get("metrics", {}).get("full_label", {})
            row["valid_depth_miou"].append(valid.get("miou"))
            row["valid_depth_pixel_accuracy"].append(valid.get("pixel_accuracy"))
            row["valid_depth_coverage"].append(valid.get("coverage"))
            row["valid_depth_missing_rate"].append(valid.get("missing_rate"))
            row["full_label_miou"].append(full.get("miou"))
            row["full_label_pixel_accuracy"].append(full.get("pixel_accuracy"))
            row["boundary_f1_tolerance_2"].append(details.get("boundary", {}).get("2", {}).get("f1"))

    output = {}
    for name, values in rows.items():
        output[name] = {metric: _summary(items) for metric, items in values.items()}
    return output


def _write_csv(path, aggregate):
    fields = ["representation", "samples", "element_count_mean", "construction_time_s_mean",
              "element_purity_mean", "boundary_crossing_rate_mean", "valid_depth_miou_mean",
              "valid_depth_pixel_accuracy_mean", "valid_depth_coverage_mean",
              "valid_depth_missing_rate_mean", "full_label_miou_mean",
              "full_label_pixel_accuracy_mean", "boundary_f1_tolerance_2_mean"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metrics in aggregate.items():
            row = {"representation": name}
            for metric, stats in metrics.items():
                row[f"{metric}_mean"] = stats["mean"]
                if metric == "element_count":
                    row["samples"] = stats["count"]
            writer.writerow(row)


def run_study(config, sample_ids=None, max_samples=None, overwrite=False):
    adapter = build_adapter(config["dataset"])
    available = adapter.sample_ids()
    selected = list(sample_ids) if sample_ids else available
    if max_samples is None:
        max_samples = config.get("study", {}).get("max_samples")
    if max_samples is not None and int(max_samples) > 0:
        selected = selected[: int(max_samples)]

    study = config["study"]
    base = Path(study.get("output_root", "outputs")) / config["dataset"]["name"] / study["name"]
    base.mkdir(parents=True, exist_ok=True)
    summaries, failures = [], []
    started = time.perf_counter()
    for index, sample_id in enumerate(selected, 1):
        summary_path = base / sample_id / "aggregate_metrics.json"
        try:
            if summary_path.exists() and not overwrite:
                with summary_path.open(encoding="utf-8") as handle:
                    summary = json.load(handle)
                status = "cached"
            else:
                summary = run_from_config(config, sample_id)
                status = "computed"
            summaries.append(summary)
            print(f"[{index}/{len(selected)}] {sample_id}: {status}")
        except Exception as exc:  # keep a pilot run resumable when one sample is malformed
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(selected)}] {sample_id}: FAILED ({failures[-1]['error']})")

    payload = {
        "dataset": config["dataset"]["name"],
        "split": config["dataset"].get("split"),
        "study": study["name"],
        "requested_samples": len(selected),
        "completed_samples": len(summaries),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "representations": aggregate_summaries(summaries),
    }
    with (base / "study_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    _write_csv(base / "study_summary.csv", payload["representations"])
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run a multi-sample RGB-D representation oracle study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", action="append", default=None,
                        help="Specific sample ID; repeat the flag for multiple samples")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override study.max_samples; <=0 means all available samples")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing sample outputs")
    args = parser.parse_args()
    result = run_study(load_config(args.config), args.sample_id, args.max_samples, args.overwrite)
    print(json.dumps({"completed_samples": result["completed_samples"],
                      "failures": len(result["failures"]),
                      "representations": list(result["representations"])}, indent=2))


if __name__ == "__main__":
    main()
