from __future__ import annotations

import argparse
import json

from .pipeline import load_config, run_from_config


def main():
    parser = argparse.ArgumentParser(description="Run one RGB-D representation oracle sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", default=None)
    args = parser.parse_args()
    result = run_from_config(load_config(args.config), args.sample_id)
    print(json.dumps({"sample_id": result["sample_id"], "representations": list(result["representations"])}, indent=2))


if __name__ == "__main__": main()


