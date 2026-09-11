"""
CLI entry point: run Phase 2 preprocessing on top of an existing dataset index.

Usage (from repo root):
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --config config.yaml

Requires data/processed/dataset_index.json to already exist - run
scripts/build_dataset_index.py first (Phase 1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/run_preprocessing.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.pipeline import build_preprocessing_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoRefine Phase 2 preprocessing.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["paths"]["data_processed"])
    dataset_index_filename = config.get("dataset", {}).get("index_filename", "dataset_index.json")
    dataset_index_path = processed_dir / dataset_index_filename

    if not dataset_index_path.exists():
        print(
            f"dataset_index.json not found at {dataset_index_path}.\n"
            "Run `python scripts/build_dataset_index.py` first (Phase 1)."
        )
        sys.exit(1)

    output_filename = config.get("preprocessing", {}).get("output_index_filename", "preprocessing_index.json")
    output_path = processed_dir / output_filename

    index = build_preprocessing_index(dataset_index_path, config, output_path)

    print(f"Scenes processed:              {index['num_scenes']}")
    print(f"Scenes with real HR aligned:   {index['num_scenes_with_real_hr_aligned']}")
    print(f"Scenes using synthetic HR:     {index['num_scenes_synthetic_hr']}")
    print(f"Scenes with issues:            {index['num_scenes_with_issues']}")
    print(f"Index written to:              {output_path}")

    if index["num_scenes"] == 0:
        print(
            "\nNo scenes found in dataset_index.json. Add raw imagery and re-run "
            "scripts/build_dataset_index.py before running preprocessing."
        )


if __name__ == "__main__":
    main()
