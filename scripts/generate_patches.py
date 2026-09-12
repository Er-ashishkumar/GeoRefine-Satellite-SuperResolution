"""
CLI entry point: generate LR/HR patch pairs from preprocessing_index.json.

Usage (from repo root):
    python scripts/generate_patches.py
    python scripts/generate_patches.py --config config.yaml

Requires data/processed/dataset_index.json (Phase 1) and
data/processed/preprocessing_index.json (Phase 2) to already exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/generate_patches.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.patch_pipeline import build_patches_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoRefine Phase 3 patch generation.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["paths"]["data_processed"])
    dataset_index_filename = config.get("dataset", {}).get("index_filename", "dataset_index.json")
    preprocessing_index_filename = config.get("preprocessing", {}).get(
        "output_index_filename", "preprocessing_index.json"
    )

    dataset_index_path = processed_dir / dataset_index_filename
    preprocessing_index_path = processed_dir / preprocessing_index_filename

    if not dataset_index_path.exists():
        print(
            f"dataset_index.json not found at {dataset_index_path}.\n"
            "Run `python scripts/build_dataset_index.py` first (Phase 1)."
        )
        sys.exit(1)

    if not preprocessing_index_path.exists():
        print(
            f"preprocessing_index.json not found at {preprocessing_index_path}.\n"
            "Run `python scripts/run_preprocessing.py` first (Phase 2)."
        )
        sys.exit(1)

    output_filename = config.get("patches", {}).get("index_filename", "patches_index.json")
    output_path = processed_dir / output_filename

    index = build_patches_index(preprocessing_index_path, dataset_index_path, config, output_path)

    print(f"Scenes processed:       {index['num_scenes']}")
    print(f"Total patches written:  {index['num_total_patches']}")
    print(f"Total patches rejected: {index['num_total_rejected']}")
    print(f"Scenes with issues:     {index['num_scenes_with_issues']}")
    print(f"Index written to:       {output_path}")

    if index["num_scenes"] == 0:
        print(
            "\nNo scenes found in preprocessing_index.json. Add raw imagery and "
            "re-run build_dataset_index.py then run_preprocessing.py before "
            "generating patches."
        )


if __name__ == "__main__":
    main()