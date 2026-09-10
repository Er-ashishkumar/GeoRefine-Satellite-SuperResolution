"""
CLI entry point: scan data/raw and build the dataset index.

Usage (from repo root):
    python scripts/build_dataset_index.py
    python scripts/build_dataset_index.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/build_dataset_index.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.registry import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GeoRefine dataset index.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["data_raw"])
    processed_dir = Path(config["paths"]["data_processed"])
    dataset_cfg = config.get("dataset", {})
    lr_subdir = dataset_cfg.get("lr_subdir", "lr")
    hr_subdir = dataset_cfg.get("hr_subdir", "hr")
    index_filename = dataset_cfg.get("index_filename", "dataset_index.json")

    output_path = processed_dir / index_filename

    index = build_index(raw_dir, output_path, lr_subdir=lr_subdir, hr_subdir=hr_subdir)

    print(f"Scanned: {raw_dir}")
    print(f"Scenes found:          {index['num_scenes']}")
    print(f"Scenes with HR ref:    {index['num_scenes_with_hr']}")
    print(f"Scenes with issues:    {index['num_scenes_with_issues']}")
    print(f"Index written to:      {output_path}")

    if index["num_scenes"] == 0:
        print(
            "\nNo scenes found. Add imagery under "
            f"{raw_dir}/<scene_id>/{lr_subdir}/*.tif "
            "(see data/README.md for the expected layout)."
        )


if __name__ == "__main__":
    main()