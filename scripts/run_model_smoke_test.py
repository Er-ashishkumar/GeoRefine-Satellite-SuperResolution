"""
Smoke test CLI: confirm the pretrained SR model loads and runs correctly.
This is NOT a training or evaluation script - it only checks that:
  1. The pretrained checkpoint loads without error.
  2. A forward pass produces output of the expected shape (H*scale, W*scale).
Usage (from repo root):
    python scripts/run_model_smoke_test.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import rasterio
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.super_resolution import apply_super_resolution, load_model
def _find_a_real_lr_patch(config: dict) -> Path | None:
    processed_dir = Path(config["paths"]["data_processed"])
    patches_index_path = processed_dir / config.get("patches", {}).get("index_filename", "patches_index.json")
    if not patches_index_path.exists():
        return None
    with open(patches_index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    for scene in index.get("scenes", []):
        for patch in scene.get("patches", []):
            lr_path = Path(patch["lr_path"])
            if lr_path.exists():
                return lr_path
    return None
def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loading pretrained model...")
    model = load_model(models_dir=config["paths"]["models"])
    print(f"Model loaded. Scale factor: {model.scale}")
    lr_patch_path = _find_a_real_lr_patch(config)
    if lr_patch_path is not None:
        print(f"Using real LR patch: {lr_patch_path}")
        with rasterio.open(lr_patch_path) as src:
            lr_array = src.read()
    else:
        print("No real patches found (patches_index.json missing or empty). Using a small synthetic array.")
        lr_array = (np.random.rand(2, 32, 32) * 200 + 10).astype("float32")
    print(f"Input shape:  {lr_array.shape}")
    sr_array = apply_super_resolution(model, lr_array)
    print(f"Output shape: {sr_array.shape}")
    expected_shape = (lr_array.shape[0], lr_array.shape[1] * model.scale, lr_array.shape[2] * model.scale)
    if sr_array.shape == expected_shape:
        print(f"OK: output shape matches expected {expected_shape}")
    else:
        print(f"MISMATCH: expected {expected_shape}, got {sr_array.shape}")
        sys.exit(1)
if __name__ == "__main__":
    main()
