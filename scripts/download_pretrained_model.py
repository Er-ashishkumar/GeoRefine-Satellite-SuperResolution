"""
One-time download of the pretrained Real-ESRGAN x4plus checkpoint.

This script is never called automatically by any other part of the
pipeline - it must be run explicitly, once, before src/model/super_resolution.py
can load a model. This keeps the pipeline's network usage explicit rather
than hidden inside a training or inference script.

Usage (from repo root):
    python scripts/download_pretrained_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
MODEL_FILENAME = "RealESRGAN_x4plus.pth"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / MODEL_FILENAME

    if out_path.exists():
        print(f"Already present: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) - skipping download.")
        return

    print(f"Downloading {MODEL_URL}")
    print(f"  -> {out_path}")
    try:
        urlretrieve(MODEL_URL, out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        print(
            "If this is a network/proxy issue, you can manually download the file "
            f"from:\n  {MODEL_URL}\nand place it at:\n  {out_path}"
        )
        sys.exit(1)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Downloaded successfully: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()