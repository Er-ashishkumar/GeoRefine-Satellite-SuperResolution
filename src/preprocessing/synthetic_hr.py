"""
Synthetic HR fallback pipeline (dataset.synthetic_hr_fallback in config.yaml).

data/README.md documents the strategy: when a scene has no real,
independently-sourced HR reference under data/raw/<scene_id>/hr/, a
synthetic degradation is used instead of leaving the scene unusable for
supervised training/validation.

Concrete interpretation used by this implementation (this is the one part
of Phase 1's data/README.md description that is ambiguous purely from the
prose, so it is made explicit and testable here):

    proxy_hr   = the scene's actual acquired LR imagery (data/raw/<scene_id>/lr/),
                 i.e. the best real resolution available for that scene.
                 It is NOT independently-sourced higher-resolution ground
                 truth, so it is only a "proxy" for HR - hence is_synthetic_hr.
    synthetic_lr = gaussian_blur(proxy_hr, blur_sigma) then
                   block-mean-downsample(., downsample_factor)

This is the standard self-supervised degradation setup used when no real
HR reference exists: you cannot synthesize genuine higher-resolution detail
that was never captured, so instead you synthesize a *lower*-resolution
companion to the real image you do have, and train/validate the model's
ability to reconstruct that known, deliberately-applied degradation. This
measures reconstruction-of-a-known-degradation, not recovery of real-world
detail - the same caveat as GeoRefine's top-level scientific honesty notice.

No new pixel values are invented for the HR side of the pair; only the LR
side is synthetically generated, by applying a known, documented
degradation to real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio import Affine
from scipy.ndimage import gaussian_filter


@dataclass
class SyntheticPairResult:
    scene_id: str
    proxy_hr_paths: List[str]  # the real source file(s), unmodified, used as proxy HR
    synthetic_lr_path: str  # newly written, degraded raster
    band_order: List[str]
    blur_sigma: float
    downsample_factor: int
    proxy_hr_shape: tuple
    synthetic_lr_shape: tuple


def _blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur each band independently (never blur across the band axis)."""
    if sigma <= 0:
        return array
    out = np.empty_like(array, dtype=np.float64)
    for b in range(array.shape[0]):
        out[b] = gaussian_filter(array[b].astype(np.float64), sigma=sigma)
    return out


def _block_mean_downsample(array: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool each band by `factor` in both spatial dimensions.

    Cropping any remainder rows/columns that don't evenly divide by `factor`,
    which keeps the operation simple and its output size exactly
    (H // factor, W // factor) - documented behavior, not a silent surprise.
    """
    if factor <= 1:
        return array
    bands, height, width = array.shape
    new_h, new_w = height // factor, width // factor
    cropped = array[:, : new_h * factor, : new_w * factor]
    reshaped = cropped.reshape(bands, new_h, factor, new_w, factor)
    return reshaped.mean(axis=(2, 4))


def degrade_to_synthetic_lr(proxy_hr: np.ndarray, blur_sigma: float, downsample_factor: int) -> np.ndarray:
    """Apply the documented degradation: blur, then block-mean downsample."""
    blurred = _blur(proxy_hr, blur_sigma)
    return _block_mean_downsample(blurred, downsample_factor)


def generate_synthetic_pair_for_scene(
    scene_id: str,
    proxy_hr_array: np.ndarray,
    band_order: List[str],
    source_paths: List[str],
    crs: str,
    transform_6: tuple,
    blur_sigma: float,
    downsample_factor: int,
    output_dir: Path,
    dtype: str = "float32",
) -> SyntheticPairResult:
    """Generate and write the synthetic LR companion for one scene, in memory-safe steps.

    proxy_hr_array: the scene's real band stack (bands, H, W) - already loaded,
        e.g. via src.preprocessing.bands.load_band_stack.
    Writes to: <output_dir>/<scene_id>/synthetic_lr.tif
    """
    synthetic_lr = degrade_to_synthetic_lr(proxy_hr_array, blur_sigma, downsample_factor).astype(dtype)

    scene_dir = Path(output_dir) / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_path = scene_dir / "synthetic_lr.tif"

    src_transform = Affine(*transform_6)
    # Scale the transform's pixel size to match the downsampled grid.
    dst_transform = src_transform @ Affine.scale(downsample_factor, downsample_factor)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=synthetic_lr.shape[1],
        width=synthetic_lr.shape[2],
        count=synthetic_lr.shape[0],
        dtype=dtype,
        crs=crs,
        transform=dst_transform,
    ) as dst:
        dst.write(synthetic_lr)

    return SyntheticPairResult(
        scene_id=scene_id,
        proxy_hr_paths=source_paths,
        synthetic_lr_path=str(out_path),
        band_order=band_order,
        blur_sigma=blur_sigma,
        downsample_factor=downsample_factor,
        proxy_hr_shape=tuple(proxy_hr_array.shape),
        synthetic_lr_shape=tuple(synthetic_lr.shape),
    )
