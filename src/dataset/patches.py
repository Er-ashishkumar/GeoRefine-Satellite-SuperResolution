"""
Deterministic LR/HR patch grid computation and cropping for GeoRefine.

This module is intentionally pure (no file I/O): it operates on already
loaded numpy arrays and known scale factors. File reading, writing, and
orchestration across scenes live in patch_pipeline.py. Keeping this layer
pure makes patch logic easy to unit test without touching disk or rasterio
open/close semantics.

Grid convention: patches are laid out over the HR array's extent, top-left
anchored, in row-major order, with a fixed size and stride (no randomness -
see config.yaml's `patches` section). For each HR patch position
(hr_row, hr_col), the corresponding LR patch position is
(hr_row // scale_factor, hr_col // scale_factor), and the LR patch size is
patch_size // scale_factor.

Important assumption (documented, not silently hidden): this grid mapping
assumes the LR and HR arrays are already pixel-grid-aligned, i.e. LR pixel
(0, 0) and HR pixel (0, 0) cover the same ground location, and every LR
pixel corresponds to exactly `scale_factor` x `scale_factor` HR pixels with
no fractional offset. This holds exactly for the synthetic HR fallback
(the synthetic LR is generated directly from the HR array by downsampling,
so both share the same origin by construction). For real HR references
that have been reprojected onto the LR's CRS, this alignment is only as
good as the reprojection - true sub-pixel registration accuracy is a
Phase 6 (Validation) concern, not solved here. See
src/preprocessing/README.md and data/README.md for related caveats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def compute_patch_grid(height: int, width: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    """Return a deterministic, row-major list of (row, col) top-left patch positions.

    Positions are chosen so every patch fits fully within [0, height) x [0, width);
    any remainder that doesn't fit a full patch is dropped (not padded), which
    keeps every emitted patch exactly patch_size x patch_size.
    """
    if patch_size <= 0 or stride <= 0:
        return []

    positions: List[Tuple[int, int]] = []
    row = 0
    while row + patch_size <= height:
        col = 0
        while col + patch_size <= width:
            positions.append((row, col))
            col += stride
        row += stride
    return positions


def nodata_fraction(array: np.ndarray, nodata_value: Optional[float]) -> float:
    """Fraction of pixels equal to nodata_value, across all bands.

    Returns 0.0 if nodata_value is None (no nodata value defined for this
    raster) - the caller should treat that as "cannot check" rather than
    "definitely no nodata", since a None here just means Phase 1 didn't
    find a nodata value declared in the file's metadata.
    """
    if nodata_value is None or array.size == 0:
        return 0.0
    return float(np.count_nonzero(array == nodata_value)) / float(array.size)


@dataclass
class PatchExtractionResult:
    ok: bool
    lr_patch: Optional[np.ndarray]
    hr_patch: Optional[np.ndarray]
    lr_row: int
    lr_col: int
    hr_row: int
    hr_col: int
    lr_nodata_fraction: float
    hr_nodata_fraction: float
    issues: List[str] = field(default_factory=list)


def extract_patch_pair(
    lr_array: np.ndarray,
    hr_array: np.ndarray,
    scale_factor: int,
    hr_row: int,
    hr_col: int,
    patch_size: int,
    max_nodata_fraction: float,
    lr_nodata: Optional[float] = None,
    hr_nodata: Optional[float] = None,
) -> PatchExtractionResult:
    """Crop one LR/HR patch pair at the given HR-grid position.

    lr_array, hr_array: shape (bands, height, width). Band count must match
        between the two (checked by the caller before calling this in a
        loop, but re-checked here defensively).
    scale_factor: LR pixels are `scale_factor` times larger than HR pixels
        (e.g. 4 for 10m LR vs ~2.5m-equivalent synthetic/real HR).
    """
    if lr_array.shape[0] != hr_array.shape[0]:
        return PatchExtractionResult(
            False, None, None, 0, 0, hr_row, hr_col, 0.0, 0.0,
            [f"Band count mismatch: LR has {lr_array.shape[0]}, HR has {hr_array.shape[0]}"],
        )

    lr_size = patch_size // scale_factor
    if lr_size <= 0:
        return PatchExtractionResult(
            False, None, None, 0, 0, hr_row, hr_col, 0.0, 0.0,
            [f"patch_size {patch_size} is too small for scale_factor {scale_factor} (LR patch size would be 0)"],
        )

    lr_row, lr_col = hr_row // scale_factor, hr_col // scale_factor

    _, hr_h, hr_w = hr_array.shape
    _, lr_h, lr_w = lr_array.shape

    if hr_row + patch_size > hr_h or hr_col + patch_size > hr_w:
        return PatchExtractionResult(
            False, None, None, lr_row, lr_col, hr_row, hr_col, 0.0, 0.0,
            ["HR patch window falls outside the HR array bounds"],
        )
    if lr_row + lr_size > lr_h or lr_col + lr_size > lr_w:
        return PatchExtractionResult(
            False, None, None, lr_row, lr_col, hr_row, hr_col, 0.0, 0.0,
            ["Corresponding LR patch window falls outside the LR array bounds"],
        )

    hr_patch = hr_array[:, hr_row : hr_row + patch_size, hr_col : hr_col + patch_size]
    lr_patch = lr_array[:, lr_row : lr_row + lr_size, lr_col : lr_col + lr_size]

    hr_nd = nodata_fraction(hr_patch, hr_nodata)
    lr_nd = nodata_fraction(lr_patch, lr_nodata)

    issues: List[str] = []
    if hr_nd > max_nodata_fraction:
        issues.append(f"HR patch nodata fraction {hr_nd:.1%} exceeds max {max_nodata_fraction:.1%}")
    if lr_nd > max_nodata_fraction:
        issues.append(f"LR patch nodata fraction {lr_nd:.1%} exceeds max {max_nodata_fraction:.1%}")

    if issues:
        return PatchExtractionResult(False, None, None, lr_row, lr_col, hr_row, hr_col, lr_nd, hr_nd, issues)

    return PatchExtractionResult(True, lr_patch, hr_patch, lr_row, lr_col, hr_row, hr_col, lr_nd, hr_nd, [])