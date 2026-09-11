"""
Co-registration and alignment checks between LR and real HR reference imagery.

This only applies to scenes that have a genuine, independently-sourced HR
reference under data/raw/<scene_id>/hr/ (see data/README.md). Scenes without
a real HR reference use the synthetic degradation fallback instead
(see synthetic_hr.py) and never go through this module.

Checks performed, all at the metadata + bounds level (no pixel data is
loaded here - that keeps this fast and lets pixel-level work in
generate-aligned-raster be an explicit, separate step):

- CRS match (reprojecting bounds for comparison when they differ, using
  pyproj, which is already a project dependency)
- resolution ratio vs. the expected LR:HR scale factor
- spatial bounds overlap fraction

reproject_hr_to_lr_grid performs the actual pixel-level alignment (via
rasterio.warp) once a pair has been checked and is worth aligning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from pyproj import Transformer

from src.dataset.scene import RasterMetadata


@dataclass
class AlignmentReport:
    crs_match: bool
    resolution_ratio: Optional[float]  # lr_resolution / hr_resolution
    expected_ratio: Optional[float]
    bounds_overlap_fraction: float  # 0.0-1.0, fraction of the smaller footprint covered by the overlap
    is_aligned: bool
    issues: List[str] = field(default_factory=list)


def _reproject_bounds(
    bounds: Tuple[float, float, float, float], src_crs: str, dst_crs: str
) -> Tuple[float, float, float, float]:
    """Reproject a (left, bottom, right, top) bounding box's corners into dst_crs."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    left, bottom, right, top = bounds
    xs, ys = transformer.transform([left, right, left, right], [bottom, bottom, top, top])
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_fraction(
    bounds_a: Tuple[float, float, float, float], bounds_b: Tuple[float, float, float, float]
) -> float:
    ax0, ay0, ax1, ay1 = bounds_a
    bx0, by0, bx1, by1 = bounds_b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    intersection_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    smaller_area = min(area_a, area_b)
    if smaller_area <= 0:
        return 0.0
    return intersection_area / smaller_area


def check_alignment(
    lr_meta: RasterMetadata,
    hr_meta: RasterMetadata,
    expected_scale_factor: Optional[float] = None,
    min_overlap_fraction: float = 0.5,
    ratio_tolerance: float = 0.15,
) -> AlignmentReport:
    """Compare one LR raster and one real HR raster for co-registration fitness.

    expected_scale_factor: the LR:HR pixel-size ratio you expect (e.g. 4 for
        10m LR vs ~2.5m HR). If given, a large deviation is flagged as an issue
        but does not by itself fail is_aligned - resolution mismatches are
        common and are informational for the alignment/resampling step.
    min_overlap_fraction: below this, the pair is considered not usable
        together and is_aligned is False.
    """
    issues: List[str] = []

    crs_match = lr_meta.crs == hr_meta.crs
    if not lr_meta.crs or not hr_meta.crs:
        issues.append("Missing CRS on LR and/or HR raster; cannot reliably check alignment")
        return AlignmentReport(
            crs_match=False,
            resolution_ratio=None,
            expected_ratio=expected_scale_factor,
            bounds_overlap_fraction=0.0,
            is_aligned=False,
            issues=issues,
        )

    hr_bounds_in_lr_crs = hr_meta.bounds
    if not crs_match:
        try:
            hr_bounds_in_lr_crs = _reproject_bounds(hr_meta.bounds, hr_meta.crs, lr_meta.crs)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Failed to reproject HR bounds into LR CRS for comparison: {exc}")
            return AlignmentReport(
                crs_match=False,
                resolution_ratio=None,
                expected_ratio=expected_scale_factor,
                bounds_overlap_fraction=0.0,
                is_aligned=False,
                issues=issues,
            )

    overlap = _overlap_fraction(lr_meta.bounds, hr_bounds_in_lr_crs)
    if overlap < min_overlap_fraction:
        issues.append(
            f"Low spatial overlap between LR and HR footprints ({overlap:.1%}, "
            f"expected at least {min_overlap_fraction:.0%})"
        )

    resolution_ratio = None
    if hr_meta.resolution_x > 0:
        resolution_ratio = lr_meta.resolution_x / hr_meta.resolution_x
        if expected_scale_factor:
            deviation = abs(resolution_ratio - expected_scale_factor) / expected_scale_factor
            if deviation > ratio_tolerance:
                issues.append(
                    f"LR:HR resolution ratio {resolution_ratio:.2f} deviates from expected "
                    f"{expected_scale_factor:.2f} by {deviation:.0%} (tolerance {ratio_tolerance:.0%})"
                )

    if not crs_match:
        issues.append(f"CRS mismatch: LR={lr_meta.crs}, HR={hr_meta.crs} (bounds were reprojected for comparison)")

    is_aligned = overlap >= min_overlap_fraction

    return AlignmentReport(
        crs_match=crs_match,
        resolution_ratio=resolution_ratio,
        expected_ratio=expected_scale_factor,
        bounds_overlap_fraction=overlap,
        is_aligned=is_aligned,
        issues=issues,
    )


def reproject_hr_to_lr_grid(hr_path: str, lr_meta: RasterMetadata, out_path: str) -> str:
    """Resample/reproject an HR raster onto the LR raster's CRS, writing a new file.

    This does not change the HR raster's resolution - only its CRS/grid
    alignment - so the output remains a valid, higher-resolution reference
    that is now directly comparable pixel-for-pixel-region with the LR scene.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(hr_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, lr_meta.crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": lr_meta.crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=lr_meta.crs,
                    resampling=Resampling.bilinear,
                )

    return out_path
