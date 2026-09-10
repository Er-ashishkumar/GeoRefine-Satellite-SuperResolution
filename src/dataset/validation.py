"""
Lightweight, metadata-only sanity checks for raw imagery.

These checks run during dataset indexing (Phase 1) and are intentionally
cheap - they inspect raster headers, not pixel values. Deeper checks
(nodata fraction, co-registration accuracy, radiometric sanity) belong to
Phase 2 (preprocessing/alignment) and Phase 6 (validation/metrics), where
pixel data is already being loaded for other reasons.
"""
from __future__ import annotations

from typing import List

from .scene import RasterMetadata


def validate_raster_metadata(meta: RasterMetadata) -> List[str]:
    """Return a list of human-readable issues found in a raster's metadata.

    An empty list means no issues were found at the metadata level.
    """
    issues: List[str] = []

    if not meta.crs:
        issues.append("missing CRS (coordinate reference system)")

    if meta.width <= 0 or meta.height <= 0:
        issues.append(f"invalid raster size ({meta.width}x{meta.height})")

    if meta.count <= 0:
        issues.append("no bands found")

    if meta.resolution_x <= 0 or meta.resolution_y <= 0:
        issues.append(
            f"invalid pixel resolution ({meta.resolution_x}, {meta.resolution_y})"
        )

    return issues