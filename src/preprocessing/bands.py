"""
Band identification and merging for GeoRefine.

Phase 1 (src/dataset) records metadata for every raster file under a scene's
lr/ and hr/ folders, but does not interpret *which* spectral band each file
represents, and does not load pixel data. Phase 2 does both: it identifies
Sentinel-2-style band tokens (B4, B04, B8, B08, B8A, ...) in filenames and
loads matching files into an aligned, stacked array so downstream code
(NDVI, patch generation, the model) can work with band-indexed arrays
instead of per-file bookkeeping.

Two input shapes are supported for a scene's lr/ files:

1. Multiple single-band files (e.g. B04.tif, B08.tif) - the common case for
   raw Sentinel-2 downloads. Each file is matched to a requested band name
   by filename, then stacked.
2. A single pre-merged multi-band composite file. It is read as-is; Phase 2
   cannot infer which band index corresponds to which spectral band without
   additional metadata, so callers must know the band order for composite
   files out-of-band (e.g. via a naming convention documented in
   data/README.md). This module still returns it, flagged as "unknown"
   band order, rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from src.dataset.scene import RasterMetadata

# Matches Sentinel-2 style band tokens: B4, B04, B8, B08, B8A, B11, B12, ...
_BAND_PATTERN = re.compile(r"B(\d{1,2})(A)?", re.IGNORECASE)


def canonical_band_name(text: str) -> Optional[str]:
    """Extract and normalize a Sentinel-2 band token from a filename or string.

    Normalizes zero-padding so "B04" and "B4" both become "B4"; preserves
    the "A" suffix on B8A. Returns None if no band token is found.
    """
    match = _BAND_PATTERN.search(text)
    if not match:
        return None
    number = int(match.group(1))
    suffix = "A" if match.group(2) else ""
    return f"B{number}{suffix}"


def match_band_file(
    files: List[RasterMetadata], band_name: str
) -> Optional[RasterMetadata]:
    """Return the single-band file whose filename matches band_name, if any."""
    target = canonical_band_name(band_name) or band_name.upper()
    matches = [f for f in files if canonical_band_name(Path(f.path).name) == target]
    if len(matches) == 1:
        return matches[0]
    return None  # 0 or >1 matches: ambiguous, caller should treat as an issue


@dataclass
class BandStackResult:
    data: Optional[np.ndarray]  # shape (bands, height, width), or None if it could not be built
    band_order: List[str]  # logical names from requested_bands, e.g. ["red", "nir"] (or ["unknown"] * count for a pre-merged composite)
    crs: Optional[str]
    transform: Optional[Tuple[float, float, float, float, float, float]]
    source_files: List[str]
    issues: List[str]

    @property
    def ok(self) -> bool:
        return self.data is not None and not self.issues


def _read_full(path: str) -> Tuple[np.ndarray, str, tuple]:
    with rasterio.open(path) as src:
        return src.read(), (src.crs.to_string() if src.crs else None), tuple(src.transform)[:6]


def load_band_stack(
    lr_files: List[RasterMetadata],
    requested_bands: Dict[str, str],
) -> BandStackResult:
    """Build an aligned, stacked array for the requested spectral bands.

    Args:
        lr_files: RasterMetadata entries for one scene's lr/ files (from the
            Phase 1 dataset index).
        requested_bands: e.g. {"red": "B4", "nir": "B8"} - logical name to
            Sentinel-2 band token, taken from config.yaml's `sentinel2` section.

    Returns:
        A BandStackResult. `issues` is non-empty (and `data` is None) if the
        requested bands could not be unambiguously found and stacked - this
        is a Phase-2-level, pixel/metadata-consistency issue distinct from
        Phase 1's per-file header issues.
    """
    issues: List[str] = []

    if not lr_files:
        return BandStackResult(None, [], None, None, [], ["No LR files provided"])

    # Case 1: a single, already multi-band composite file.
    if len(lr_files) == 1 and lr_files[0].count >= len(requested_bands):
        meta = lr_files[0]
        data, crs, transform = _read_full(meta.path)
        return BandStackResult(
            data=data,
            band_order=["unknown"] * data.shape[0],
            crs=crs,
            transform=transform,
            source_files=[meta.path],
            issues=[
                "Single multi-band composite file: band order could not be "
                "verified against config's requested band names (red/nir/...). "
                "Confirm band order out-of-band before using this stack for NDVI."
            ],
        )

    # Case 2: multiple single-band files - match each requested band by filename.
    matched: Dict[str, RasterMetadata] = {}
    for logical_name, band_token in requested_bands.items():
        found = match_band_file(lr_files, band_token)
        if found is None:
            issues.append(
                f"Could not unambiguously find a file for band '{band_token}' "
                f"(logical name '{logical_name}') among {[Path(f.path).name for f in lr_files]}"
            )
        else:
            matched[logical_name] = found

    if issues:
        return BandStackResult(None, [], None, None, [], issues)

    # Verify all matched files share the same grid before stacking.
    metas = list(matched.values())
    ref = metas[0]
    for m in metas[1:]:
        if m.crs != ref.crs:
            issues.append(f"CRS mismatch between {Path(ref.path).name} ({ref.crs}) and {Path(m.path).name} ({m.crs})")
        if (m.width, m.height) != (ref.width, ref.height):
            issues.append(
                f"Size mismatch between {Path(ref.path).name} ({ref.width}x{ref.height}) "
                f"and {Path(m.path).name} ({m.width}x{m.height})"
            )
        if (round(m.resolution_x, 6), round(m.resolution_y, 6)) != (
            round(ref.resolution_x, 6),
            round(ref.resolution_y, 6),
        ):
            issues.append(
                f"Resolution mismatch between {Path(ref.path).name} "
                f"({ref.resolution_x},{ref.resolution_y}) and {Path(m.path).name} "
                f"({m.resolution_x},{m.resolution_y})"
            )

    if issues:
        return BandStackResult(None, [], None, None, [m.path for m in metas], issues)

    band_order = list(matched.keys())
    arrays = []
    transform = None
    crs = None
    for logical_name in band_order:
        data, crs, transform = _read_full(matched[logical_name].path)
        arrays.append(data[0])  # single-band file: first band

    stacked = np.stack(arrays, axis=0)
    return BandStackResult(
        data=stacked,
        band_order=band_order,
        crs=crs,
        transform=transform,
        source_files=[matched[n].path for n in band_order],
        issues=[],
    )
