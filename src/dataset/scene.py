"""
Scene-level metadata utilities for GeoRefine.

A "scene" is a single satellite acquisition, represented as one or more
raster files on disk (see data/README.md for the expected folder layout).
This module only reads metadata (CRS, transform, size, band count) - it
does not load pixel data into memory. That keeps dataset indexing fast and
safe to run on large collections of imagery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import rasterio


@dataclass
class RasterMetadata:
    path: str
    crs: Optional[str]
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]
    resolution_x: float
    resolution_y: float
    bounds: Tuple[float, float, float, float]

    @property
    def is_valid(self) -> bool:
        return bool(self.crs) and self.width > 0 and self.height > 0 and self.count > 0


def read_raster_metadata(path: Path) -> RasterMetadata:
    """Read raster metadata for a single file without loading pixel data."""
    with rasterio.open(path) as src:
        return RasterMetadata(
            path=str(path),
            crs=src.crs.to_string() if src.crs else None,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=str(src.dtypes[0]) if src.dtypes else "unknown",
            nodata=src.nodata,
            resolution_x=abs(src.transform.a),
            resolution_y=abs(src.transform.e),
            bounds=tuple(src.bounds),
        )


@dataclass
class SceneRecord:
    scene_id: str
    lr_files: List[RasterMetadata]
    hr_files: List[RasterMetadata]
    issues: List[str] = field(default_factory=list)

    @property
    def has_hr_reference(self) -> bool:
        return len(self.hr_files) > 0

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "lr_files": [vars(m) for m in self.lr_files],
            "hr_files": [vars(m) for m in self.hr_files],
            "has_hr_reference": self.has_hr_reference,
            "issues": self.issues,
        }