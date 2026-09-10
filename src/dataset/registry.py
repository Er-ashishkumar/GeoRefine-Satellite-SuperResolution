"""
Dataset discovery and indexing.

Expected raw data layout (see data/README.md):

    data/raw/<scene_id>/lr/*.tif   (required - medium-resolution imagery)
    data/raw/<scene_id>/hr/*.tif   (optional - real high-resolution reference)

This module scans that layout, reads lightweight metadata for every raster
file found, runs basic sanity validation, and writes a single JSON index
(data/processed/dataset_index.json) that later phases (preprocessing,
patch generation) consume instead of re-scanning the filesystem.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from .scene import RasterMetadata, SceneRecord, read_raster_metadata
from .validation import validate_raster_metadata

DEFAULT_EXTENSIONS = (".tif", ".tiff")


def _list_raster_files(folder: Path, extensions: Sequence[str] = DEFAULT_EXTENSIONS) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def discover_scene_ids(raw_dir: Path, lr_subdir: str = "lr") -> List[str]:
    """Return sorted scene_ids: subfolders of raw_dir containing an lr/ folder."""
    if not raw_dir.exists():
        return []
    scene_ids = []
    for child in sorted(raw_dir.iterdir()):
        if child.is_dir() and (child / lr_subdir).exists():
            scene_ids.append(child.name)
    return scene_ids


def build_scene_record(
    raw_dir: Path,
    scene_id: str,
    lr_subdir: str = "lr",
    hr_subdir: str = "hr",
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> SceneRecord:
    scene_dir = raw_dir / scene_id
    lr_paths = _list_raster_files(scene_dir / lr_subdir, extensions)
    hr_paths = _list_raster_files(scene_dir / hr_subdir, extensions)

    issues: List[str] = []
    lr_meta: List[RasterMetadata] = []
    hr_meta: List[RasterMetadata] = []

    if not lr_paths:
        issues.append(f"No LR files found under {scene_dir / lr_subdir}")

    for p in lr_paths:
        try:
            meta = read_raster_metadata(p)
            lr_meta.append(meta)
            issues.extend(f"LR {p.name}: {msg}" for msg in validate_raster_metadata(meta))
        except Exception as exc:  # noqa: BLE001 - record any read failure, do not crash indexing
            issues.append(f"Failed to read LR file {p.name}: {exc}")

    for p in hr_paths:
        try:
            meta = read_raster_metadata(p)
            hr_meta.append(meta)
            issues.extend(f"HR {p.name}: {msg}" for msg in validate_raster_metadata(meta))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Failed to read HR file {p.name}: {exc}")

    return SceneRecord(scene_id=scene_id, lr_files=lr_meta, hr_files=hr_meta, issues=issues)


def build_index(
    raw_dir: Path,
    output_path: Path,
    lr_subdir: str = "lr",
    hr_subdir: str = "hr",
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> dict:
    """Scan raw_dir, build metadata for every scene, and write a JSON index.

    Returns the index dict that was written (also useful for tests/CLI summaries).
    """
    scene_ids = discover_scene_ids(raw_dir, lr_subdir=lr_subdir)
    records = [
        build_scene_record(raw_dir, sid, lr_subdir=lr_subdir, hr_subdir=hr_subdir, extensions=extensions)
        for sid in scene_ids
    ]

    index = {
        "num_scenes": len(records),
        "num_scenes_with_hr": sum(1 for r in records if r.has_hr_reference),
        "num_scenes_with_issues": sum(1 for r in records if r.issues),
        "scenes": [r.to_dict() for r in records],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return index
