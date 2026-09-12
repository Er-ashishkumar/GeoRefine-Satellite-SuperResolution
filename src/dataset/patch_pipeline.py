"""
Phase 3 orchestration: turn Phase 2's preprocessing_index.json into on-disk
LR/HR patch pairs plus data/processed/patches_index.json.

Design decision (documented here for later phases, per the same convention
as src/preprocessing/pipeline.py's docstring):

Phase 2's preprocessing_index.json describes, per scene, either:
  - a real HR reference with an alignment *report* (check_alignment()), or
  - a synthetic LR/proxy-HR pair with both rasters already written to disk.

Phase 2 does not materialize an aligned real-HR raster on disk - only the
alignment report exists for that path. Rather than reopening Phase 2 (an
already-committed, tested module) to add that output, Phase 3 calls
src.preprocessing.alignment.reproject_hr_to_lr_grid() itself, lazily, at
patch-generation time, only for real-HR scenes it is actually about to crop.
This keeps Phase 2's output contract unchanged and avoids Phase 3 silently
depending on an interface Phase 2 never promised.

For synthetic-HR scenes, no reprojection is needed: the synthetic LR was
generated directly from the proxy-HR array by downsampling (see
src/preprocessing/synthetic_hr.py), so the two arrays are pixel-grid-aligned
by construction.

Every written patch keeps a real CRS and geotransform (derived from its
source raster, shifted to the patch's row/col offset) rather than being
written as a bare pixel array - this keeps patches independently inspectable
in GIS tools and preserves resolution information at the patch level,
consistent with the project's raster-first conventions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio import Affine

from src.dataset.scene import read_raster_metadata
from src.preprocessing.alignment import reproject_hr_to_lr_grid
from src.preprocessing.bands import load_band_stack

from .patches import compute_patch_grid, extract_patch_pair


def _read_full_array(path: str) -> Tuple[np.ndarray, Optional[str], tuple, Optional[float]]:
    with rasterio.open(path) as src:
        return (
            src.read(),
            src.crs.to_string() if src.crs else None,
            tuple(src.transform)[:6],
            src.nodata,
        )


PairResult = Tuple[
    np.ndarray, np.ndarray, int, Optional[float], Optional[float],
    Optional[str], tuple, Optional[str], tuple,
]
# (hr_array, lr_array, scale_factor, hr_nodata, lr_nodata,
#  hr_crs, hr_transform, lr_crs, lr_transform)


def _load_pair_for_synthetic_scene(scene: Dict[str, Any]) -> Optional[PairResult]:
    """Returns the PairResult tuple, or None if this scene has no synthetic data."""
    synth = scene.get("synthetic")
    if not synth:
        return None

    proxy_paths = synth["proxy_hr_paths"]
    hr_bands = []
    hr_nodata = None
    hr_crs = None
    hr_transform = None
    for i, p in enumerate(proxy_paths):
        data, crs, transform, nodata = _read_full_array(p)
        hr_bands.append(data[0] if data.shape[0] == 1 else data)
        hr_nodata = nodata if hr_nodata is None else hr_nodata
        if i == 0:
            hr_crs, hr_transform = crs, transform
    hr_array = np.stack(hr_bands, axis=0) if len(hr_bands) > 1 else hr_bands[0]
    if hr_array.ndim == 2:
        hr_array = hr_array[np.newaxis, ...]

    lr_array, lr_crs, lr_transform, lr_nodata = _read_full_array(synth["synthetic_lr_path"])

    return (
        hr_array, lr_array, synth["downsample_factor"], hr_nodata, lr_nodata,
        hr_crs, hr_transform, lr_crs, lr_transform,
    )


def _load_pair_for_real_hr_scene(
    scene: Dict[str, Any],
    dataset_scene: Dict[str, Any],
    config: Dict[str, Any],
    scratch_dir: Path,
) -> Optional[PairResult]:
    """Returns the PairResult tuple, or None if this scene has no usable aligned HR.

    Only processes scenes whose alignment report says is_aligned for at
    least one HR file - picks the first aligned HR file, reprojects it onto
    the LR grid on the fly (see module docstring), then loads the LR band
    stack via the same logic Phase 2 uses.
    """
    aligned_entries = [a for a in scene.get("alignment", []) if a["is_aligned"]]
    if not aligned_entries:
        return None

    hr_path = aligned_entries[0]["hr_file"]

    lr_metas = [read_raster_metadata(f["path"]) for f in dataset_scene["lr_files"]]
    sentinel2_cfg = config.get("sentinel2", {})
    requested_bands = {"red": sentinel2_cfg.get("red_band", "B4"), "nir": sentinel2_cfg.get("nir_band", "B8")}
    band_stack = load_band_stack(lr_metas, requested_bands)
    if not band_stack.ok:
        return None

    lr_representative_meta = lr_metas[0]
    scratch_dir.mkdir(parents=True, exist_ok=True)
    aligned_hr_path = scratch_dir / f"{scene['scene_id']}_aligned_hr.tif"
    reproject_hr_to_lr_grid(hr_path, lr_representative_meta, str(aligned_hr_path))

    hr_array, hr_crs, hr_transform, hr_nodata = _read_full_array(str(aligned_hr_path))
    scale_factor = aligned_entries[0]["resolution_ratio"]
    if scale_factor is None:
        return None
    scale_factor = int(round(scale_factor))

    lr_crs = band_stack.crs
    lr_transform = band_stack.transform

    return (
        hr_array, band_stack.data, scale_factor, hr_nodata, lr_representative_meta.nodata,
        hr_crs, hr_transform, lr_crs, lr_transform,
    )


def _write_patch(
    path: Path,
    array: np.ndarray,
    nodata: Optional[float],
    crs: Optional[str],
    source_transform_6: tuple,
    row_offset: int,
    col_offset: int,
) -> None:
    """Write one patch as a standalone GeoTIFF with its own real CRS and transform.

    The patch's transform is the source raster's transform shifted by
    (row_offset, col_offset) pixels, so the patch correctly reports its true
    ground location and pixel size rather than an identity/no-CRS transform.
    """
    bands, height, width = array.shape
    src_transform = Affine(*source_transform_6)
    patch_transform = src_transform @ Affine.translation(col_offset, row_offset)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=bands,
        dtype=str(array.dtype), nodata=nodata,
        crs=crs, transform=patch_transform,
    ) as dst:
        dst.write(array)


def _process_scene(
    scene: Dict[str, Any],
    dataset_scenes_by_id: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    scratch_dir: Path,
) -> Dict[str, Any]:
    scene_id = scene["scene_id"]
    patches_cfg = config.get("patches", {})
    patch_size = patches_cfg.get("size", 128)
    stride = patches_cfg.get("stride", 128)
    max_nodata = patches_cfg.get("max_nodata_fraction", 0.1)

    result: Dict[str, Any] = {
        "scene_id": scene_id,
        "is_synthetic_hr": scene.get("is_synthetic_hr", False),
        "num_patches": 0,
        "num_rejected": 0,
        "patches": [],
        "issues": [],
    }

    if scene.get("is_synthetic_hr"):
        pair = _load_pair_for_synthetic_scene(scene)
    else:
        dataset_scene = dataset_scenes_by_id.get(scene_id)
        if dataset_scene is None:
            result["issues"].append(f"No matching entry for '{scene_id}' in dataset_index.json")
            return result
        pair = _load_pair_for_real_hr_scene(scene, dataset_scene, config, scratch_dir)

    if pair is None:
        result["issues"].append("Could not load an LR/HR array pair for this scene (no aligned or synthetic data)")
        return result

    (
        hr_array, lr_array, scale_factor, hr_nodata, lr_nodata,
        hr_crs, hr_transform, lr_crs, lr_transform,
    ) = pair
    _, hr_h, hr_w = hr_array.shape

    grid = compute_patch_grid(hr_h, hr_w, patch_size, stride)
    if not grid:
        result["issues"].append(
            f"No patch positions fit within HR array of shape {hr_array.shape} "
            f"for patch_size={patch_size}"
        )
        return result

    scene_out_dir = output_dir / scene_id
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (hr_row, hr_col) in enumerate(grid):
        extraction = extract_patch_pair(
            lr_array, hr_array, scale_factor, hr_row, hr_col, patch_size,
            max_nodata_fraction=max_nodata, lr_nodata=lr_nodata, hr_nodata=hr_nodata,
        )
        if not extraction.ok:
            result["num_rejected"] += 1
            continue

        lr_path = scene_out_dir / f"patch_{idx:04d}_lr.tif"
        hr_path = scene_out_dir / f"patch_{idx:04d}_hr.tif"

        _write_patch(
            lr_path, extraction.lr_patch, lr_nodata,
            lr_crs, lr_transform, extraction.lr_row, extraction.lr_col,
        )
        _write_patch(
            hr_path, extraction.hr_patch, hr_nodata,
            hr_crs, hr_transform, extraction.hr_row, extraction.hr_col,
        )

        result["num_patches"] += 1
        result["patches"].append(
            {
                "index": idx,
                "lr_path": str(lr_path),
                "hr_path": str(hr_path),
                "hr_row": extraction.hr_row,
                "hr_col": extraction.hr_col,
                "lr_row": extraction.lr_row,
                "lr_col": extraction.lr_col,
                "lr_nodata_fraction": extraction.lr_nodata_fraction,
                "hr_nodata_fraction": extraction.hr_nodata_fraction,
            }
        )

    return result


def build_patches_index(
    preprocessing_index_path: Path,
    dataset_index_path: Path,
    config: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """Read preprocessing_index.json + dataset_index.json, write patches to disk
    and a patches_index.json summarizing every generated pair.
    """
    with open(preprocessing_index_path, "r", encoding="utf-8") as f:
        preprocessing_index = json.load(f)
    with open(dataset_index_path, "r", encoding="utf-8") as f:
        dataset_index = json.load(f)

    dataset_scenes_by_id = {s["scene_id"]: s for s in dataset_index.get("scenes", [])}

    processed_dir = Path(config["paths"]["data_processed"])
    patches_cfg = config.get("patches", {})
    output_subdir = patches_cfg.get("output_subdir", "patches")
    output_dir = processed_dir / output_subdir
    scratch_dir = processed_dir / "_patch_scratch"

    scenes_out: List[Dict[str, Any]] = [
        _process_scene(scene, dataset_scenes_by_id, config, output_dir, scratch_dir)
        for scene in preprocessing_index.get("scenes", [])
    ]

    index = {
        "num_scenes": len(scenes_out),
        "num_total_patches": sum(s["num_patches"] for s in scenes_out),
        "num_total_rejected": sum(s["num_rejected"] for s in scenes_out),
        "num_scenes_with_issues": sum(1 for s in scenes_out if s["issues"]),
        "scenes": scenes_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return index