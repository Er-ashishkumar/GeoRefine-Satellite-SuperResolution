"""
Tests for src.dataset.patches and src.dataset.patch_pipeline.

Like the Phase 1/2 test suites, these never touch real satellite imagery.
test_build_patches_index_end_to_end runs the real Phase 1 -> Phase 2 ->
Phase 3 pipeline against small synthetic GeoTIFFs, covering both the
synthetic-HR fallback path and the real-HR alignment path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset.patches import compute_patch_grid, extract_patch_pair, nodata_fraction
from src.dataset.patch_pipeline import build_patches_index
from src.dataset.registry import build_index
from src.preprocessing.pipeline import build_preprocessing_index


def _write_fake_geotiff(path: Path, width=32, height=32, count=1, crs="EPSG:4326", res=1.0, origin=(0.0, None)):
    path.parent.mkdir(parents=True, exist_ok=True)
    top = origin[1] if origin[1] is not None else height * res
    transform = from_origin(origin[0], top, res, res)
    data = (np.random.rand(count, height, width) * 200 + 10).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype="uint16", crs=crs, transform=transform,
    ) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# patches.py - pure logic
# ---------------------------------------------------------------------------

def test_compute_patch_grid_no_overlap():
    grid = compute_patch_grid(height=256, width=256, patch_size=128, stride=128)
    assert grid == [(0, 0), (0, 128), (128, 0), (128, 128)]


def test_compute_patch_grid_drops_remainder():
    grid = compute_patch_grid(height=130, width=130, patch_size=128, stride=128)
    assert grid == [(0, 0)]


def test_compute_patch_grid_empty_when_too_small():
    grid = compute_patch_grid(height=64, width=64, patch_size=128, stride=128)
    assert grid == []


def test_nodata_fraction_basic():
    array = np.array([[[0, 0], [1, 1]]])
    assert nodata_fraction(array, nodata_value=0) == pytest.approx(0.5)


def test_nodata_fraction_none_value_returns_zero():
    array = np.array([[[0, 0], [1, 1]]])
    assert nodata_fraction(array, nodata_value=None) == 0.0


def test_extract_patch_pair_success():
    hr = np.random.rand(2, 128, 128).astype("float32")
    lr = np.random.rand(2, 32, 32).astype("float32")

    result = extract_patch_pair(
        lr, hr, scale_factor=4, hr_row=0, hr_col=0, patch_size=128, max_nodata_fraction=0.1,
    )

    assert result.ok
    assert result.hr_patch.shape == (2, 128, 128)
    assert result.lr_patch.shape == (2, 32, 32)


def test_extract_patch_pair_rejects_high_nodata():
    hr = np.zeros((1, 128, 128), dtype="float32")  # entirely nodata
    lr = np.zeros((1, 32, 32), dtype="float32")

    result = extract_patch_pair(
        lr, hr, scale_factor=4, hr_row=0, hr_col=0, patch_size=128,
        max_nodata_fraction=0.1, hr_nodata=0.0, lr_nodata=0.0,
    )

    assert not result.ok
    assert any("nodata fraction" in issue for issue in result.issues)


def test_extract_patch_pair_out_of_bounds():
    hr = np.random.rand(1, 64, 64).astype("float32")
    lr = np.random.rand(1, 16, 16).astype("float32")

    result = extract_patch_pair(
        lr, hr, scale_factor=4, hr_row=32, hr_col=32, patch_size=128, max_nodata_fraction=0.1,
    )

    assert not result.ok
    assert any("outside" in issue for issue in result.issues)


def test_extract_patch_pair_band_mismatch():
    hr = np.random.rand(2, 128, 128).astype("float32")
    lr = np.random.rand(1, 32, 32).astype("float32")

    result = extract_patch_pair(
        lr, hr, scale_factor=4, hr_row=0, hr_col=0, patch_size=128, max_nodata_fraction=0.1,
    )

    assert not result.ok
    assert any("Band count mismatch" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# patch_pipeline.py - full Phase 1 -> 2 -> 3 integration
# ---------------------------------------------------------------------------

def test_build_patches_index_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    # Scene A: no HR -> synthetic fallback path (Phase 2), then patches (Phase 3)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=64, height=64)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=64, height=64)

    # Scene B: real, well-aligned HR reference -> alignment path (Phase 2), then patches (Phase 3)
    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "B04.tif", width=32, height=32, res=4.0, origin=(0.0, 128.0))
    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "B08.tif", width=32, height=32, res=4.0, origin=(0.0, 128.0))
    _write_fake_geotiff(raw_dir / "sceneB" / "hr" / "ref.tif", width=128, height=128, res=1.0, origin=(0.0, 128.0), count=2)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "dataset": {
            "synthetic_hr_fallback": {"enabled": True, "blur_sigma": 1.0, "downsample_factor": 4},
        },
        "preprocessing": {
            "synthetic_output_subdir": "synthetic_lr",
            "min_overlap_fraction": 0.5,
            "resolution_ratio_tolerance": 0.15,
        },
        "patches": {
            "size": 32,
            "stride": 32,
            "max_nodata_fraction": 0.5,
            "output_subdir": "patches",
        },
    }

    preprocessing_index_path = processed_dir / "preprocessing_index.json"
    build_preprocessing_index(dataset_index_path, config, preprocessing_index_path)

    output_path = processed_dir / "patches_index.json"
    index = build_patches_index(preprocessing_index_path, dataset_index_path, config, output_path)

    assert index["num_scenes"] == 2
    assert index["num_total_patches"] > 0
    assert output_path.exists()

    scenes_by_id = {s["scene_id"]: s for s in index["scenes"]}
    assert scenes_by_id["sceneA"]["is_synthetic_hr"] is True
    assert scenes_by_id["sceneA"]["num_patches"] > 0
    assert scenes_by_id["sceneB"]["is_synthetic_hr"] is False
    assert scenes_by_id["sceneB"]["num_patches"] > 0

    first_patch = scenes_by_id["sceneA"]["patches"][0]
    assert Path(first_patch["lr_path"]).exists()
    assert Path(first_patch["hr_path"]).exists()

    with open(output_path) as f:
        saved = json.load(f)
    assert saved["num_scenes"] == 2


def test_build_patches_index_handles_no_scenes(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "dataset": {"synthetic_hr_fallback": {"enabled": True}},
        "preprocessing": {},
        "patches": {"size": 128, "stride": 128, "max_nodata_fraction": 0.1},
    }

    preprocessing_index_path = processed_dir / "preprocessing_index.json"
    build_preprocessing_index(dataset_index_path, config, preprocessing_index_path)

    output_path = processed_dir / "patches_index.json"
    index = build_patches_index(preprocessing_index_path, dataset_index_path, config, output_path)

    assert index["num_scenes"] == 0
    assert index["num_total_patches"] == 0