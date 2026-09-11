"""
Tests for src.preprocessing - band merging, alignment checks, and the
synthetic HR fallback degradation pipeline.

Like tests/test_dataset.py, these never touch real satellite imagery: they
write small synthetic GeoTIFFs to a temporary directory and run the real
Phase 2 code against them, including a full Phase 1 -> Phase 2 integration
test that runs the real Phase 1 indexer first.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset.registry import build_index, build_scene_record
from src.preprocessing.alignment import check_alignment, reproject_hr_to_lr_grid
from src.preprocessing.bands import canonical_band_name, load_band_stack, match_band_file
from src.preprocessing.pipeline import build_preprocessing_index
from src.preprocessing.synthetic_hr import degrade_to_synthetic_lr


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
# bands.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename,expected",
    [
        ("B04.tif", "B4"),
        ("B4.tif", "B4"),
        ("scene_B08_10m.tif", "B8"),
        ("B8A.tif", "B8A"),
        ("no_band_here.tif", None),
    ],
)
def test_canonical_band_name(filename, expected):
    assert canonical_band_name(filename) == expected


def test_match_band_file_unambiguous(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif")
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif")
    record = build_scene_record(raw_dir, "sceneA")

    match = match_band_file(record.lr_files, "B4")
    assert match is not None
    assert "B04" in match.path


def test_load_band_stack_from_separate_files(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=16, height=16)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=16, height=16)
    record = build_scene_record(raw_dir, "sceneA")

    result = load_band_stack(record.lr_files, {"red": "B4", "nir": "B8"})

    assert result.ok
    assert result.data.shape == (2, 16, 16)
    assert result.band_order == ["red", "nir"]


def test_load_band_stack_flags_mismatched_grid(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=16, height=16)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=32, height=32)
    record = build_scene_record(raw_dir, "sceneA")

    result = load_band_stack(record.lr_files, {"red": "B4", "nir": "B8"})

    assert not result.ok
    assert any("Size mismatch" in issue for issue in result.issues)


def test_load_band_stack_flags_missing_band(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif")
    record = build_scene_record(raw_dir, "sceneA")

    result = load_band_stack(record.lr_files, {"red": "B4", "nir": "B8"})

    assert not result.ok
    assert any("Could not unambiguously find" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# alignment.py
# ---------------------------------------------------------------------------

def test_check_alignment_matching_pair(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "lr.tif", width=32, height=32, res=4.0, origin=(0.0, 128.0))
    _write_fake_geotiff(raw_dir / "hr.tif", width=128, height=128, res=1.0, origin=(0.0, 128.0))
    record = build_scene_record(tmp_path, "raw")  # not used for scanning; read directly instead

    from src.dataset.scene import read_raster_metadata

    lr_meta = read_raster_metadata(raw_dir / "lr.tif")
    hr_meta = read_raster_metadata(raw_dir / "hr.tif")

    report = check_alignment(lr_meta, hr_meta, expected_scale_factor=4)

    assert report.crs_match
    assert report.is_aligned
    assert report.bounds_overlap_fraction == pytest.approx(1.0)
    assert report.resolution_ratio == pytest.approx(4.0)
    assert report.issues == []


def test_check_alignment_flags_low_overlap(tmp_path):
    from src.dataset.scene import read_raster_metadata

    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "lr.tif", width=32, height=32, res=1.0, origin=(0.0, 32.0))
    _write_fake_geotiff(raw_dir / "hr.tif", width=32, height=32, res=1.0, origin=(1000.0, 1032.0))

    lr_meta = read_raster_metadata(raw_dir / "lr.tif")
    hr_meta = read_raster_metadata(raw_dir / "hr.tif")

    report = check_alignment(lr_meta, hr_meta)

    assert not report.is_aligned
    assert report.bounds_overlap_fraction == 0.0
    assert any("overlap" in issue for issue in report.issues)


def test_reproject_hr_to_lr_grid_writes_valid_raster(tmp_path):
    from src.dataset.scene import read_raster_metadata

    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "lr.tif", width=16, height=16, res=4.0, origin=(0.0, 64.0), crs="EPSG:4326")
    _write_fake_geotiff(raw_dir / "hr.tif", width=64, height=64, res=1.0, origin=(0.0, 64.0), crs="EPSG:3857")

    lr_meta = read_raster_metadata(raw_dir / "lr.tif")
    out_path = tmp_path / "aligned_hr.tif"

    reproject_hr_to_lr_grid(str(raw_dir / "hr.tif"), lr_meta, str(out_path))

    assert out_path.exists()
    with rasterio.open(out_path) as src:
        assert src.crs.to_string() == lr_meta.crs


# ---------------------------------------------------------------------------
# synthetic_hr.py
# ---------------------------------------------------------------------------

def test_degrade_to_synthetic_lr_shape_and_smoothing():
    rng = np.random.default_rng(0)
    proxy_hr = rng.integers(0, 255, size=(2, 64, 64)).astype("float64")

    synthetic_lr = degrade_to_synthetic_lr(proxy_hr, blur_sigma=1.0, downsample_factor=4)

    assert synthetic_lr.shape == (2, 16, 16)
    # Blurring + averaging should reduce high-frequency variance relative to the source.
    assert synthetic_lr.std() < proxy_hr.std()


def test_degrade_to_synthetic_lr_noop_with_factor_one():
    rng = np.random.default_rng(0)
    proxy_hr = rng.integers(0, 255, size=(2, 16, 16)).astype("float64")

    synthetic_lr = degrade_to_synthetic_lr(proxy_hr, blur_sigma=0.0, downsample_factor=1)

    np.testing.assert_array_equal(synthetic_lr, proxy_hr)


# ---------------------------------------------------------------------------
# pipeline.py - full Phase 1 -> Phase 2 integration
# ---------------------------------------------------------------------------

def test_build_preprocessing_index_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    # Scene A: no HR -> synthetic fallback path
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif", width=32, height=32)
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif", width=32, height=32)

    # Scene B: real HR reference, well-aligned -> alignment path
    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "B04.tif", width=16, height=16, res=4.0, origin=(0.0, 64.0))
    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "B08.tif", width=16, height=16, res=4.0, origin=(0.0, 64.0))
    _write_fake_geotiff(raw_dir / "sceneB" / "hr" / "ref.tif", width=64, height=64, res=1.0, origin=(0.0, 64.0), count=3)

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "dataset": {
            "synthetic_hr_fallback": {
                "enabled": True, "blur_sigma": 1.0, "downsample_factor": 4,
            },
        },
        "preprocessing": {
            "synthetic_output_subdir": "synthetic_lr",
            "min_overlap_fraction": 0.5,
            "resolution_ratio_tolerance": 0.15,
        },
    }

    output_path = processed_dir / "preprocessing_index.json"
    index = build_preprocessing_index(dataset_index_path, config, output_path)

    assert index["num_scenes"] == 2
    assert index["num_scenes_synthetic_hr"] == 1
    assert index["num_scenes_with_real_hr_aligned"] == 1
    assert index["num_scenes_with_issues"] == 0
    assert output_path.exists()

    scenes_by_id = {s["scene_id"]: s for s in index["scenes"]}
    assert scenes_by_id["sceneA"]["is_synthetic_hr"] is True
    assert Path(scenes_by_id["sceneA"]["synthetic"]["synthetic_lr_path"]).exists()
    assert scenes_by_id["sceneB"]["is_synthetic_hr"] is False
    assert scenes_by_id["sceneB"]["alignment"][0]["is_aligned"] is True

    with open(output_path) as f:
        saved = json.load(f)
    assert saved["num_scenes"] == 2


def test_build_preprocessing_index_flags_no_hr_and_fallback_disabled(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B04.tif")
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "B08.tif")

    dataset_index_path = processed_dir / "dataset_index.json"
    build_index(raw_dir, dataset_index_path)

    config = {
        "paths": {"data_raw": str(raw_dir), "data_processed": str(processed_dir)},
        "sentinel2": {"red_band": "B4", "nir_band": "B8"},
        "dataset": {"synthetic_hr_fallback": {"enabled": False}},
        "preprocessing": {},
    }

    output_path = processed_dir / "preprocessing_index.json"
    index = build_preprocessing_index(dataset_index_path, config, output_path)

    assert index["num_scenes_with_issues"] == 1
    scene = index["scenes"][0]
    assert scene["is_synthetic_hr"] is False
    assert any("synthetic_hr_fallback.enabled is false" in issue for issue in scene["issues"])
