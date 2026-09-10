"""
Tests for src.dataset - scene metadata reading, validation, and indexing.

These tests never touch real satellite imagery. They write small synthetic
GeoTIFFs to a temporary directory that follow the same
data/raw/<scene_id>/lr|hr/*.tif layout described in data/README.md, then
run the real indexing code against them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.dataset.registry import build_index, build_scene_record, discover_scene_ids
from src.dataset.scene import read_raster_metadata
from src.dataset.validation import validate_raster_metadata


def _write_fake_geotiff(path: Path, width=16, height=16, count=1, crs="EPSG:4326"):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0, height, 1, 1)
    data = np.random.randint(0, 255, size=(count, height, width), dtype="uint8")
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width,
        count=count, dtype="uint8",
        crs=crs, transform=transform,
    ) as dst:
        dst.write(data)


def test_read_raster_metadata(tmp_path):
    tif_path = tmp_path / "scene.tif"
    _write_fake_geotiff(tif_path, width=32, height=16, count=4)

    meta = read_raster_metadata(tif_path)

    assert meta.width == 32
    assert meta.height == 16
    assert meta.count == 4
    assert meta.crs is not None
    assert meta.is_valid


def test_validate_raster_metadata_flags_missing_crs(tmp_path):
    tif_path = tmp_path / "no_crs.tif"
    transform = from_origin(0, 16, 1, 1)
    data = np.zeros((1, 16, 16), dtype="uint8")
    with rasterio.open(
        tif_path, "w", driver="GTiff", height=16, width=16,
        count=1, dtype="uint8", transform=transform,
    ) as dst:
        dst.write(data)

    meta = read_raster_metadata(tif_path)
    issues = validate_raster_metadata(meta)

    assert any("CRS" in issue for issue in issues)


def test_discover_scene_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "b1.tif")
    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "b1.tif")
    (raw_dir / "sceneC" / "other").mkdir(parents=True)

    scene_ids = discover_scene_ids(raw_dir)

    assert scene_ids == ["sceneA", "sceneB"]


def test_build_scene_record_with_and_without_hr(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "b1.tif")
    _write_fake_geotiff(raw_dir / "sceneA" / "hr" / "ref.tif", width=64, height=64)

    record = build_scene_record(raw_dir, "sceneA")

    assert record.scene_id == "sceneA"
    assert len(record.lr_files) == 1
    assert len(record.hr_files) == 1
    assert record.has_hr_reference is True
    assert record.issues == []


def test_build_scene_record_flags_missing_lr(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "sceneEmpty" / "lr").mkdir(parents=True)

    record = build_scene_record(raw_dir, "sceneEmpty")

    assert record.lr_files == []
    assert any("No LR files found" in issue for issue in record.issues)


def test_build_index_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_fake_geotiff(raw_dir / "sceneA" / "lr" / "b1.tif")
    _write_fake_geotiff(raw_dir / "sceneA" / "hr" / "ref.tif", width=64, height=64)
    _write_fake_geotiff(raw_dir / "sceneB" / "lr" / "b1.tif")

    output_path = tmp_path / "processed" / "dataset_index.json"
    index = build_index(raw_dir, output_path)

    assert index["num_scenes"] == 2
    assert index["num_scenes_with_hr"] == 1
    assert output_path.exists()

    with open(output_path) as f:
        saved = json.load(f)
    assert saved["num_scenes"] == 2