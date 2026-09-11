"""
Phase 2 orchestration: turn Phase 1's dataset_index.json into preprocessing_index.json.

Design decision (documented here for the handoff to later phases):

Phase 1's dataset_index.json is metadata-only and deliberately untouched by
Phase 2 - src/dataset/ and its output stay a stable, minimal contract for
any future consumer. Phase 2 reads dataset_index.json as input and writes
its own output, data/processed/preprocessing_index.json, which carries the
pixel-level results this phase is responsible for:

- band_stack: which files were matched to which spectral band, and whether
  that succeeded
- for scenes with a real HR reference: an alignment report per HR file
- for scenes without a real HR reference (and synthetic_hr_fallback
  enabled in config.yaml): is_synthetic_hr = true and the generated
  synthetic LR raster's path

This keeps the two phases' outputs independently inspectable and avoids
Phase 2 silently rewriting a file Phase 1 owns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.dataset.scene import RasterMetadata

from .alignment import check_alignment
from .bands import load_band_stack
from .synthetic_hr import generate_synthetic_pair_for_scene


def _meta_from_dict(d: Dict[str, Any]) -> RasterMetadata:
    return RasterMetadata(
        path=d["path"],
        crs=d["crs"],
        width=d["width"],
        height=d["height"],
        count=d["count"],
        dtype=d["dtype"],
        nodata=d["nodata"],
        resolution_x=d["resolution_x"],
        resolution_y=d["resolution_y"],
        bounds=tuple(d["bounds"]),
    )


def _process_scene(scene: Dict[str, Any], config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    scene_id = scene["scene_id"]
    lr_metas = [_meta_from_dict(d) for d in scene["lr_files"]]
    hr_metas = [_meta_from_dict(d) for d in scene["hr_files"]]

    sentinel2_cfg = config.get("sentinel2", {})
    requested_bands = {"red": sentinel2_cfg.get("red_band", "B4"), "nir": sentinel2_cfg.get("nir_band", "B8")}

    result: Dict[str, Any] = {
        "scene_id": scene_id,
        "phase1_issues": scene.get("issues", []),
        "is_synthetic_hr": False,
        "band_stack": None,
        "alignment": [],
        "synthetic": None,
        "issues": [],
    }

    if not lr_metas:
        result["issues"].append("No LR files available (Phase 1 reported no LR files for this scene)")
        return result

    band_stack = load_band_stack(lr_metas, requested_bands)
    result["band_stack"] = {
        "ok": band_stack.ok,
        "band_order": band_stack.band_order,
        "source_files": band_stack.source_files,
        "issues": band_stack.issues,
    }
    if not band_stack.ok:
        result["issues"].extend(f"Band merge: {msg}" for msg in band_stack.issues)
        return result

    pp_cfg = config.get("preprocessing", {})
    min_overlap = pp_cfg.get("min_overlap_fraction", 0.5)
    ratio_tolerance = pp_cfg.get("resolution_ratio_tolerance", 0.15)

    if hr_metas:
        expected_ratio = None
        synth_cfg = config.get("dataset", {}).get("synthetic_hr_fallback", {})
        if synth_cfg:
            expected_ratio = synth_cfg.get("downsample_factor")

        representative_lr = lr_metas[0]
        for hr_meta in hr_metas:
            report = check_alignment(
                representative_lr,
                hr_meta,
                expected_scale_factor=expected_ratio,
                min_overlap_fraction=min_overlap,
                ratio_tolerance=ratio_tolerance,
            )
            result["alignment"].append(
                {
                    "hr_file": hr_meta.path,
                    "crs_match": report.crs_match,
                    "resolution_ratio": report.resolution_ratio,
                    "expected_ratio": report.expected_ratio,
                    "bounds_overlap_fraction": report.bounds_overlap_fraction,
                    "is_aligned": report.is_aligned,
                    "issues": report.issues,
                }
            )
            if not report.is_aligned:
                result["issues"].append(f"HR file {Path(hr_meta.path).name} failed alignment check")
    else:
        synth_cfg = config.get("dataset", {}).get("synthetic_hr_fallback", {})
        if synth_cfg.get("enabled", False):
            synth_output_subdir = pp_cfg.get("synthetic_output_subdir", "synthetic_lr")
            pair = generate_synthetic_pair_for_scene(
                scene_id=scene_id,
                proxy_hr_array=band_stack.data,
                band_order=band_stack.band_order,
                source_paths=band_stack.source_files,
                crs=band_stack.crs,
                transform_6=band_stack.transform,
                blur_sigma=synth_cfg.get("blur_sigma", 1.0),
                downsample_factor=synth_cfg.get("downsample_factor", 4),
                output_dir=Path(output_dir) / synth_output_subdir,
            )
            result["is_synthetic_hr"] = True
            result["synthetic"] = {
                "proxy_hr_paths": pair.proxy_hr_paths,
                "synthetic_lr_path": pair.synthetic_lr_path,
                "band_order": pair.band_order,
                "blur_sigma": pair.blur_sigma,
                "downsample_factor": pair.downsample_factor,
                "proxy_hr_shape": list(pair.proxy_hr_shape),
                "synthetic_lr_shape": list(pair.synthetic_lr_shape),
            }
        else:
            result["issues"].append(
                "No real HR reference and dataset.synthetic_hr_fallback.enabled is false in config.yaml"
            )

    return result


def build_preprocessing_index(
    dataset_index_path: Path,
    config: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """Read dataset_index.json (Phase 1) and write preprocessing_index.json (Phase 2)."""
    with open(dataset_index_path, "r", encoding="utf-8") as f:
        dataset_index = json.load(f)

    processed_dir = Path(config["paths"]["data_processed"])

    scenes_out: List[Dict[str, Any]] = [
        _process_scene(scene, config, processed_dir) for scene in dataset_index.get("scenes", [])
    ]

    index = {
        "num_scenes": len(scenes_out),
        "num_scenes_with_real_hr_aligned": sum(
            1 for s in scenes_out if s["alignment"] and all(a["is_aligned"] for a in s["alignment"])
        ),
        "num_scenes_synthetic_hr": sum(1 for s in scenes_out if s["is_synthetic_hr"]),
        "num_scenes_with_issues": sum(1 for s in scenes_out if s["issues"]),
        "scenes": scenes_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return index
