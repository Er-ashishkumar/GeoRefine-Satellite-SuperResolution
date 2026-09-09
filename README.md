# GeoRefine — Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imagery

## Structure

- `src/` — AI/data pipeline: preprocessing, dataset, model, inference, validation, uncertainty, downstream (NDVI).
- `app/` — dashboard/UI (built later).
- `demo_data/` — standardized output contract consumed by the dashboard.
- `scripts/` — CLI entry points (e.g. run_inference.py).
- `data/`, `models/` — raw/processed data and trained weights (not committed; see .gitignore).

## Output contract
demo_data/
├── input/
│ └── scene.tif
├── sr/
│ └── scene_sr.tif
├── uncertainty/
│ └── scene_uncertainty.tif
├── ndvi/
│ ├── original_ndvi.tif
│ └── sr_ndvi.tif
├── metrics/
│ └── metrics.json
└── manifest.json

`manifest.json` lists relative paths to each output so downstream code
(dashboard or otherwise) never needs to hard-code filenames.
`metrics.json` always contains at least:

```json
{ "psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "sam": 0.0 }
```

## Development phases

0. Interface + foundation (this commit)
1. Dataset
2. Preprocessing
3. LR/HR patch generation
4. Super-resolution model
5. Inference
6. Validation (PSNR/SSIM/RMSE/SAM)
7. Uncertainty estimation
8. NDVI (original vs. SR)
9. Full reproducible pipeline + demo outputs

## Scientific honesty notice

The super-resolution model produces a **finer spatial representation based
on learned patterns and available imagery** — it does not create new
ground-truth information. Any HR reference imagery that is not truly
independent ground truth (e.g. synthetically degraded HR proxies) is
documented as such in the dataset docs. Uncertainty maps indicate regions
where the model's reconstruction is less stable and should be interpreted
cautiously, not treated as calibrated statistical confidence unless
explicitly stated.