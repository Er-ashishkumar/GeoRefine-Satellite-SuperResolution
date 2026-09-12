# src/model - Phase 4: Super-Resolution Model
## Why a direct RRDBNet implementation instead of the `realesrgan` package
The official way to run Real-ESRGAN is via the `realesrgan` PyPI package,
which depends on `basicsr`. As of this project's Phase 4 implementation,
`basicsr`'s packaging (`setup.py`) fails to build on current Python
versions (a `KeyError: '__version__'` during its own version-string
parsing) - this is a bug in `basicsr` itself, unrelated to this project's
code or environment.
Since the actual model architecture (RRDBNet) needed for inference with
a pretrained checkpoint is straightforward, `rrdbnet.py` reimplements it
directly in plain PyTorch, matching the official checkpoint's layer names
and shapes exactly. `load_rrdbnet_state_dict()` loads the same official
`RealESRGAN_x4plus.pth` weights this way, with no `basicsr`/`realesrgan`
dependency required. This keeps GeoRefine's dependency list minimal (per
the project's dependency-management rules) and avoids depending on a
package that currently fails to install.
If `basicsr` is fixed upstream in the future, or if this project later
needs `basicsr`'s training infrastructure (not just inference), that
would be a deliberate, documented decision to revisit - not a silent
requirement to undo this approach.
## Band-handling: per-band pseudo-RGB approximation
Sentinel-2 imagery here is 2-band (red, nir). Real-ESRGAN's pretrained
weights expect 3-channel RGB. This module applies the model independently
to each band (replicated to 3 identical channels, one output channel taken
back) rather than fine-tuning the architecture for 2-channel input or
inventing an undocumented fake-third-channel scheme. See the docstring at
the top of `super_resolution.py` for the full reasoning and the known
limitation this implies (no cross-band correlation is exploited).
## Pretrained checkpoint
Source: https://github.com/xinntao/Real-ESRGAN (RealESRGAN_x4plus.pth,
~64MB, 4x upscaling factor - matches this project's `downsample_factor: 4`
already used throughout Phases 1-3).
Download with:
    python scripts/download_pretrained_model.py
This is never called automatically by any other script - it must be run
explicitly once. The checkpoint file itself is not committed to git (see
`.gitignore`'s `models/*.pth` rule).
## Fine-tuning
Not implemented in Phase 4. Per the project's model-strategy instructions,
the pretrained model is used as-is; fine-tuning is deferred and would only
be pursued if the per-band pseudo-RGB approximation proves clearly
inadequate once real data and Phase 6 validation metrics are available.
