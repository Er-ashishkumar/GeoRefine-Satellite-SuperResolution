"""
Super-resolution model interface for GeoRefine.

Loads the pretrained Real-ESRGAN x4plus checkpoint (RRDBNet architecture,
see rrdbnet.py) and exposes a simple apply_super_resolution() function that
Phase 5 (Inference) calls on a scene or patch's band-stacked array.

Band-handling decision (documented here, not hidden):

The pretrained model was trained on 3-channel RGB natural images. This
project's Sentinel-2 imagery is 2-band (red, nir) - a different sensing
modality entirely, with no pretrained multi-spectral SR model readily
available. Rather than inventing an undocumented "fake third channel" hack
or spending significant time fine-tuning the architecture's input/output
layers for 2 channels (out of scope per the project's model-strategy
instructions: prefer pretrained, do not over-invest in architecture
changes), this module applies the model **independently per band**:

    for each band separately:
        replicate the single band to 3 identical channels (grayscale -> RGB)
        run the pretrained model
        take one of the 3 (identical) output channels back

This treats each spectral band as an independent grayscale super-resolution
problem. It does not exploit any cross-band spatial correlation the model
might otherwise pick up from true multi-channel input, and it is not
equivalent to a model actually trained on multispectral data. This is
documented as a per-band approximation, not presented as ideal multispectral
SR - see the project's scientific honesty notice in README.md. Revisiting
this with genuine multi-band fine-tuning is a documented known limitation,
to be reconsidered only if this approximation proves inadequate once
real data and validation metrics (Phase 6) are available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .rrdbnet import RRDBNet, load_rrdbnet_state_dict

_DEFAULT_SCALE = 4
_DEFAULT_CHECKPOINT_NAME = "RealESRGAN_x4plus.pth"


def load_model(
    checkpoint_path: Optional[str] = None,
    models_dir: str = "models",
    device: str = "cpu",
) -> RRDBNet:
    """Load the pretrained RRDBNet model once, ready for repeated inference calls.

    Raises FileNotFoundError with a clear, actionable message if the
    checkpoint hasn't been downloaded yet (see scripts/download_pretrained_model.py).
    """
    if checkpoint_path is None:
        checkpoint_path = str(Path(models_dir) / _DEFAULT_CHECKPOINT_NAME)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {checkpoint_path}.\n"
            "Run `python scripts/download_pretrained_model.py` first."
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=_DEFAULT_SCALE)
    model = load_rrdbnet_state_dict(model, checkpoint_path, device=device)
    return model


def _normalize_band(band: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale a single band to [0, 1] float32 for the model, returning the
    original min/max so the caller can rescale the output back if desired.
    """
    band = band.astype(np.float32)
    band_min, band_max = float(band.min()), float(band.max())
    if band_max <= band_min:
        return np.zeros_like(band), band_min, band_max
    normalized = (band - band_min) / (band_max - band_min)
    return normalized, band_min, band_max


def apply_super_resolution(
    model: RRDBNet,
    band_stack: np.ndarray,
    device: str = "cpu",
    rescale_to_original_range: bool = True,
) -> np.ndarray:
    """Run the pretrained model independently on each band of a (bands, H, W) array.

    Returns a (bands, H*scale, W*scale) array. Each band is processed as an
    independent grayscale-replicated-to-RGB image - see this module's
    docstring for why, and for the documented limitation this implies.

    Values are normalized to [0, 1] per band before inference (required by
    the pretrained model's expected input range) and, if
    rescale_to_original_range is True, rescaled back to that band's
    original min/max afterward so downstream code (NDVI, metrics) can work
    with values on a comparable scale to the input - this rescaling does
    not recover any lost radiometric calibration, it only undoes the
    normalization this function itself applied.
    """
    if band_stack.ndim != 3:
        raise ValueError(f"Expected a (bands, H, W) array, got shape {band_stack.shape}")

    num_bands, height, width = band_stack.shape
    scale = model.scale
    output = np.zeros((num_bands, height * scale, width * scale), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for b in range(num_bands):
            normalized, band_min, band_max = _normalize_band(band_stack[b])

            rgb_like = np.stack([normalized, normalized, normalized], axis=0)  # (3, H, W)
            tensor = torch.from_numpy(rgb_like).unsqueeze(0).to(device)  # (1, 3, H, W)

            sr_tensor = model(tensor)  # (1, 3, H*scale, W*scale)
            sr_single_channel = sr_tensor[0, 0].cpu().numpy()  # take one of the 3 identical-ish output channels

            sr_single_channel = np.clip(sr_single_channel, 0.0, 1.0)

            if rescale_to_original_range and band_max > band_min:
                sr_single_channel = sr_single_channel * (band_max - band_min) + band_min

            output[b] = sr_single_channel

    return output