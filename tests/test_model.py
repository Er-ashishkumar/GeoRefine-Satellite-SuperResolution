"""
Tests for src.model.rrdbnet and src.model.super_resolution.
These tests use a small, randomly-initialized RRDBNet (not the real
pretrained checkpoint - loading and running the full 64MB model in every
test run would be slow and would require the checkpoint to be downloaded
first). They confirm architecture output shapes and the per-band
processing logic in apply_super_resolution, independent of the actual
pretrained weights.
"""
from __future__ import annotations
import numpy as np
import pytest
import torch
from src.model.rrdbnet import RRDBNet
from src.model.super_resolution import _normalize_band, apply_super_resolution
@pytest.fixture
def tiny_model():
    """A small, fast RRDBNet for testing - same interface as the real x4plus model."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=2, num_grow_ch=4, scale=4)
    model.eval()
    return model
def test_rrdbnet_output_shape(tiny_model):
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out = tiny_model(x)
    assert out.shape == (1, 3, 64, 64)  # 16 * scale(4)
def test_normalize_band_scales_to_unit_range():
    band = np.array([[10.0, 20.0], [30.0, 40.0]])
    normalized, band_min, band_max = _normalize_band(band)
    assert band_min == 10.0
    assert band_max == 40.0
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
def test_normalize_band_constant_input_returns_zeros():
    band = np.full((4, 4), 5.0)
    normalized, band_min, band_max = _normalize_band(band)
    assert band_min == band_max == 5.0
    np.testing.assert_array_equal(normalized, np.zeros((4, 4)))
def test_apply_super_resolution_output_shape(tiny_model):
    band_stack = (np.random.rand(2, 16, 16) * 200 + 10).astype("float32")
    sr = apply_super_resolution(tiny_model, band_stack)
    assert sr.shape == (2, 64, 64)  # 2 bands preserved, spatial dims x4
def test_apply_super_resolution_rejects_wrong_ndim(tiny_model):
    bad_input = np.random.rand(16, 16).astype("float32")  # missing band dimension
    with pytest.raises(ValueError):
        apply_super_resolution(tiny_model, bad_input)
def test_apply_super_resolution_rescales_to_original_range(tiny_model):
    band_stack = (np.random.rand(2, 16, 16) * 200 + 50).astype("float32")
    sr_rescaled = apply_super_resolution(tiny_model, band_stack, rescale_to_original_range=True)
    sr_normalized = apply_super_resolution(tiny_model, band_stack, rescale_to_original_range=False)
    # Rescaled output should generally occupy a different (typically larger) numeric
    # range than the [0, 1]-clipped normalized output, for non-degenerate input.
    assert sr_rescaled.max() != sr_normalized.max() or sr_rescaled.min() != sr_normalized.min()
