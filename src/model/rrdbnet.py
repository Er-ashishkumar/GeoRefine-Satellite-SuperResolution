"""
RRDBNet (Residual-in-Residual Dense Block Network) - the architecture used
by Real-ESRGAN's x4plus pretrained model.

This is a direct, dependency-free reimplementation of the published
architecture, written to match the exact layer names and shapes used in
the official RealESRGAN_x4plus.pth checkpoint (originally released by the
Real-ESRGAN authors: https://github.com/xinntao/Real-ESRGAN). It exists so
this project can load that checkpoint using only plain torch, without
depending on the `basicsr`/`realesrgan` PyPI packages - which, as of this
writing, fail to build on current Python versions due to an unmaintained
setup.py in basicsr. See src/model/README.md for the full reasoning.

Architecture reference: "ESRGAN: Enhanced Super-Resolution Generative
Adversarial Networks" (Wang et al., 2018), with the RRDB (Residual-in-
Residual Dense Block) design, as adapted by Real-ESRGAN for x4plus.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """One densely-connected residual block (5 conv layers) used inside an RRDB."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block: three ResidualDenseBlocks with an outer residual."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Full RRDBNet generator, matching RealESRGAN_x4plus.pth's layer layout.

    Default args (num_feat=64, num_block=23, num_grow_ch=32) match the
    official x4plus checkpoint exactly - do not change these unless loading
    a different checkpoint with a documented different configuration.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling: two 2x pixel-shuffle-free (nearest + conv) stages for x4 total.
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def load_rrdbnet_state_dict(model: RRDBNet, checkpoint_path: str, device: str = "cpu") -> RRDBNet:
    """Load an official RealESRGAN_x4plus.pth checkpoint into an RRDBNet instance.

    The official checkpoint stores weights under a top-level "params_ema" key
    (or occasionally "params") rather than a bare state_dict - this handles
    both, and raises a clear error rather than silently loading nothing if
    the checkpoint format is unrecognized.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unrecognized checkpoint format at {checkpoint_path}: "
            f"expected a dict with 'params_ema' or 'params' key, or a bare state_dict."
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model