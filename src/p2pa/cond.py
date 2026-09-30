"""The conditioning encoder: a piano roll at the latent rate in, one ACE-Step latent frame out.

    [B, C_roll, T] -> [B, T, 64]

`p2p`'s condition encoder: dilation-free residual 1-D convolutions. Depth buys context, never
resolution — kernel 5 over 5 layers reaches under a second at 25 Hz, enough to tell an attack from
a sustain or a release. There is no strided stem, so the roll must already be at the latent rate
(`roll.oversample=1`), and `validate_config` enforces it.

The output is **time-major and 64-wide**, because it goes straight into ACE-Step's `src_latents`
slot. Two things follow from that:

* the final projection is **zero-initialised** and the checkpoint's own `silence_latent` is added
  as a per-frame bias, so an untrained encoder emits exactly the "no reference audio" latent the
  backbone was pretrained to see. A fresh model is therefore bit-identical to stock ACE-Step
  text2music, and training moves away from that rather than starting somewhere meaningless;
* nothing here is normalised into a private scale. The encoder has to speak the VAE's latent
  distribution, so the residual it learns is added to a vector already in it.

Under `backbone=scratch` there is no pretrained slot to imitate: the encoder ends at its GroupNorm
and hands `cond_dim` channels to the transformer's own `aligned_proj`, exactly as in `p2p`.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from .pianoroll import total_roll_channels


class RollEncoder(nn.Module):
    """Roll in, `src_latents` out."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        cond_dim: int = 512,
        layers: int = 5,
        kernel: int = 5,
        silence: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("cond_layers counts the stem, so it must be at least 1")
        if kernel % 2 == 0:
            raise ValueError(
                f"model.cond_kernel={kernel} must be odd: an even kernel cannot be centred, so "
                "the condition would sit half a frame off the audio"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel // 2
        self.stem = nn.Conv1d(in_channels, cond_dim, kernel, padding=padding)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(8, cond_dim),
                nn.SiLU(),
                nn.Conv1d(cond_dim, cond_dim, kernel, padding=padding),
            )
            for _ in range(layers - 1)
        )
        self.out_norm = nn.GroupNorm(8, cond_dim)
        # Identity when nothing needs reshaping (the scratch backbone): no extra parameters.
        self.out = nn.Identity() if out_channels == cond_dim else nn.Linear(cond_dim, out_channels)
        # Zero, not small-random. Together with the silence bias below this makes the untrained
        # condition an exact no-op rather than an approximate one, which is what lets the identity
        # test assert maxdiff 0.0 instead of picking a tolerance.
        if isinstance(self.out, nn.Linear):
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        # `[1, T0, 64]`, non-persistent: it is reproducible from the checkpoint by name and would
        # otherwise add 3.8 MB to every saved state dict.
        if silence is None:
            silence = torch.zeros(1, 1, out_channels)
        self.register_buffer("silence", silence.float(), persistent=False)

    def set_silence(self, silence: torch.Tensor) -> None:
        self.silence = silence.to(device=self.silence.device, dtype=self.silence.dtype)

    def _silence_for(self, frames: int, dtype, device) -> torch.Tensor:
        available = self.silence.shape[1]
        if available >= frames:
            tile = self.silence[:, :frames]
        else:
            tile = self.silence.repeat(1, -(-frames // available), 1)[:, :frames]
        return tile.to(device=device, dtype=dtype)

    def forward(self, roll: torch.Tensor) -> torch.Tensor:
        """`roll` is `[B, C, T]`; returns `[B, T, out_channels]`, ready to pass as `src_latents`."""
        if roll.shape[1] != self.in_channels:
            raise ValueError(
                f"roll has {roll.shape[1]} channels but the encoder expects {self.in_channels}"
            )
        hidden = self.stem(roll)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        encoded = self.out(self.out_norm(hidden).transpose(1, 2))
        return encoded + self._silence_for(encoded.shape[1], encoded.dtype, encoded.device)


def build_cond_encoder(cfg: DictConfig, silence: torch.Tensor | None = None) -> RollEncoder:
    """64 channels biased at ACE-Step's silence latent for the finetune; `cond_dim` channels and
    no bias for the scratch backbone, whose zero-init output layer makes the untrained condition a
    no-op instead."""
    scratch = str(cfg.model.backbone) == "scratch"
    return RollEncoder(
        total_roll_channels(cfg),
        int(cfg.model.cond_dim) if scratch else int(cfg.ace.latent_channels),
        cond_dim=int(cfg.model.cond_dim),
        layers=int(cfg.model.cond_layers),
        kernel=int(cfg.model.cond_kernel),
        silence=None if scratch else silence,
    )
