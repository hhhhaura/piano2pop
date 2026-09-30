"""Rectified flow objective and conditioning dropout for the from-scratch backbone, from `p2p`.

Convention throughout: t=0 is noise, t=1 is data, velocity = data - noise.

Conditioning travels as one dict so that a model with any mix of context / aligned / global
conditioning uses the same objective, and CFG dropout zeroes every stream it was given.
The keys are whatever the denoiser closure expects.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

# A denoiser: (x_t [B,C,T], t [B], conditioning) -> velocity [B,C,T]
Conditioning = Mapping[str, torch.Tensor]
Denoiser = Callable[[torch.Tensor, torch.Tensor, Conditioning], torch.Tensor]


def sample_timesteps(
    batch: int, device: torch.device, *, logit_normal: bool = True, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """Timesteps in (0, 1).

    Logit-normal (Flux / SD3) concentrates samples near t=0.5, where the velocity field is
    hardest to learn. Uniform is the baseline.
    """
    if not logit_normal:
        return torch.rand(batch, device=device)
    return torch.sigmoid(torch.randn(batch, device=device) * std + mean)


def flow_loss(
    denoiser: Denoiser,
    x1: torch.Tensor,
    cond: Conditioning | None = None,
    *,
    logit_normal: bool = True,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rectified flow MSE on the velocity.

    ``mask``, when given, is ``[B, T]`` boolean (True = real frame) — excludes a curriculum
    batch's silence-padded frames from the loss and normalises by the count of real elements
    rather than every element, so the loss stays comparable across batches with different amounts
    of padding. None (the fixed-clip path, where nothing is ever padded) is a plain mean, byte-
    for-byte the same computation as before this parameter existed.
    """
    eps = torch.randn_like(x1)
    t = sample_timesteps(x1.size(0), x1.device, logit_normal=logit_normal)
    t_view = t.view(-1, 1, 1)
    x_t = (1 - t_view) * eps + t_view * x1
    velocity = x1 - eps
    prediction = denoiser(x_t, t, cond or {})
    if mask is not None:
        weight = mask[:, None, :].to(prediction.dtype)  # [B, 1, T], broadcasts over channels
        diff = (prediction.float() - velocity.float()).pow(2) * weight
        loss = diff.sum() / (weight.sum() * prediction.size(1)).clamp_min(1)
    else:
        loss = F.mse_loss(prediction.float(), velocity.float())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite flow loss: {loss.item()}")
    return loss, {"flow": loss.detach()}


def drop_conditioning(
    cond: Conditioning,
    probs: Mapping[str, float] | None = None,
    *,
    default_p: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Zero each conditioning stream independently, per sample in the batch.

    Per-sample rather than per-batch: a batch-level coin flip correlates the dropout across
    every item, which wastes most of the batch on the same conditioning combination.
    """
    probs = probs or {}
    dropped: dict[str, torch.Tensor] = {}
    for key, value in cond.items():
        p = probs.get(key, default_p)
        if p <= 0.0:
            dropped[key] = value
            continue
        keep = torch.rand(value.size(0), device=value.device) >= p
        dropped[key] = value * keep.view(-1, *([1] * (value.dim() - 1))).to(value.dtype)
    return dropped
