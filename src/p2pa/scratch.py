"""The from-scratch backbone: `p2p`'s rectified-flow DiT, in ACE-Step's latent space.

`backbone=scratch` swaps the pretrained 2.39B ACE-Step DiT for a 220M transformer trained from
random initialisation on the same data, windows and roll encoder. Only the ACE-Step VAE is kept,
frozen, to turn latents back into audio. This backbone is not part of the paper.

The transformer and the flow objective are `p2p`'s own (`flowmatching.py`, `flow.py`). What this
module supplies is the seam, and the seam is where the danger is, because the two backbones
disagree on three things that no shape error would catch:

| | acestep | scratch |
|---|---|---|
| flow time | t=1 noise -> t=0 data, target `noise - data` | **t=0 noise -> t=1 data, target `data - noise`** |
| latent layout | `[B, T, C]` time-major | **`[B, C, T]` channels-first** |
| condition | 64-wide, into `src_latents` | `cond_dim`-wide, summed as `aligned_cond` |

A reversed time convention still produces a falling loss, so both conventions are kept whole and
separate rather than unified into one parameterised objective, and the transpose happens at this
boundary rather than in `dataset.py`, so both backbones train on identical bytes.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from .flow import drop_conditioning, flow_loss, sample_timesteps
from .flowmatching import FlowTransformer


class ScratchBackbone(nn.Module):
    """`p2p`'s FlowTransformer, wrapped to the interface `P2PAModule` speaks."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        scratch = cfg.scratch
        self.transformer = FlowTransformer(
            latent_channels=int(cfg.ace.latent_channels),
            hidden_dim=int(scratch.hidden_dim),
            num_heads=int(scratch.num_heads),
            mmdit_blocks=int(scratch.mmdit_blocks),
            dit_blocks=int(scratch.dit_blocks),
            # The condition is frame-aligned, so it is summed into the residual stream rather than
            # cross-attended: a sum states what attention would otherwise have to learn.
            context_dim=None,
            global_cond_dim=None,
            aligned_cond_dim=int(cfg.model.cond_dim),
            mlp_ratio=float(scratch.mlp_ratio),
        )
        # `FlowTransformer.final_proj` and `final_mod` are zero-init, standard DiT practice: the
        # model starts predicting exactly zero velocity, so at step 0 nothing upstream of the
        # output receives a gradient, the roll encoder included. `final_proj` moves on the first
        # update, after which everything behind it is connected;
        # `tests/test_scratch.py::test_the_whole_stack_is_connected_after_one_step` pins that.
        self.logit_normal = bool(scratch.logit_normal)
        self.cond_dropout = float(scratch.cond_dropout)

    # --- layout ------------------------------------------------------------
    #
    # One pair of functions, used everywhere, so the convention cannot drift between the loss and
    # the sampler. `dataset.py` is time-major because ACE-Step is; this backbone is channels-first
    # because `p2p`'s convolutions are.

    @staticmethod
    def to_backbone(x: torch.Tensor) -> torch.Tensor:
        """`[B, T, C]` -> `[B, C, T]`."""
        return x.transpose(1, 2).contiguous()

    @staticmethod
    def to_project(x: torch.Tensor) -> torch.Tensor:
        """`[B, C, T]` -> `[B, T, C]`."""
        return x.transpose(1, 2).contiguous()

    # --- conditioning ------------------------------------------------------

    def conditioning(self, encoded: torch.Tensor) -> dict:
        """`encoded` is `[B, T, cond_dim]` from the roll encoder; returned channels-first, which
        is what `FlowTransformer.aligned_proj` (a Conv1d) reads."""
        return {"aligned_cond": self.to_backbone(encoded)}

    def drop(self, cond: dict, training: bool) -> dict:
        """Classifier-free-guidance dropout of the roll, per item, on the training path only.

        Unlike the finetune, whose `src_latents` slot has no null value the checkpoint would
        recognise, this model learns one — zero — so sampling can guide on the roll itself.
        """
        if not training:
            return cond
        return drop_conditioning(cond, {"aligned_cond": self.cond_dropout})

    # --- objective ---------------------------------------------------------

    def denoise(self, noisy, timestep, cond, mask=None):
        return self.transformer(noisy, timestep, key_padding_mask=mask, **cond)

    def loss(
        self,
        latent: torch.Tensor,
        cond: dict,
        *,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rectified flow MSE, in *this* backbone's convention: t=0 noise, t=1 data.

        `timestep` and `noise` are the hooks the module's fixed-timestep, per-item-seeded
        validation needs. When both are None this is exactly `flow.flow_loss`.

        A caller's `timestep` is in the *caller's* convention, which is ACE-Step's (t=1 noise).
        It is flipped here rather than at the call site, so the module never has to know which
        backbone it is holding.
        """
        x1 = self.to_backbone(latent)
        if timestep is None and noise is None:
            value, _ = flow_loss(
                lambda x, t, c: self.denoise(x, t, c, mask),
                x1,
                cond,
                logit_normal=self.logit_normal,
                mask=mask,
            )
            return value

        eps = self.to_backbone(noise) if noise is not None else torch.randn_like(x1)
        if timestep is None:
            t = sample_timesteps(x1.size(0), x1.device, logit_normal=self.logit_normal)
        else:
            # ACE-Step counts t=1 as noise; this backbone counts t=1 as data.
            t = (1.0 - timestep).to(device=x1.device, dtype=x1.dtype)
        view = t.view(-1, 1, 1)
        x_t = (1 - view) * eps + view * x1
        velocity = x1 - eps
        prediction = self.denoise(x_t, t, cond, mask)
        if mask is not None:
            weight = mask[:, None, :].to(prediction.dtype)
            error = (prediction.float() - velocity.float()).pow(2) * weight
            return error.sum() / (weight.sum() * prediction.size(1)).clamp_min(1)
        return torch.nn.functional.mse_loss(prediction.float(), velocity.float())
