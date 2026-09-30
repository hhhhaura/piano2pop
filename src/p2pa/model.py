"""The LightningModule: ACE-Step v1.5, plus a roll encoder wired into its conditioning slot.

The backbone's objective is used as it stands — rectified flow with ACE-Step's own timestep
distribution (`sample_t_r`, a logit-normal at mu=-0.4, sigma=1.0). Its time convention is the same
as Stable Audio 3's and the *opposite* of p2p's: `t = 0` is data, `t = 1` is noise,
`x_t = x0(1-t) + x1·t`, and the target is `x1 - x0`.

Conditioning is assembled here rather than through `AceStepConditionGenerationModel.forward` for
three reasons, each of which would otherwise be a silent bug:

1. `prepare_condition` is `@torch.no_grad()`. The roll encoder's output has to reach the DiT with
   its graph intact, so `context_latents` is built here and only the text/lyric/timbre pack goes
   through the (differentiable) `encoder` submodule.
2. Upstream's `forward` computes `F.mse_loss` over the whole tensor and samples its own timestep.
   Validation here needs fixed timesteps and seeded noise, and the loss needs a length mask.
3. The reference-audio and lyric conditions are *identical for every item* in this task — a null
   timbre and an empty lyric. Running them at batch 1 and expanding is exact and saves a third of
   the forward pass, since the 4-layer timbre encoder over 750 frames is not far off the cost of
   the 24-layer DiT over 375 patches.

A trap worth naming: `AceStepDiTModel.forward` assigns `attention_mask = None` before it builds any
mask, so the padding mask it is handed is **ignored** — self- and cross-attention are geometric
only. That is upstream behaviour and the checkpoint was trained under it, so it is reproduced
rather than fixed. It is also why every batch here is length-homogeneous and every cached prompt is
a fixed 256 tokens: padding the model cannot mask is padding that changes the answer.

`backbone=scratch` replaces ACE-Step's DiT with `scratch.ScratchBackbone`, which carries its own
objective and time convention; the module dispatches to it in `build_conditioning`, `flow_loss`
and the optimiser, and everything else (data, roll encoder, validation, checkpoints) is shared.
"""

from __future__ import annotations

import hashlib
import math
import sys

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from .ace import load_dit, load_silence_latent, tile_silence, unused_parameter_names
from .cond import build_cond_encoder
from .config import latent_fps, resolve_path


def finite_summary(name: str, value: torch.Tensor) -> str:
    """A compact tensor summary for the exceptional non-finite-loss path.

    This is intentionally not run on healthy steps: reducing every decoder activation just to
    prove it is finite would add synchronisation to the hot path. When the scalar loss is already
    bad, however, naming the first bad boundary is much more useful than merely saying "NaN".
    """
    detached = value.detach()
    finite = torch.isfinite(detached)
    count = int(finite.sum().item())
    total = detached.numel()
    if count:
        good = detached[finite].float()
        bounds = f"min={float(good.min()):.6g},max={float(good.max()):.6g}"
    else:
        bounds = "min=none,max=none"
    return (
        f"{name}[shape={tuple(detached.shape)},dtype={detached.dtype},"
        f"finite={count}/{total},{bounds}]"
    )


class P2PAModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self.scratch = str(cfg.model.backbone) == "scratch"
        if self.scratch:
            from .scratch import ScratchBackbone

            # No pretrained weights, no text encoder, no `src_latents` slot: everything below that
            # reads ACE-Step's conditioning is skipped for this backbone.
            self.backbone = ScratchBackbone(cfg)
            self._remote = None
            self.roll_encoder = build_cond_encoder(cfg)
        else:
            # float32 masters: a full finetune updates every base weight.
            self.backbone = load_dit(cfg, "cpu", dtype=torch.float32)
            # The checkpoint ships its own modelling code, so `pack_sequences` and `sample_t_r`
            # live in a dynamically imported module rather than an installed package. Reaching
            # them through the loaded class is what keeps this file from reimplementing — and
            # drifting from — either.
            self._remote = sys.modules[type(self.backbone).__module__]

            # ACE-Step's own "no reference audio" value, which an untrained roll encoder
            # reproduces.
            silence = load_silence_latent(cfg)
            self.register_buffer("silence_latent", silence, persistent=False)
            self.roll_encoder = build_cond_encoder(cfg, silence)

        self.latent_channels = int(cfg.ace.latent_channels)
        self.latent_fps = latent_fps(cfg)
        self.refer_frames = int(cfg.ace.refer_frames)
        self.cfg_ratio = float(cfg.ace.cfg_ratio)
        self._unused = set() if self.scratch else set(unused_parameter_names(self.backbone))

        # The VAE is only needed to turn latents back into audio. Held in a plain dict so it stays
        # outside the module tree and its frozen parameters never enter a checkpoint.
        self._vae: dict[str, object] = {}
        # The fixed lyric and text blocks, loaded once per process. See `prompt.py`.
        self._null_prompt: dict[str, torch.Tensor] = {}

        # Freeze at construction only if there is actually a freeze window. Lightning prints its
        # parameter summary during `setup`, before `UnfreezeBackbone` runs on train start, so with
        # `freeze_steps=0` the summary reported 17.5M trainable for a run that trains 2.2B — a
        # display that contradicts the truth is worse than no display, and it cost a real
        # second-guessing of a correct dry run.
        self.backbone_frozen = self.freeze_steps > 0
        self.apply_freeze(self.backbone_frozen)
        if bool(cfg.trainer.gradient_checkpointing) and not self.scratch:
            self.backbone.gradient_checkpointing_enable()

    # --- freezing ------------------------------------------------------------

    @property
    def freeze_steps(self) -> int:
        """The finetune's freeze window; a from-scratch model has nothing pretrained to protect."""
        return 0 if self.scratch else int(self.cfg.finetune.freeze_steps)

    def conditioning_parameters(self):
        """The roll encoder: the only part of the model that has never seen a gradient."""
        yield from self.roll_encoder.parameters()

    def backbone_parameters(self):
        """Every DiT and condition-encoder weight, minus the audio tokenizer and detokenizer.

        No forward pass here can reach those two (`is_covers` is always 0), and training them would
        have AdamW carrying moments for 24 layers that never receive a gradient.
        """
        for name, parameter in self.backbone.named_parameters():
            if name not in self._unused:
                yield parameter

    def apply_freeze(self, frozen: bool) -> None:
        self.backbone_frozen = bool(frozen)
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
        if not frozen:
            for parameter in self.backbone_parameters():
                parameter.requires_grad_(True)
        for parameter in self.conditioning_parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        """A frozen backbone stays in eval mode even when Lightning puts the module in train mode.

        `requires_grad_(False)` stops the weights moving; eval mode stops the *module* behaving
        differently, which is a separate thing (dropout, in this architecture). Lightning calls
        `module.train()` at the start of every epoch, so this has to be re-asserted rather than set
        once.
        """
        super().train(mode)
        if mode and self.backbone_frozen:
            self.backbone.eval()
        return self

    def trainable_count(self) -> tuple[int, int]:
        conditioning = sum(p.numel() for p in self.conditioning_parameters())
        backbone = sum(p.numel() for p in self.backbone_parameters())
        return conditioning + backbone, conditioning

    # --- conditioning ------------------------------------------------------

    def vae(self):
        if "model" not in self._vae:
            from .ace import load_vae

            self._vae["model"] = load_vae(self.cfg, self.device)
        return self._vae["model"]

    def null_prompt(self, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """The fixed lyric and text blocks, at batch 1, on this module's device."""
        if not self._null_prompt:
            from .prompt import load_null_prompt

            self._null_prompt.update(load_null_prompt(self.cfg))
        return {
            key: value.to(device=self.device, dtype=dtype if value.is_floating_point() else None)
            for key, value in self._null_prompt.items()
        }

    def _timbre(self, batch_size: int, dtype: torch.dtype):
        """The null reference-audio timbre: a 750-frame slice of the checkpoint's silence latent.

        Deliberately null in every run. The obvious alternative — the target's own timbre — would
        hand the model an encoding of the answer, and every metric would improve for the wrong
        reason. Computed once and expanded, because it is identical for every item.
        """
        encoder = self.backbone.encoder
        reference = tile_silence(self.silence_latent, self.refer_frames).to(dtype)
        order = torch.zeros(1, dtype=torch.long, device=self.device)
        timbre, mask = encoder.timbre_encoder(reference, order)
        return timbre.expand(batch_size, -1, -1), mask.expand(batch_size, -1)

    def build_text_conditioning(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(encoder_hidden_states, encoder_attention_mask)`, ACE-Step's own packing order.

        Reproduces `AceStepConditionEncoder.forward` exactly — lyric with timbre, then text — at
        batch 1, then expands: every item carries the same fixed prompt, so running the three
        branches once is exact and saves a third of the forward pass.
        """
        encoder = self.backbone.encoder
        dtype = next(encoder.text_projector.parameters()).dtype
        prompt = self.null_prompt(dtype)

        text = encoder.text_projector(prompt["text"])
        lyric = encoder.lyric_encoder(
            inputs_embeds=prompt["lyric"], attention_mask=prompt["lyric_mask"]
        ).last_hidden_state
        timbre, timbre_mask = self._timbre(1, dtype)

        pack = self._remote.pack_sequences
        hidden, mask = pack(lyric, timbre, prompt["lyric_mask"].long(), timbre_mask.long())
        hidden, mask = pack(hidden, text, mask.long(), prompt["text_mask"].long())
        return hidden.expand(batch_size, -1, -1), mask.expand(batch_size, -1)

    def build_conditioning(self, batch: dict) -> dict:
        """Everything the DiT needs except the noisy latent and the timestep.

        `chunk_masks` is all ones: upstream, 1 means "generate this frame" and 0 means "keep the
        source", and every frame of every training window is generated. `context_latents` is then
        the same concatenation `prepare_condition` would have built, except that the source half is
        the roll encoder's output and carries a gradient.
        """
        encoded = self.roll_encoder(batch["roll"].to(self.device))
        if self.scratch:
            cond = self.backbone.conditioning(encoded)
            return self.backbone.drop(cond, self.training)
        hidden, encoder_mask = self.build_text_conditioning(encoded.shape[0])
        context = torch.cat([encoded, torch.ones_like(encoded)], dim=-1)
        return {
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": encoder_mask,
            "context_latents": context,
        }

    def drop_text(self, hidden: torch.Tensor) -> torch.Tensor:
        """ACE-Step's own classifier-free-guidance dropout.

        The whole cross-attention pack is replaced by the checkpoint's learned `null_condition_emb`
        — not by the embedding of an empty prompt, and not by zeros. `context_latents` is not
        touched, so the roll is never dropped and sampling has a single guidance scale with the
        condition always in force.
        """
        if not self.training or self.cfg_ratio <= 0:
            return hidden
        keep = torch.rand(hidden.shape[0], 1, 1, device=hidden.device) >= self.cfg_ratio
        null = self.backbone.null_condition_emb.to(hidden.dtype).expand_as(hidden)
        return torch.where(keep, hidden, null)

    def run_decoder(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        """The only way this project calls ACE-Step's decoder, for training and sampling alike."""
        outputs = self.backbone.decoder(
            hidden_states=noisy,
            timestep=timestep,
            # `use_meanflow=False` upstream means r == t always, so the second timestep embedding
            # is always evaluated at zero. Passing t for both reproduces that exactly.
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
        )
        return outputs[0]

    def denoise(self, noisy: torch.Tensor, timestep: torch.Tensor, cond: dict) -> torch.Tensor:
        return self.run_decoder(
            noisy,
            timestep,
            encoder_hidden_states=self.drop_text(cond["encoder_hidden_states"]),
            encoder_attention_mask=cond["encoder_attention_mask"],
            context_latents=cond["context_latents"],
        )

    # --- objective ---------------------------------------------------------

    def sample_timesteps(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        """ACE-Step's own logit-normal timestep sampler, at the checkpoint's own mu and sigma."""
        timestep, _ = self._remote.sample_t_r(
            batch_size,
            self.device,
            dtype,
            float(self.backbone.config.data_proportion),
            float(self.backbone.config.timestep_mu),
            float(self.backbone.config.timestep_sigma),
            use_meanflow=False,
        )
        return timestep

    def flow_loss(
        self,
        batch: dict,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        cond: dict | None = None,
    ) -> torch.Tensor:
        latent = batch["latent"].to(self.device)
        mask = batch["mask"].to(self.device)

        if self.scratch:
            # The whole objective lives in the backbone, because the two time conventions are
            # opposite and merging them into one parameterised loss is how a model gets trained
            # backwards. Batches are length-homogeneous, so the mask is passed only when it
            # actually excludes something.
            conditioning = cond if cond is not None else self.build_conditioning(batch)
            padding = None if bool(mask.all()) else mask
            value = self.backbone.loss(
                latent, conditioning, timestep=timestep, noise=noise, mask=padding
            )
            if not torch.isfinite(value):
                sample_ids = ",".join(str(v) for v in batch.get("sample_id", []))
                raise FloatingPointError(f"non-finite flow loss for sample_ids=[{sample_ids}]")
            return value

        if timestep is None:
            timestep = self.sample_timesteps(latent.shape[0], latent.dtype)
        if noise is None:
            noise = torch.randn_like(latent)

        view = timestep.view(-1, 1, 1).to(latent.dtype)
        noisy = latent * (1 - view) + noise * view
        target = noise - latent

        # Zero whatever is past each item's real length. Batches are homogeneous, so this is a
        # no-op in every configured run — but the DiT cannot mask padding (it discards the
        # attention mask), so without this a padded item's tail would reach the *unpadded* frames
        # through attention and change their predictions. Masking the loss alone does not fix
        # that: the contamination is in the input, not in the weighting.
        noisy = noisy * mask[:, :, None].to(noisy.dtype)

        conditioning = cond or self.build_conditioning(batch)
        prediction = self.denoise(noisy, timestep, conditioning)

        weight = mask[:, :, None].to(torch.float32)
        error = (prediction.float() - target.float()).pow(2) * weight
        loss = error.sum() / (weight.sum() * latent.shape[-1]).clamp_min(1.0)
        if not torch.isfinite(loss):
            boundaries = {
                "latent": latent,
                "timestep": timestep,
                "noise": noise,
                "noisy": noisy,
                "target": target,
                "encoder_hidden_states": conditioning["encoder_hidden_states"],
                "encoder_attention_mask": conditioning["encoder_attention_mask"],
                "context_latents": conditioning["context_latents"],
                "prediction": prediction,
                "error": error,
            }
            detail = "; ".join(finite_summary(name, value) for name, value in boundaries.items())
            sample_ids = ",".join(str(value) for value in batch.get("sample_id", []))
            raise FloatingPointError(
                f"non-finite flow loss for sample_ids=[{sample_ids}]: {detail}"
            )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.flow_loss(batch)
        size = batch["latent"].shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=size)
        self.log("train/seconds", batch["seconds"].mean(), prog_bar=True, batch_size=size)
        return loss

    def _fixed_noise(self, batch: dict, step: int) -> torch.Tensor:
        """Noise seeded per item and per timestep, so validation measures the model, not the draw."""
        latent = batch["latent"]
        pieces = []
        for sample_id in batch["sample_id"]:
            digest = hashlib.sha256(f"{sample_id}:{step}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
            pieces.append(torch.randn(latent.shape[1:], generator=generator))
        return torch.stack(pieces).to(device=self.device, dtype=latent.dtype)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """A one-sample estimate of an expectation makes top-k selection near random, and sixteen
        finetunes of one base model sit close enough together for that to matter. The loss is
        averaged over fixed timesteps with per-item seeded noise instead."""
        steps = max(1, int(self.cfg.trainer.val_timesteps))
        size = batch["latent"].shape[0]
        # The condition does not depend on the timestep, so the roll encoder and the text pack run
        # once per item rather than once per fixed timestep.
        cond = self.build_conditioning(batch)
        total = torch.zeros((), device=self.device)
        for step in range(steps):
            timestep = torch.full(
                (size,), (step + 0.5) / steps, device=self.device, dtype=batch["latent"].dtype
            )
            total = total + self.flow_loss(
                batch, timestep=timestep, noise=self._fixed_noise(batch, step), cond=cond
            )
        loss = total / steps
        self.log("val/loss", loss, prog_bar=True, batch_size=size, sync_dist=True)
        return loss

    # --- optimisation ------------------------------------------------------

    def configure_optimizers(self):
        conditioning = list(self.conditioning_parameters())
        backbone = list(self.backbone_parameters())
        if self.scratch:
            # One rate for both: the transformer is as new as the roll encoder.
            lr_cond = lr_backbone = float(self.cfg.scratch.lr)
        else:
            lr_cond = float(self.cfg.finetune.lr_cond)
            lr_backbone = float(self.cfg.finetune.lr_backbone)
        groups = [
            {"params": conditioning, "lr": lr_cond, "name": "cond"},
            {"params": backbone, "lr": lr_backbone, "name": "backbone"},
        ]
        optimizer = torch.optim.AdamW(
            groups,
            lr=lr_cond,
            weight_decay=float(self.cfg.trainer.weight_decay),
            betas=(0.9, 0.95),
        )

        warmup = max(1, int(self.cfg.trainer.warmup_steps))
        total = max(warmup + 1, int(self.cfg.trainer.max_steps))
        freeze = self.freeze_steps

        def cosine(step: int, start: int) -> float:
            # Warmup matters more in a finetune than in a from-scratch run: the roll encoder starts
            # at exactly zero and the backbone does not, so without it the first steps move the
            # pretrained weights the most and the condition the least.
            if step < start + warmup:
                return (step - start + 1) / warmup
            progress = (step - start - warmup) / max(1, total - start - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        def cond_schedule(step: int) -> float:
            return cosine(step, 0)

        def backbone_schedule(step: int) -> float:
            # Exactly zero until the freeze window ends, then its own warmup from there. Holding
            # the rate at zero rather than rebuilding the optimizer keeps AdamW's state continuous
            # and means a resume inside the window behaves the same as one outside it.
            return 0.0 if step < freeze else cosine(step, freeze)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [cond_schedule, backbone_schedule]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def load_checkpoint(cfg: DictConfig, checkpoint_path: str, map_location="cpu") -> P2PAModule:
    """Rebuild a module from a checkpoint, with the EMA weights of a from-scratch run.

    `cfg` is passed explicitly rather than read back from the checkpoint's hyperparameters, so a
    run recorded under an older schema loads through `migrate_config` like any other.
    """
    path = str(resolve_path(checkpoint_path))
    module = P2PAModule.load_from_checkpoint(path, cfg=cfg, map_location=map_location)
    if module.scratch:
        applied = apply_ema(module, path, map_location=map_location)
        print(f"[p2pa] applied {applied} EMA weights from the checkpoint")
    return module


def apply_ema(module: P2PAModule, checkpoint_path: str, map_location="cpu") -> int:
    """Copy the EMA shadow into the live weights.

    Lightning moves `ModelCheckpoint` to the end of the callback list, so the EMA callback has
    already swapped the raw weights back in by the time a checkpoint is written; the shadow lives
    in the callback's own state instead. Validation, and so top-k selection, ran under the averaged
    weights, so sampling uses them too.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False,
                            mmap=True)
    shadow = {}
    for state in (checkpoint.get("callbacks") or {}).values():
        if isinstance(state, dict) and "shadow" in state:
            shadow = state["shadow"]
            break
    own = dict(module.named_parameters())
    missing = sorted(set(own) - set(shadow))
    if shadow and missing:
        raise ValueError(f"the checkpoint's EMA shadow is missing {len(missing)} parameters, "
                         f"e.g. {missing[:3]}")
    with torch.no_grad():
        for name, value in shadow.items():
            if name in own:
                own[name].copy_(value.to(own[name].dtype))
    return len(shadow)
