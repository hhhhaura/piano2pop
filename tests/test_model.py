from __future__ import annotations

import pytest
import torch

from tests.conftest import make_batch


def test_the_gradient_reaches_the_encoder_through_the_dit(tiny_module, tiny_cfg):
    """The `@torch.no_grad()` trap.

    ACE-Step's `prepare_condition` is decorated `@torch.no_grad()`. Routing the roll encoder's
    output through it would sever the graph silently — the loss would still fall, driven by the
    backbone alone, and the condition would never learn to mean anything. `build_conditioning`
    assembles `context_latents` itself for exactly this reason, and this is what proves it.
    """
    batch = make_batch(tiny_cfg)
    tiny_module.train()
    tiny_module.flow_loss(batch).backward()
    grads = [
        p.grad for p in tiny_module.roll_encoder.parameters() if p.grad is not None
    ]
    assert grads, "no parameter of the roll encoder received a gradient at all"
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_the_condition_is_the_src_latents_slot(tiny_module, tiny_cfg):
    """`context_latents` is `[src_latents, chunk_masks]`, and the mask half is all ones."""
    batch = make_batch(tiny_cfg, frames=16)
    cond = tiny_module.build_conditioning(batch)
    channels = int(tiny_cfg.ace.latent_channels)
    assert cond["context_latents"].shape == (2, 16, 2 * channels)
    source, chunk = cond["context_latents"].split(channels, dim=-1)
    assert torch.equal(chunk, torch.ones_like(chunk))
    # Untrained, so the source half is exactly the silence latent the fixture supplies.
    assert torch.allclose(source, torch.full_like(source, 0.25))


def test_the_loss_ignores_padded_frames(tiny_module, tiny_cfg):
    """Batches are homogeneous in production, but the mask must still be load-bearing.

    The DiT discards the attention mask it is handed, so if the loss ever stopped honouring one, a
    padded item's tail would silently become a training target of pure noise.
    """
    batch = make_batch(tiny_cfg, frames=16)
    batch["mask"][:, 8:] = False
    # eval, so guidance dropout does not redraw between the two calls.
    tiny_module.eval()
    timestep = torch.full((2,), 0.5)
    noise = torch.randn_like(batch["latent"])
    cond = tiny_module.build_conditioning(batch)
    before = tiny_module.flow_loss(batch, timestep=timestep, noise=noise, cond=cond)
    batch["latent"][:, 8:] += 100.0
    after = tiny_module.flow_loss(batch, timestep=timestep, noise=noise, cond=cond)
    assert torch.allclose(before, after)


def test_non_finite_loss_names_the_bad_boundary_and_samples(tiny_module, tiny_cfg, monkeypatch):
    batch = make_batch(tiny_cfg, frames=16)

    def bad_decoder(noisy, timestep, cond):
        result = torch.zeros_like(noisy)
        result[0, 0, 0] = torch.nan
        return result

    monkeypatch.setattr(tiny_module, "denoise", bad_decoder)
    with pytest.raises(
        FloatingPointError, match=r"sample_ids=\[track0,track1\].*prediction.*255/256"
    ):
        tiny_module.flow_loss(batch)


def test_freezing_leaves_only_the_encoder_trainable(tiny_module):
    tiny_module.apply_freeze(True)
    trainable = {n for n, p in tiny_module.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("roll_encoder.") for n in trainable)

    tiny_module.apply_freeze(False)
    trainable = {n for n, p in tiny_module.named_parameters() if p.requires_grad}
    assert any(n.startswith("roll_encoder.") for n in trainable)
    assert any(n.startswith("backbone.decoder.layers.") for n in trainable), (
        "unfreezing must release the DiT itself: this is a full finetune"
    )


def test_the_audio_tokenizer_is_never_trained(tiny_module):
    """`is_covers` is always 0, so the tokenizer and detokenizer cannot receive a gradient.

    Leaving them in the optimizer would have AdamW carrying moments for weights that never move.
    """
    tiny_module.apply_freeze(False)
    names = {n for n, p in tiny_module.named_parameters() if p.requires_grad}
    assert not any(n.startswith(("backbone.tokenizer.", "backbone.detokenizer.")) for n in names)


def test_the_backbone_learning_rate_is_zero_inside_the_freeze_window(tiny_module, tiny_cfg):
    tiny_cfg.finetune.freeze_steps = 100
    tiny_cfg.trainer.warmup_steps = 10
    tiny_cfg.trainer.max_steps = 1000
    bundle = tiny_module.configure_optimizers()
    schedule = bundle["lr_scheduler"]["scheduler"]
    conditioning, backbone = schedule.lr_lambdas
    assert backbone(0) == 0.0 and backbone(99) == 0.0
    assert backbone(100) > 0.0
    assert conditioning(0) > 0.0


def test_validation_loss_is_deterministic(tiny_module, tiny_cfg):
    batch = make_batch(tiny_cfg, frames=16)
    tiny_module.eval()
    with torch.no_grad():
        first = tiny_module.validation_step(batch, 0)
        second = tiny_module.validation_step(batch, 0)
    assert torch.allclose(first, second)


def test_text_dropout_only_happens_in_training(tiny_module, tiny_cfg):
    hidden = torch.randn(4, 6, tiny_module.backbone.config.hidden_size)
    tiny_module.eval()
    assert torch.equal(tiny_module.drop_text(hidden), hidden)
    tiny_module.train()
    tiny_module.cfg_ratio = 1.0
    dropped = tiny_module.drop_text(hidden)
    null = tiny_module.backbone.null_condition_emb.expand_as(hidden)
    assert torch.allclose(dropped, null)


@pytest.mark.parametrize("frames", [8, 16, 30])
def test_any_whole_number_of_patches_runs(tiny_module, tiny_cfg, frames):
    batch = make_batch(tiny_cfg, frames=frames)
    assert torch.isfinite(tiny_module.flow_loss(batch))
