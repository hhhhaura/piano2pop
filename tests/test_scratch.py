"""`backbone=scratch`: `p2p`'s rectified-flow DiT, trained from random initialisation.

The seam between the two backbones is where the danger is, because they disagree on three things
that no shape error would catch — the flow time convention, the tensor layout, and how the
condition enters. A reversed time convention still produces a falling loss, so these tests check
behaviour rather than shapes.
"""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from p2pa.config import TrainConfig, config_dir, validate_config
from p2pa.pianoroll import total_roll_channels

ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)


def scratch_cfg(tmp_path, monkeypatch, **overrides):
    (tmp_path / "data" / "p2pdata").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("P2PA_DATA_ROOT", str(tmp_path))
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        cfg = compose(
            config_name="config",
            overrides=["backbone=scratch", *[f"{k}={v}" for k, v in overrides.items()]],
        )
    # Small enough to build and run on CPU inside a test; the architecture is unchanged.
    cfg.scratch.hidden_dim = 64
    cfg.scratch.num_heads = 4
    cfg.scratch.dit_blocks = 2
    cfg.model.cond_dim = 32
    cfg.model.cond_layers = 2
    cfg.trainer.val_timesteps = 2
    return cfg


def scratch_batch(cfg, frames=40, batch=2):
    return {
        "sample_id": [f"t{i}" for i in range(batch)],
        "latent": torch.randn(batch, frames, int(cfg.ace.latent_channels)),
        "roll": torch.rand(batch, total_roll_channels(cfg), frames),
        "mask": torch.ones(batch, frames, dtype=torch.bool),
        "seconds": torch.full((batch,), frames / 25.0),
        "start_seconds": torch.zeros(batch),
    }


@pytest.fixture
def module(tmp_path, monkeypatch):
    from p2pa.model import P2PAModule

    return P2PAModule(scratch_cfg(tmp_path, monkeypatch))


def test_no_pretrained_weights_are_loaded(module):
    from p2pa.scratch import ScratchBackbone

    assert isinstance(module.backbone, ScratchBackbone)
    assert module.scratch and module._remote is None
    assert not hasattr(module, "silence_latent"), "scratch has no ACE-Step silence convention"
    assert isinstance(module.roll_encoder.out, torch.nn.Identity)


def test_everything_is_trainable_from_the_first_step(module):
    total, conditioning = module.trainable_count()
    assert module.freeze_steps == 0 and not module.backbone_frozen
    assert total == sum(p.numel() for p in module.parameters())
    assert conditioning > 0 and total > conditioning


def test_the_released_size(tmp_path, monkeypatch):
    from p2pa.scratch import ScratchBackbone

    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        cfg = compose(config_name="config", overrides=["backbone=scratch"])
    backbone = sum(p.numel() for p in ScratchBackbone(cfg).parameters())
    assert backbone == 220_003_904, "the from-scratch checkpoints' transformer has 220,003,904"


def test_the_condition_is_channels_first_at_the_seam(module):
    """`dataset.py` stays time-major so both backbones train on identical bytes."""
    cfg = module.cfg
    batch = scratch_batch(cfg)
    cond = module.build_conditioning(batch)
    assert cond["aligned_cond"].shape == (2, int(cfg.model.cond_dim), batch["latent"].shape[1])


def test_the_flow_convention_is_this_backbone_s_own(module):
    """t=0 noise, t=1 data, the opposite of ACE-Step's.

    A caller's timestep is in ACE-Step's convention, so t=0 there is data. With the model's
    prediction forced to the true velocity, the loss must then be exactly zero; had the flip been
    lost, the interpolant would be pure noise and the target the same, and the loss would not
    vanish.
    """
    backbone = module.backbone
    latent = torch.randn(2, 8, int(module.cfg.ace.latent_channels))
    noise = torch.randn_like(latent)
    seen = {}

    def oracle(noisy, timestep, cond, mask=None):
        seen["noisy"], seen["t"] = noisy, timestep
        return backbone.to_backbone(latent) - backbone.to_backbone(noise)

    backbone.denoise = oracle
    loss = backbone.loss(latent, {}, timestep=torch.zeros(2), noise=noise)
    assert torch.allclose(seen["t"], torch.ones(2))
    assert torch.allclose(seen["noisy"], backbone.to_backbone(latent))
    assert float(loss) == 0.0


def test_the_whole_stack_is_connected_after_one_step(module):
    """`final_proj` is zero-init, so step 0 gives the encoder no gradient — by design.

    A zero-init output self-corrects on the first update; a dead conditioning path never does.
    Both look identical at step 0, so the test has to take a step.
    """
    batch = scratch_batch(module.cfg)
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)

    module.flow_loss(batch).backward()
    encoder = sum(
        float(p.grad.abs().sum()) for p in module.roll_encoder.parameters() if p.grad is not None
    )
    assert encoder == 0.0, "unexpected: something upstream of a zero-init output has a gradient"
    assert float(module.backbone.transformer.final_proj.weight.grad.abs().sum()) > 0
    optimizer.step()

    module.zero_grad(set_to_none=True)
    module.flow_loss(batch).backward()
    encoder = sum(
        float(p.grad.abs().sum()) for p in module.roll_encoder.parameters() if p.grad is not None
    )
    assert encoder > 0, "the roll encoder never becomes connected"


def test_the_roll_is_dropped_for_guidance_only_in_training(module):
    cond = {"aligned_cond": torch.ones(64, 3, 5)}
    module.backbone.cond_dropout = 0.5
    torch.manual_seed(0)
    dropped = module.backbone.drop(cond, training=True)["aligned_cond"]
    zeroed = int((dropped.flatten(1).abs().sum(1) == 0).sum())
    assert 10 < zeroed < 54
    assert module.backbone.drop(cond, training=False) is cond


def test_the_loss_ignores_padded_frames(module):
    batch = scratch_batch(module.cfg, frames=40)
    batch["mask"][:, 20:] = False
    module.eval()
    timestep, noise = torch.full((2,), 0.5), torch.randn_like(batch["latent"])
    cond = module.build_conditioning(batch)
    before = module.flow_loss(batch, timestep=timestep, noise=noise, cond=cond)
    batch["latent"][:, 20:] += 100.0
    after = module.flow_loss(batch, timestep=timestep, noise=noise, cond=cond)
    assert torch.allclose(before, after)


def test_validation_is_deterministic(module):
    batch = scratch_batch(module.cfg)
    module.eval()
    with torch.no_grad():
        first = module.validation_step(batch, 0)
        second = module.validation_step(batch, 0)
    assert torch.allclose(first, second)


def test_both_parameter_groups_share_the_scratch_rate(module):
    optimizer = module.configure_optimizers()["optimizer"]
    rates = {group["name"]: group["initial_lr"] for group in optimizer.param_groups}
    assert rates == {"cond": 1.0e-4, "backbone": 1.0e-4}


@pytest.mark.parametrize(
    "override,expect",
    [
        ({"scratch.mmdit_blocks": "2"}, "must be 0"),
        ({"model.guidance": "7.0"}, "linear guidance"),
        ({"scratch.cond_dropout": "1.0"}, "cond_dropout"),
        ({"model.backbone": "lora"}, "must be one of"),
    ],
)
def test_inapplicable_settings_are_refused(tmp_path, monkeypatch, override, expect):
    with pytest.raises(ValueError, match=expect):
        validate_config(scratch_cfg(tmp_path, monkeypatch, **override))


def _move_off_zero(module):
    # `final_proj` is zero-init, so an untrained model predicts zero velocity and returns its own
    # noise for every roll. Move it off zero, as one optimizer step would.
    torch.manual_seed(0)
    with torch.no_grad():
        projection = module.backbone.transformer.final_proj
        projection.weight.add_(torch.randn_like(projection.weight) * 0.05)


def test_sampling_is_deterministic_and_depends_on_the_roll(module):
    from p2pa.sample import sample_latent

    module.eval()
    _move_off_zero(module)
    fixed = scratch_batch(module.cfg, frames=40)
    channels = module.roll_encoder.stem.in_channels
    quiet = {**fixed, "roll": torch.zeros(2, channels, 40)}
    busy = {**fixed, "roll": torch.rand(2, channels, 40)}

    first = sample_latent(module, quiet, steps=2, guidance=1.0, seed=7)
    again = sample_latent(module, quiet, steps=2, guidance=1.0, seed=7)
    other = sample_latent(module, busy, steps=2, guidance=1.0, seed=7)
    assert first.shape == fixed["latent"].shape            # [B, T, C], time-major
    assert torch.allclose(first, again), "sampling is not deterministic at a fixed seed"
    assert not torch.allclose(first, other, atol=1e-5), "the roll does not reach the decoder"


def test_guidance_changes_the_sample(module):
    from p2pa.sample import sample_latent

    module.eval()
    _move_off_zero(module)
    batch = scratch_batch(module.cfg, frames=40)
    plain = sample_latent(module, batch, steps=2, guidance=1.0, seed=0)
    guided = sample_latent(module, batch, steps=2, guidance=2.0, seed=0)
    assert torch.isfinite(guided).all()
    assert not torch.allclose(plain, guided)


def test_blending_is_applied_on_the_path(module):
    """Overlapping windows are reconciled after every step, not cross-faded at the end."""
    from p2pa.sample import WindowBlend, sample_latent

    module.eval()
    frames, total = 40, 100
    starts = [0, 30, total - frames]          # deliberately ragged, as a real song's tail is
    batch = scratch_batch(module.cfg, frames=frames, batch=len(starts))
    blend = WindowBlend(starts, frames, total, torch.device("cpu"))
    calls = []

    def counting(x):
        calls.append(x.shape)
        return blend(x)

    generated = sample_latent(module, batch, steps=3, guidance=1.0, seed=0, blend=counting)
    assert len(calls) == 3, f"blend ran {len(calls)} times, expected once per solver step"
    assert generated.shape == (len(starts), frames, int(module.cfg.ace.latent_channels))


def test_the_ema_shadow_is_what_a_loaded_checkpoint_samples_with(module, tmp_path):
    from p2pa.callbacks import EMACallback
    from p2pa.model import apply_ema

    ema = EMACallback(decay=0.5)
    ema._seed(module)
    for value in ema.shadow.values():
        value.fill_(0.25)
    path = tmp_path / "last.ckpt"
    torch.save({"callbacks": {"EMACallback": ema.state_dict()}}, path)

    assert apply_ema(module, str(path)) == len(dict(module.named_parameters()))
    assert all(torch.all(p == 0.25) for p in module.parameters())
