from __future__ import annotations

import pytest

from p2pa.config import latent_fps, latent_frames, validate_config


def test_latent_rate_is_exactly_25(cfg):
    assert latent_fps(cfg) == 25.0
    assert float(cfg.roll.frames_per_second) == 25.0


def test_window_lengths_are_whole_patches(cfg):
    # `proj_in` is a stride-2 convolution and the model zero-pads a ragged tail before it, which
    # would put frames describing nothing into the loss. Windows are quantised instead.
    assert latent_frames(cfg, 30.0) == 750
    assert latent_frames(cfg, 10.0) == 250
    assert latent_frames(cfg, 10.04) % int(cfg.ace.patch_size) == 0
    assert latent_frames(cfg, 0.001) == int(cfg.ace.patch_size)


def test_multiple_devices_are_refused(cfg):
    cfg.trainer.devices = [0, 1]
    with pytest.raises(ValueError, match="exactly one GPU"):
        validate_config(cfg)
    cfg.trainer.devices = "auto"
    with pytest.raises(ValueError, match="exactly one GPU"):
        validate_config(cfg)


def test_roll_rate_must_match_the_latent_rate(cfg):
    cfg.roll.frames_per_second = 24.0
    with pytest.raises(ValueError, match="silently misaligns"):
        validate_config(cfg)


def test_the_roll_must_arrive_at_the_latent_rate(cfg):
    cfg.roll.oversample = 4
    cfg.roll.frames_per_second = 100.0
    with pytest.raises(ValueError, match="oversample must be 1"):
        validate_config(cfg)


def test_the_kernel_must_be_odd(cfg):
    cfg.model.cond_kernel = 4
    with pytest.raises(ValueError, match="must be odd"):
        validate_config(cfg)


def test_the_freeze_window_must_end(cfg):
    cfg.finetune.freeze_steps = int(cfg.trainer.max_steps)
    with pytest.raises(ValueError, match="never ends"):
        validate_config(cfg)


def test_the_length_ramp_must_finish_inside_the_run(cfg):
    cfg.trainer.max_steps = int(cfg.length.hold_steps) + 5_000
    with pytest.raises(ValueError, match="reaches 30s only at step"):
        validate_config(cfg)


def test_listening_sample_counts_must_not_be_negative(cfg):
    cfg.trainer.kong_sample_items = -1
    with pytest.raises(ValueError, match="sample counts"):
        validate_config(cfg)


def test_default_listening_panel_is_two_slow_fast_pairs(cfg):
    assert int(cfg.trainer.sample_items) == 2
    assert int(cfg.trainer.kong_sample_items) == 2
    assert list(cfg.trainer.kong_sample_bpms) == [63.8, 130.4]
