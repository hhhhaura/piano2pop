from __future__ import annotations

import torch

from p2pa.cond import build_cond_encoder
from p2pa.pianoroll import total_roll_channels


def test_channel_count_is_sustain_and_onset_per_pitch(cfg):
    assert total_roll_channels(cfg) == 2 * (108 - 21 + 1) == 176


def test_output_is_one_frame_per_latent_frame(tiny_cfg):
    encoder = build_cond_encoder(tiny_cfg)
    out = encoder(torch.randn(2, total_roll_channels(tiny_cfg), 40))
    assert out.shape == (2, 40, int(tiny_cfg.ace.latent_channels))


def test_an_untrained_encoder_emits_exactly_the_silence_latent(tiny_cfg):
    """The whole zero-init story: before training the condition is the checkpoint's own "no
    reference audio" latent, so the model is bit-identical to stock ACE-Step text2music."""
    silence = torch.full((1, 64, int(tiny_cfg.ace.latent_channels)), 0.25)
    encoder = build_cond_encoder(tiny_cfg, silence)
    out = encoder(torch.randn(3, total_roll_channels(tiny_cfg), 12))
    assert torch.equal(out, silence[:, :12].expand(3, -1, -1))


def test_the_stem_is_centred_on_its_frame(tiny_cfg):
    """An odd kernel with `kernel // 2` padding reaches equally far either side of a frame.

    An off-centre stem would put every condition a fraction of a kernel away from the audio it
    describes, and nothing downstream has a shape to complain about.
    """
    encoder = build_cond_encoder(tiny_cfg)
    kernel = int(tiny_cfg.model.cond_kernel)
    roll = torch.zeros(1, total_roll_channels(tiny_cfg), 20)
    before = encoder.stem(roll)
    roll[..., 10] = 1.0
    changed = ((before - encoder.stem(roll)).abs().sum(dim=1)[0] > 0).nonzero().flatten().tolist()
    assert changed == list(range(10 - kernel // 2, 10 + kernel // 2 + 1))


def test_the_gradient_reaches_the_encoder(tiny_cfg):
    """Against a real target, not against the encoder's own output.

    A zero-init projection makes `output.square()` stationary at init — its gradient is exactly
    zero because the output is exactly zero — so an objective shaped like that would pass whether
    or not the encoder were connected to anything. Training's objective is a distance to something
    else, which is what this uses.
    """
    encoder = build_cond_encoder(tiny_cfg)
    roll = torch.randn(2, total_roll_channels(tiny_cfg), 10)
    target = torch.randn(2, 10, int(tiny_cfg.ace.latent_channels))
    (encoder(roll) - target).square().mean().backward()
    # The output projection is zero-init, so it is the first thing to move; the stem receives a
    # gradient only once it has, which is exactly why the freeze window trains the encoder alone.
    assert encoder.out.weight.grad.abs().sum() > 0
    assert encoder.stem.weight.grad is not None
