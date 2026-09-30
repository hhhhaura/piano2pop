from __future__ import annotations

import pytest
import torch

from p2pa.config import latent_fps
from p2pa.dataset import LengthBatchSampler, collate


def sampler(cfg, **kwargs):
    settings = dict(
        min_seconds=float(cfg.length.min_seconds),
        max_seconds=float(cfg.length.max_seconds),
        fps=latent_fps(cfg),
        patch=int(cfg.ace.patch_size),
        batch_size=4,
        seed=0,
        repeats=1,
    )
    settings.update(kwargs)
    return LengthBatchSampler(20, **settings)


def test_a_batch_carries_exactly_one_length(cfg):
    for batch in sampler(cfg):
        assert len({target for _, target in batch}) == 1


def test_random_mode_varies_across_batches(cfg):
    lengths = {batch[0][1] for batch in sampler(cfg, repeats=8)}
    assert len(lengths) > 1
    assert all(10.0 <= value <= 30.0 for value in lengths)


def test_fixed_mode_never_varies(cfg):
    lengths = {batch[0][1] for batch in sampler(cfg, fixed=True, repeats=8)}
    assert lengths == {30.0}


def test_every_drawn_length_is_a_whole_number_of_patches(cfg):
    fps, patch = latent_fps(cfg), int(cfg.ace.patch_size)
    for batch in sampler(cfg, repeats=20):
        frames = round(batch[0][1] * fps)
        assert frames % patch == 0


def test_repeats_give_a_song_several_independent_windows(cfg):
    items = [item for batch in sampler(cfg, repeats=5) for item in batch]
    assert len(items) == 100
    first = [target for index, target in items if index == 0]
    assert len(first) == 5


def test_collate_refuses_a_mixed_batch(cfg):
    def item(frames):
        return {
            "sample_id": "x",
            "latent": torch.zeros(frames, 64),
            "roll": torch.zeros(8, frames * 4),
            "seconds": frames / 25.0,
            "start_seconds": 0.0,
            "prompt": torch.zeros(7, 1024),
            "prompt_mask": torch.ones(7, dtype=torch.bool),
        }

    batched = collate([item(20), item(20)])
    assert batched["latent"].shape == (2, 20, 64)
    assert bool(batched["mask"].all())
    with pytest.raises(RuntimeError, match="homogeneous"):
        collate([item(20), item(18)])


