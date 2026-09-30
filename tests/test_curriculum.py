"""The window-length schedule: a ceiling that holds, then rises with the optimizer step.

20 s for the first 100,000 steps, then +5 s every 10,000 until 30 s. These tests pin the
schedule, the widening (not sliding) range, and the one property that keeps runs comparable:
validation never follows the curriculum.
"""

from __future__ import annotations

import contextlib
import io
import re

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from p2pa import train
from p2pa.config import TrainConfig, config_dir, latent_fps
from p2pa.dataset import LengthBatchSampler, P2PADataModule, curriculum_ceiling

ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)


def composed(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        return compose(config_name="config", overrides=list(overrides))


def sampler(cfg, step_box, **kwargs):
    settings = dict(
        min_seconds=float(cfg.length.min_seconds),
        max_seconds=float(cfg.length.max_seconds),
        fps=latent_fps(cfg),
        patch=int(cfg.ace.patch_size),
        batch_size=8,
        get_step=lambda: step_box["v"],
        ramp_seconds=float(cfg.length.ramp_seconds),
        ramp_every_steps=int(cfg.length.ramp_every_steps),
        hold_steps=int(cfg.length.hold_steps),
    )
    settings.update(kwargs)
    return LengthBatchSampler(100, **settings)


def validation_sampler(cfg):
    module = P2PADataModule(cfg)
    module.train_ds = module.val_ds = type("D", (), {"__len__": lambda self: 100})()
    return (
        module._sampler(module.train_ds, validation=False, seed=0, repeats=1),
        module._sampler(module.val_ds, validation=True, seed=1, repeats=1),
    )


@pytest.mark.parametrize(
    "step,expected",
    [(0, 20.0), (99_999, 20.0), (100_000, 25.0), (109_999, 25.0), (110_000, 30.0),
     (500_000, 30.0)],
)
def test_the_ceiling_follows_the_schedule(step, expected):
    """The first increment lands *at* the end of the hold: step 99,999 is 20 s, 100,000 is 25 s."""
    assert curriculum_ceiling(composed(), step) == expected


def test_the_ceiling_clamps_and_never_exceeds_max_seconds():
    cfg = composed()
    for step in range(0, 1_000_000, 7_919):
        assert curriculum_ceiling(cfg, step) <= float(cfg.length.max_seconds)


def test_a_zero_hold_is_the_plain_ramp():
    cfg = composed("length.hold_steps=0", "length.min_seconds=10", "length.ramp_every_steps=25000")
    assert curriculum_ceiling(cfg, 0) == 10.0
    assert curriculum_ceiling(cfg, 24_999) == 10.0
    assert curriculum_ceiling(cfg, 25_000) == 15.0
    assert curriculum_ceiling(cfg, 100_000) == 30.0


def test_the_range_widens_rather_than_sliding():
    """The floor never moves: a model that has learned 20 s windows keeps being asked for them."""
    cfg = composed()
    box = {"v": 200_000}
    drawn = [target for batch in sampler(cfg, box, repeats=40) for _, target in batch]
    assert min(drawn) < 22.0
    assert max(drawn) > 28.0


def test_every_drawn_length_is_a_whole_number_of_patches():
    cfg = composed()
    box = {"v": 0}
    fps, patch = latent_fps(cfg), int(cfg.ace.patch_size)
    s = sampler(cfg, box, repeats=6)
    for step in (0, 100_000, 105_000, 200_000):
        box["v"] = step
        for batch in s:
            assert round(batch[0][1] * fps) % patch == 0, (step, batch[0][1])


def test_a_batch_still_carries_exactly_one_length():
    cfg = composed()
    box = {"v": 120_000}
    for batch in sampler(cfg, box, repeats=3):
        assert len({target for _, target in batch}) == 1


def test_validation_does_not_follow_the_curriculum():
    """`val/loss` must measure one fixed length, or it is not comparable across the run."""
    cfg = composed()
    training, validation = validation_sampler(cfg)
    assert training.get_step is not None and training.ramp_seconds > 0
    assert validation.get_step is None and validation.ramp_seconds == 0.0
    assert validation.ceiling() == float(cfg.length.max_seconds)


def test_the_sampler_and_the_listening_export_agree():
    """One ramp, two consumers, and they must not drift.

    `LengthBatchSampler.ceiling` draws training lengths from its own fields; `curriculum_ceiling`
    answers the same question from a config, and is what `SampleCallback` uses to decide how long
    a listening export should be.
    """
    cfg = composed()
    box = {"v": 0}
    s = sampler(cfg, box)
    for step in (0, 1, 99_999, 100_000, 105_000, 109_999, 110_000, 500_000, 10**7):
        box["v"] = step
        assert s.ceiling() == curriculum_ceiling(cfg, step), step


def test_the_banner_reports_the_step_the_ceiling_is_actually_reached():
    """The banner asks the schedule rather than restating it, so it cannot contradict the run."""
    cfg = composed()
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        train.banner(cfg, _StubModule())
    match = re.search(r"30s from step (\d+)", printed.getvalue())
    assert match, printed.getvalue()
    reached = int(match.group(1))
    assert reached == 110_000
    assert curriculum_ceiling(cfg, reached) == float(cfg.length.max_seconds)
    assert curriculum_ceiling(cfg, reached - 1) < float(cfg.length.max_seconds)


class _StubModule:
    """Enough of `P2PAModule` for the banner; building a real one is 2.4B parameters of no use."""

    def trainable_count(self):
        return 0, 0

    def parameters(self):
        return iter(())
