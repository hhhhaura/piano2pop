"""`auto_resume` must continue a run, never silently merge two of them.

A change that alters a tensor shape fails loudly on its own. Every setting that changes what a run
*means* without changing a shape (the length schedule, the augmentation weights, the step budget)
would resume perfectly cleanly and produce a run trained half under one regime and half under
another, with nothing in the log to say where the join was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from p2pa.config import TrainConfig, config_dir
from p2pa.train import RESUME_CRITICAL, check_resume_compatible

ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)


def composed(**overrides):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        return compose(config_name="config",
                       overrides=[f"{k}={v}" for k, v in overrides.items()])


def checkpoint_for(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hyper_parameters": OmegaConf.to_container(cfg, resolve=True)}, path)
    return path


def test_an_identical_config_resumes(tmp_path):
    cfg = composed()
    check_resume_compatible(cfg, checkpoint_for(cfg, tmp_path / "c" / "last.ckpt"))


@pytest.mark.parametrize(
    "overrides,expect",
    [
        ({"length.hold_steps": "0"}, "length.hold_steps"),
        ({"finetune.freeze_steps": "0"}, "finetune.freeze_steps"),
        ({"setting": "var"}, "augment"),
        ({"model.cond_dim": "256"}, "model.cond_dim"),
        # Lowering the budget is still an accident: the cosine would be rebuilt shorter than the
        # steps already taken. Only raising it is a deliberate act — see the test below.
        ({"trainer.max_steps": "100000"}, "max_steps"),
    ],
)
def test_a_changed_regime_is_refused(tmp_path, overrides, expect):
    before = composed()
    path = checkpoint_for(before, tmp_path / "c" / "last.ckpt")
    with pytest.raises(SystemExit) as raised:
        check_resume_compatible(composed(**overrides), path)
    message = str(raised.value)
    assert expect in message
    assert "exp_name=" in message, "the error must say how to proceed, not only what is wrong"


def test_raising_max_steps_is_allowed_but_never_silent(tmp_path, capsys):
    """Training a finished run further is the one critical change that is a decision, not a slip.

    `resume_main` advertises `trainer.max_steps` as something that legitimately varies between
    launches, and there is no other supported path to it: `auto_resume=false` starts from random
    init and a new `exp_name` cannot inherit the weights.

    It must say so, because the consequence is not "the same run, longer". The schedule is a
    cosine to zero at `max_steps`, so a run that ended at 500k under a 500k schedule sits at
    exactly LR 0, and resuming it under 750k restarts the rate partway up the new cosine. That is
    a warm restart, and the discontinuity it leaves in the loss curve should be one the reader
    knows was intended.
    """
    before = composed()
    path = checkpoint_for(before, tmp_path / "c" / "last.ckpt")
    check_resume_compatible(composed(**{"trainer.max_steps": "750000"}), path)
    printed = capsys.readouterr().out
    assert "500000 -> 750000" in printed
    assert "warm restart" in printed


def test_a_harmless_change_still_resumes(tmp_path):
    """Batch size, workers and device change throughput, not what the run is."""
    before = composed()
    path = checkpoint_for(before, tmp_path / "c" / "last.ckpt")
    after = composed(**{"data.batch_size": "2", "data.num_workers": "1",
                        "trainer.devices": "[1]"})
    check_resume_compatible(after, path)


def test_a_checkpoint_without_hyperparameters_is_allowed(tmp_path):
    path = tmp_path / "c" / "last.ckpt"
    path.parent.mkdir(parents=True)
    torch.save({"state_dict": {}}, path)
    check_resume_compatible(composed(), path)


def test_the_watched_set_covers_every_augmentation_weight():
    """`setting` is the experiment axis, and every weight it sets is watched."""
    for key in ("window.segment_augment", "augment.baseline_probability",
                "augment.picogen_probability", "augment.normal", "augment.chord_thin",
                "augment.octave_move", "augment.octave_shift"):
        assert key in RESUME_CRITICAL


@pytest.mark.parametrize("setting", ["base", "rule", "var", "pico", "final"])
def test_a_paper_run_resumes_under_its_release_setting(tmp_path, setting):
    """Its checkpoint recorded the older `phase.*` schema; it must still compare as equal."""
    recorded = OmegaConf.load(Path(__file__).parent / "fixtures" / "paper_runs" / f"{setting}.yaml")
    check_resume_compatible(composed(setting=setting), checkpoint_for(recorded, tmp_path / "c.ckpt"))


def test_the_message_offers_the_values_that_would_let_it_continue(tmp_path):
    """The remedy anyone actually wants, and the first version of this message omitted.

    Offering only "rename", "delete" or "disable the check" framed a recoverable situation as a
    choice between losing a 96,000-step run and losing the guard. Putting the old values back is
    almost always the right answer, and it should not have to be worked out from the diff.
    """
    before = composed(**{"length.min_seconds": "15"})
    path = checkpoint_for(before, tmp_path / "c" / "last.ckpt")
    with pytest.raises(SystemExit) as raised:
        check_resume_compatible(composed(), path)
    message = str(raised.value)
    assert "continue it as it was" in message
    assert "length.min_seconds=15" in message, message


def test_several_changed_keys_are_all_offered(tmp_path):
    before = composed(**{"length.min_seconds": "15", "finetune.freeze_steps": "500"})
    path = checkpoint_for(before, tmp_path / "c" / "last.ckpt")
    with pytest.raises(SystemExit) as raised:
        check_resume_compatible(composed(), path)
    message = str(raised.value)
    assert "length.min_seconds=15" in message and "finetune.freeze_steps=500" in message


def test_reading_the_config_does_not_materialise_the_tensors(tmp_path):
    """A `full` checkpoint is 27 GB and this function wants a small dict.

    Loading the whole thing cost 27 GB of resident memory per run, and because Lightning then
    loads it again for the real restore, a paired resume tripped the RSS guard at 66 GB before
    training had done anything. The guard was right; this was the reason it fired.
    """
    import psutil

    from p2pa.train import stored_hyperparameters

    path = tmp_path / "big.ckpt"
    torch.save(
        {
            "hyper_parameters": {"length": {"min_seconds": 5.0}, "phase": {"name": "full"}},
            "state_dict": {f"w{i}": torch.randn(32, 1024, 1024) for i in range(4)},
            "optimizer_states": [{"state": {0: {"exp_avg": torch.randn(32, 1024, 1024)}}}],
        },
        path,
    )
    on_disk = path.stat().st_size / 1024**3
    assert on_disk > 0.5, "the fixture must be big enough for the difference to be visible"

    process = psutil.Process()
    before = process.memory_info().rss
    hyperparameters = stored_hyperparameters(path)
    grew = (process.memory_info().rss - before) / 1024**3

    assert hyperparameters["length"]["min_seconds"] == 5.0
    assert hyperparameters["phase"]["name"] == "full"
    assert grew < on_disk / 4, (
        f"reading a {on_disk:.1f} GB checkpoint's config grew RSS by {grew:.2f} GB — the tensors "
        "are being materialised"
    )


def test_the_cheap_read_agrees_with_a_full_load(tmp_path):
    """Cheap is worthless if it is also wrong."""
    from p2pa.train import stored_hyperparameters

    path = tmp_path / "c.ckpt"
    config = composed(**{"length.min_seconds": "15"})
    torch.save(
        {
            "hyper_parameters": OmegaConf.to_container(config, resolve=True),
            "state_dict": {"w": torch.randn(16, 16)},
        },
        path,
    )
    cheap = stored_hyperparameters(path)
    whole = torch.load(path, map_location="cpu", weights_only=False)["hyper_parameters"]
    assert cheap == whole


def test_a_checkpoint_with_no_config_reads_as_none(tmp_path):
    from p2pa.train import stored_hyperparameters

    path = tmp_path / "bare.ckpt"
    torch.save({"state_dict": {"w": torch.randn(4)}}, path)
    assert not stored_hyperparameters(path)


def test_resume_reads_a_runs_own_resolved_config(tmp_path, monkeypatch):
    """A run has to be relaunchable from what it recorded, not from what was typed to start it.

    `check_resume_compatible` refuses to continue a checkpoint under a different config, so the
    overrides a run was launched with are load-bearing for its whole life. They used to live only
    in the shell script that happened to launch it; `p2pa-resume` reads them from the run instead.
    """
    from omegaconf import OmegaConf

    from p2pa import train as train_module

    cfg = composed()
    cfg.exp_name = "recorded_run"
    cfg.trainer.max_steps = 250000
    run_dir = tmp_path / "recorded_run"
    run_dir.mkdir()
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")

    seen = {}
    monkeypatch.setattr(train_module, "train", lambda cfg: seen.update(cfg=cfg))
    monkeypatch.setattr(
        "sys.argv", ["p2pa-resume", str(run_dir), "trainer.max_steps=500000", "trainer.devices=[0]"]
    )
    train_module.resume_main()

    resumed = seen["cfg"]
    assert resumed.exp_name == "recorded_run", "the run's own record supplies everything else"
    assert int(resumed.trainer.max_steps) == 500000, "a trailing override wins over the record"


def test_resume_refuses_a_run_directory_that_never_trained(tmp_path, monkeypatch):
    from p2pa import train as train_module

    monkeypatch.setattr("sys.argv", ["p2pa-resume", str(tmp_path)])
    with pytest.raises(SystemExit, match="resolved_config.yaml"):
        train_module.resume_main()
