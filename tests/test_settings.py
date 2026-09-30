"""Each release setting must compose to exactly the configuration its paper run trained under.

`tests/fixtures/paper_runs/<setting>.yaml` is the `resolved_config.yaml` each 500k-step run wrote
when it started, recorded under the older schema (`phase.*`, LoRA and caption keys at their off
values). Read through `migrate_config` it has to equal what `setting=<name>` composes to today, key
for key — otherwise the released code does not reproduce the released weights.
`scratch_<setting>.yaml` is the same check for the two from-scratch runs, against
`setting=<name> backbone=scratch`.

`_self_` sits first in the defaults list for the same reason: `configs/config.yaml` carries full
`augment:` and `window:` blocks, and with `_self_` last those blocks were merged over the group,
so `setting=var` silently resolved back to `base`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from p2pa.config import TrainConfig, config_dir, migrate_config, validate_config

ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)

PAPER_RUNS = Path(__file__).parent / "fixtures" / "paper_runs"
SETTINGS = ("base", "rule", "var", "pico", "final")
# The from-scratch runs that were trained, with the overrides each was launched with; the other
# settings compose and validate only. scratch_conv_final drew the original transcription 0.15 and
# PiCoGen 0.6, not the finetune `final`'s 0.3 and 0.4.
SCRATCH_RUNS = {
    "pico": [],
    "final": ["augment.baseline_probability=0.15", "augment.picogen_probability=0.6"],
}
# Properties of the machine or of the launch, not of the model.
IGNORED = {"exp_name", "data.soundfont", "trainer.devices"}


def composed(setting: str, backbone: str = "acestep", extra: list[str] = ()):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        return compose(
            config_name="config",
            overrides=[f"setting={setting}", f"backbone={backbone}", *extra],
        )


def flat(node, prefix: str = "") -> dict:
    out = {}
    for key, value in node.items():
        name = f"{prefix}{key}"
        out.update(flat(value, f"{name}.") if isinstance(value, dict) else {name: value})
    return out


@pytest.mark.parametrize(
    "setting,backbone",
    [(name, "acestep") for name in SETTINGS] + [(name, "scratch") for name in SCRATCH_RUNS],
)
def test_each_setting_reproduces_its_paper_run(setting, backbone, monkeypatch, tmp_path):
    (tmp_path / "data" / "p2pdata").mkdir(parents=True)
    monkeypatch.setenv("P2PA_DATA_ROOT", str(tmp_path))
    now = composed(setting, backbone, SCRATCH_RUNS[setting] if backbone == "scratch" else [])
    validate_config(now)
    fixture = setting if backbone == "acestep" else f"scratch_{setting}"
    paper = migrate_config(OmegaConf.load(PAPER_RUNS / f"{fixture}.yaml"))

    now = flat(OmegaConf.to_container(now, resolve=True))
    paper = flat(OmegaConf.to_container(paper, resolve=True))
    assert set(now) == set(paper)
    different = {
        key: (paper[key], now[key])
        for key in now
        if key not in IGNORED and paper[key] != now[key]
    }
    assert not different, f"setting={setting} backbone={backbone} differs: {different}"


@pytest.mark.parametrize("setting", SETTINGS)
def test_every_setting_composes_with_the_scratch_backbone(setting, monkeypatch, tmp_path):
    (tmp_path / "data" / "p2pdata").mkdir(parents=True)
    monkeypatch.setenv("P2PA_DATA_ROOT", str(tmp_path))
    cfg = composed(setting, "scratch")
    validate_config(cfg)
    assert cfg.model.backbone == "scratch" and cfg.data.batch_size == 16
    assert cfg.augment == composed(setting).augment


def test_the_settings_are_all_distinct():
    signatures = set()
    for name in SETTINGS:
        cfg = composed(name)
        signatures.add(str(OmegaConf.to_container(cfg.augment)) + str(cfg.window.segment_augment)
                       + str(list(cfg.data.source.excluded_variant_programs)))
    assert len(signatures) == len(SETTINGS)
