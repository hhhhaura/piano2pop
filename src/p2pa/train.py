"""Training entrypoint. One command; `setting` picks the conditioning augmentation.

    p2pa-train setting=var exp_name=var trainer.devices=[0]

Runs are namespaced `runs/<exp_name>/`, so settings live side by side and none can overwrite
another's checkpoints. There is no second phase and no checkpoint handoff: the backbone freeze
happens inside the run, at `finetune.freeze_steps` — see `UnfreezeBackbone`. `backbone=scratch`
trains a from-scratch DiT instead (`scratch.py`).
"""

from __future__ import annotations

import math
import pathlib

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .config import (
    TrainConfig,
    config_dir,
    data_root,
    latent_frames,
    migrate_config,
    resolve_path,
    validate_config,
)

ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)


def banner(cfg: DictConfig, module) -> None:
    from .dataset import curriculum_ceiling  # lazy, as P2PADataModule is: torch + numpy

    total, conditioning = module.trainable_count()
    everything = sum(p.numel() for p in module.parameters())
    # Asked of `curriculum_ceiling` rather than recomputed, so the banner cannot disagree with the
    # sampler about when full length arrives.
    stages = math.ceil(
        (float(cfg.length.max_seconds) - float(cfg.length.min_seconds))
        / max(float(cfg.length.ramp_seconds), 1e-9)
    )
    every, hold = int(cfg.length.ramp_every_steps), int(cfg.length.hold_steps)
    reached = next(
        (
            step
            for step in (hold + k * every for k in range(stages + 2))
            if curriculum_ceiling(cfg, step) >= float(cfg.length.max_seconds)
        ),
        hold + stages * every,
    )
    scratch = cfg.scratch
    from_scratch = str(cfg.model.backbone) == "scratch"
    if from_scratch:
        model = (f"from scratch: {scratch.dit_blocks}-block DiT, hidden {scratch.hidden_dim}, "
                 f"EMA {scratch.ema_decay:g}")
        entry = "condition summed into the residual stream as aligned_cond"
    else:
        model = f"{cfg.ace.model_name} (full finetune, float32)"
        entry = (f"condition enters as src_latents, channel-concatenated at proj_in "
                 f"({cfg.ace.in_channels} = 3 x {cfg.ace.latent_channels})")
    print(
        f"[p2pa] {cfg.exp_name}  backbone={model}\n"
        f"       length: {cfg.length.min_seconds:g}s for {hold} steps, then "
        f"+{cfg.length.ramp_seconds:g}s every {every} steps; {cfg.length.max_seconds:g}s from "
        f"step {reached} ({latent_frames(cfg, float(cfg.length.max_seconds))} latent frames)\n"
        f"       conditioning: conv roll encoder {cfg.model.cond_layers}L@{cfg.model.cond_dim} "
        f"k{cfg.model.cond_kernel}, segment_augment={cfg.window.segment_augment} "
        f"(baseline {cfg.augment.baseline_probability:g}, "
        f"picogen {cfg.augment.picogen_probability:g})\n"
        f"       {entry}\n"
        f"       {total / 1e6:.1f}M trainable of {everything / 1e6:.1f}M "
        f"({conditioning / 1e6:.1f}M in the roll encoder)\n"
        f"       backbone frozen for the first "
        f"{0 if from_scratch else cfg.finetune.freeze_steps} of "
        f"{cfg.trainer.max_steps} steps",
        flush=True,
    )


# Settings that change what a run *is*, not merely how fast it gets there. Resuming across any of
# them silently produces a model trained under two different regimes, with nothing in the log to
# say so.
RESUME_CRITICAL = (
    "model.backbone",
    "finetune.freeze_steps",
    "scratch.hidden_dim",
    "scratch.num_heads",
    "scratch.dit_blocks",
    "scratch.cond_dropout",
    "length.min_seconds",
    "length.max_seconds",
    "length.ramp_seconds",
    "length.ramp_every_steps",
    "length.hold_steps",
    "model.cond_dim",
    "model.cond_layers",
    "model.cond_kernel",
    "roll.oversample",
    "window.segment_augment",
    "augment.baseline_probability",
    "augment.picogen_probability",
    "augment.normal",
    "augment.chord_thin",
    "augment.octave_move",
    "augment.octave_shift",
    "trainer.max_steps",
)


def _lookup(cfg, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if node is None or part not in node:
            return None
        node = node[part]
    return node



def newest_valid_checkpoint(directory: pathlib.Path):
    """The most recent checkpoint that is actually readable, preferring `last.ckpt`.

    `last.ckpt` is written periodically and a `full` checkpoint is 27 GB, so writing one takes
    minutes — during which a wall clock or a `scancel` leaves a **truncated zip** behind. That is
    not an exotic failure; it is the ordinary outcome of stopping a job at the wrong moment, and
    it cost a resume of a run that was 96,000 steps in:

        RuntimeError: PytorchStreamReader failed reading zip archive:
                      failed finding central directory

    So `auto_resume` no longer trusts existence as proof of readability. Validity is checked with
    `zipfile.is_zipfile`, which reads only the end-of-central-directory record — the very thing a
    truncated write destroys — rather than by loading 27 GB to find out. On failure it falls back
    through the numbered checkpoints, newest first, so a torn `last.ckpt` costs at most the steps
    since the previous top-k save instead of the whole run.
    """
    import zipfile

    if not directory.is_dir():
        return None

    candidates = []
    last = directory / "last.ckpt"
    if last.is_file():
        candidates.append(last)
    # `{step}.ckpt` from the monitored callback. Sorted by step, not mtime: a top-k save can
    # rewrite an older file's timestamp when the list shifts.
    numbered = []
    for path in directory.glob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        stem = path.stem.split("-")[-1]
        numbered.append((int(stem) if stem.isdigit() else -1, path))
    candidates.extend(path for _, path in sorted(numbered, reverse=True))

    for path in candidates:
        if zipfile.is_zipfile(path):
            if path.name != "last.ckpt":
                print(
                    f"[p2pa] last.ckpt is unreadable (truncated by an interrupted write); "
                    f"falling back to {path.name}",
                    flush=True,
                )
            return path
        print(f"[p2pa] skipping unreadable checkpoint {path.name}", flush=True)

    if candidates:
        raise SystemExit(
            f"[p2pa] every checkpoint in {directory} is unreadable. Move them aside and start "
            "fresh, or restore from a backup — resuming is not possible."
        )
    return None



def stored_hyperparameters(checkpoint_path) -> dict | None:
    """Read a checkpoint's saved config without materialising its tensors.

    A `full` checkpoint is 27 GB, two thirds of it AdamW moments, and this function wants a small
    dict. Loading the whole thing cost 27 GB of resident memory *per run* — and because Lightning
    then loads it again for the real restore, a paired resume tripped the RSS guard at 66 GB before
    training had done anything. The guard was right; the reason it fired was this.

    Two ways to avoid it, in order of preference:

    * `pickle.Unpickler` over the archive's `data.pkl` with a stub that turns every tensor rebuild
      into `None`. The hyperparameters are plain Python — strings, numbers, lists — so they survive
      intact while not one storage is read. This is exact and costs kilobytes.
    * `torch.load(..., mmap=True)` as a fallback, which leaves storages on disk and pages in only
      what is touched. Still cheap, and it does not depend on the archive's internal layout.
    """
    import pickle
    import zipfile

    class _NoTensors(pickle.Unpickler):
        """Every tensor becomes None; everything else unpickles normally."""

        def find_class(self, module: str, name: str):
            if module.startswith("torch") and (
                "rebuild" in name or name in {"_load_from_bytes", "Tensor", "Parameter"}
            ):
                return lambda *args, **kwargs: None
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError):
                # An unpicklable class in some other part of the checkpoint must not stop us
                # reading the config, which is all this function is after.
                return lambda *args, **kwargs: None

        def persistent_load(self, saved_id):
            return None                       # storage references: never followed

    try:
        with zipfile.ZipFile(checkpoint_path) as archive:
            entry = next(n for n in archive.namelist() if n.endswith("data.pkl"))
            with archive.open(entry) as handle:
                payload = _NoTensors(handle).load()
        if isinstance(payload, dict) and payload.get("hyper_parameters"):
            return payload["hyper_parameters"]
    except Exception:  # noqa: BLE001 - any failure falls through to the slower, safer path
        pass

    import torch

    stored = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    hyperparameters = stored.get("hyper_parameters")
    # Drop the reference before returning: the caller has no use for 27 GB of optimizer state, and
    # holding it until the next collection is exactly what tripped the guard.
    del stored
    return hyperparameters


def check_resume_compatible(cfg: DictConfig, checkpoint_path) -> None:
    """Refuse to resume a checkpoint trained under a different configuration.

    `auto_resume` exists so a job that hits a wall clock is continued rather than restarted, and
    that is the only thing it should do. Without this guard it also silently continues a run whose
    config has since changed. A change that alters a tensor shape fails loudly on its own; every
    other one would resume cleanly and produce a run trained half under one regime and half under
    another, with nothing in the log to say so.

    The remedy is deliberately not "resume anyway": pick a new `exp_name` so both runs survive, or
    delete the directory if the old one is genuinely unwanted.
    """
    previous = stored_hyperparameters(checkpoint_path)
    if not previous:
        return
    # Through `migrate_config`, so a run recorded under `phase.*` compares as `finetune.*`.
    previous = migrate_config(OmegaConf.create(previous))

    changed, keys = [], []
    for key in RESUME_CRITICAL:
        before, after = _lookup(previous, key), _lookup(cfg, key)
        if before is None or before == after:
            continue
        # Raising `max_steps` is the one change here that is a deliberate act rather than an
        # accident: it is how a finished run is trained further, and `resume_main` already
        # advertises it as something that legitimately varies between launches. Allowed, but never
        # silently -- the LR schedule is a cosine to zero at `max_steps` (`configure_optimizers`), so the
        # rate does not continue from where it stopped, it *restarts*: a run that ended at 500k
        # under a 500k schedule sits at exactly 0, and resuming it under 750k puts step 500k at
        # two thirds of the new cosine, i.e. 25% of peak. That is a warm restart, and a reader of
        # the loss curve deserves to know a discontinuity there was intended.
        if key == "trainer.max_steps" and isinstance(before, int) and isinstance(after, int) \
                and after > before:
            print(
                f"[p2pa] extending {cfg.exp_name}: max_steps {before} -> {after}. The cosine "
                f"schedule is rebuilt against the new total, so the learning rate restarts rather "
                f"than continuing from zero. This is a warm restart, not a longer run of the same "
                f"schedule.",
                flush=True,
            )
            continue
        changed.append(f"    {key}: {before!r} -> {after!r}")
        keys.append(key)
    if not changed:
        return

    # The remedy that is almost always wanted, and that the first version of this message left
    # out: put the old values back and the run simply continues. Offering only "rename", "delete"
    # or "disable the check" framed a recoverable situation as a choice between losing the run and
    # losing the guard.
    restore = " ".join(f"{key}={_lookup(previous, key)}" for key in keys)
    raise SystemExit(
        f"[p2pa] refusing to resume {checkpoint_path}\n"
        f"it was trained under a different configuration:\n" + "\n".join(changed) + "\n\n"
        "Resuming would give one run two regimes and no record of the join. Either:\n"
        f"    continue it as it was   {restore}\n"
        f"    or start a new run      exp_name={cfg.exp_name}_v2\n"
        f"    or discard the old      rm -rf {checkpoint_path.parent.parent}\n"
        "    or, if the change really is cosmetic, trainer.auto_resume=false"
    )


def train(cfg: DictConfig) -> None:
    """Everything a launch does, once its config is composed and validated.

    Separated from the Hydra entrypoint so that `resume_main` can drive the same code from a
    `resolved_config.yaml` instead of from config groups and command-line overrides.
    """
    from .callbacks import (
        DryRunReport,
        EMACallback,
        LossSpikeGuard,
        ResourceGuard,
        SampleCallback,
        UnfreezeBackbone,
    )
    from .dataset import P2PADataModule
    from .model import P2PAModule

    pl.seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    run_dir = resolve_path(str(cfg.runs_dir)) / str(cfg.exp_name)
    tb_dir = resolve_path(str(cfg.tensorboard_dir)) / str(cfg.exp_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")

    module = P2PAModule(cfg)
    banner(cfg, module)
    datamodule = P2PADataModule(cfg, data_root())

    callbacks = [
        UnfreezeBackbone(module.freeze_steps),
        # Two ModelCheckpoints on purpose. A monitored callback with save_last=True only writes
        # last.ckpt when a top-k save also fires, so a validation plateau freezes `last` and the
        # run cannot resume from where it actually is.
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{step}",
            monitor="val/loss",
            mode="min",
            save_top_k=int(cfg.scratch.save_top_k if module.scratch else cfg.finetune.save_top_k),
            save_last=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="last",
            monitor=None,
            save_top_k=1,
            every_n_train_steps=int(cfg.trainer.val_check_interval),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        ResourceGuard(
            float(cfg.trainer.rss_limit_gb), int(cfg.trainer.log_every_n_steps)
        ),
        LossSpikeGuard(
            float(cfg.trainer.loss_spike_factor), int(cfg.trainer.loss_spike_warmup)
        ),
        SampleCallback(cfg, run_dir),
    ]
    if module.scratch:
        # Before the listening exports, so they are rendered from the averaged weights too.
        callbacks.insert(-1, EMACallback(decay=float(cfg.scratch.ema_decay)))

    dry_run = int(cfg.trainer.dry_run_steps)
    if dry_run:
        # Nothing that writes to disk: a rehearsal must not leave a `last.ckpt` a later real run
        # would silently resume from at step 20.
        callbacks = [c for c in callbacks if not isinstance(c, (ModelCheckpoint, SampleCallback))]
        callbacks.append(DryRunReport(dry_run))

    trainer = pl.Trainer(
        max_steps=int(cfg.trainer.max_steps),
        max_epochs=-1,
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=str(cfg.trainer.precision),
        gradient_clip_val=float(cfg.trainer.grad_clip),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        val_check_interval=int(cfg.trainer.val_check_interval),
        # Steps, not epochs: validation and checkpointing run on a step clock.
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.trainer.limit_val_batches,
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        callbacks=callbacks,
        logger=TensorBoardLogger(str(tb_dir), name=""),
        num_sanity_val_steps=0 if dry_run else 2,
        enable_checkpointing=not dry_run,
    )

    resume = None
    if bool(cfg.trainer.auto_resume) and not dry_run:
        resume = newest_valid_checkpoint(run_dir / "checkpoints")
    if resume:
        check_resume_compatible(cfg, resume)
        print(f"[p2pa] resuming {resume}", flush=True)
        resume = str(resume)
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume)


@hydra.main(version_base=None, config_path=str(config_dir()), config_name="config")
def main(cfg: DictConfig) -> None:
    validate_config(cfg)
    train(cfg)


def resume_main() -> None:
    """`p2pa-resume <run_dir> [key=value ...]` — continue a run from the config it recorded.

    Every launch writes its fully composed config to `runs/<exp_name>/resolved_config.yaml`, and
    `check_resume_compatible` refuses to continue a checkpoint under a different one. Between them
    that means the overrides a run was started with are load-bearing forever — and they used to
    live only in whichever shell script happened to launch it, so an ad-hoc cell was resumable only
    by remembering what was typed. Reading the run's own record removes that: the run carries its
    configuration, and the launcher is free to be a convenience again.

    Trailing `key=value` overrides are applied on top, for the things that legitimately change
    between launches — `trainer.devices`, `trainer.max_steps`, `data.num_workers`.
    """
    import argparse

    parser = argparse.ArgumentParser(description="continue a run from its resolved_config.yaml")
    parser.add_argument("run_dir", help="runs/<exp_name>, or the path to a resolved_config.yaml")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. trainer.devices=[0]")
    args = parser.parse_args()

    path = pathlib.Path(args.run_dir)
    if path.is_dir():
        path = path / "resolved_config.yaml"
    if not path.is_file():
        raise SystemExit(
            f"no resolved_config.yaml at {path}. Every run writes one when it starts, so a run "
            "directory without one has never trained a step — launch it with `p2pa-train` instead."
        )

    # Through `migrate_config`, as every reader of a recorded run does: a run
    # started before a config key existed recorded a config without it, and the point of reading
    # the record is that an *old* run stays launchable.
    cfg = migrate_config(OmegaConf.load(path))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    validate_config(cfg)
    print(f"[p2pa] resuming {cfg.exp_name} from {path}", flush=True)
    train(cfg)


if __name__ == "__main__":
    main()
