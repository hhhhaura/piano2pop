"""The dataset and batching over whole-track ACE-Step latents.

Length is decided *before* an item is fetched, by `LengthBatchSampler`, so `__getitem__` takes a
`(row_index, target_seconds)` tuple rather than a bare index. Everything is sliced out of one
cached whole-track latent; there is no per-window cache to rebuild when anything moves.

Two frame rates live here and must not be confused. The latent runs at `ace.sample_rate /
ace.hop_length` (exactly 25 Hz) and the roll at `roll.oversample` times that (100 Hz), because a
40 ms latent frame cannot place a note onset. A window of `frames` latent frames therefore always
pairs with `frames * oversample` roll frames, and the encoder's strided stem brings the two back
into correspondence.

**Every batch is exactly one length, and no item is ever padded.** That is not tidiness: ACE-Step's
DiT discards the attention mask it is handed (`AceStepDiTModel.forward` assigns `attention_mask =
None` before building any mask), so a padded frame is a frame the model attends to as if it were
real. Two things enforce it — the sampler draws one length per batch, and `_load_splits` refuses
any track shorter than the longest window the run will ask for.

The other thing this file is careful about is host memory. The run that took this box into swap did
it here, not on the GPU: see `_candidates`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

from .augment import segment_augment_notes
from .config import latent_fps, latent_frames
from .instrument_groups import normalize_programs
from .paths import beat_path, latent_path, manifest_path, midi_path, variant_midis
from .paths import notes_dir as notes_dir_of
from .pianoroll import cached_notes, roll_from_midi, usable_notes
from .stages import read_jsonl


def curriculum_ceiling(cfg: DictConfig, step: int) -> float:
    """The longest window reachable at `step`, in seconds.

    The single source of truth for the ramp. `LengthBatchSampler` uses it to draw training lengths
    and `SampleCallback` uses it to decide how long a listening export should be — two callers who
    would otherwise each carry their own copy of `min + ramp * (step // every)`, which is exactly
    the kind of duplication that drifts silently when the schedule changes.
    """
    ramp = float(cfg.length.ramp_seconds)
    every = int(cfg.length.ramp_every_steps)
    if ramp <= 0 or every < 1:
        return float(cfg.length.max_seconds)
    hold = int(cfg.length.hold_steps)
    if int(step) < hold:
        return float(cfg.length.min_seconds)
    # `hold` steps at the floor, then the first increment lands *at* `hold` — so "20 s for 100,000
    # steps, then +5 s every 10,000" means exactly that: step 99,999 is 20 s and step 100,000 is
    # 25 s. Counting the first post-hold interval as another stage at the floor would hold for
    # 110,000 steps while claiming 100,000.
    stage = (1 + (int(step) - hold) // every) if hold else (int(step) // every)
    return float(min(float(cfg.length.min_seconds) + ramp * stage, float(cfg.length.max_seconds)))


def deterministic_offset(sample_id: str, span: int) -> int:
    """A fixed window start for a given item, so a validation curve compares like with like."""
    if span <= 0:
        return 0
    digest = hashlib.sha256(sample_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (span + 1)


def worker_init(worker_id: int) -> None:
    """One BLAS thread per worker.

    Torch defaults every process to a thread pool the width of the machine. With four workers on a
    many-core host that is four full pools competing for the same cores, each with its own arena —
    a large, invisible memory cost on top of the obvious one, and the fastest way to make a
    dataloader slower than the model it feeds.
    """
    torch.set_num_threads(1)


class WindowDataset(Dataset):
    """One row per track. `rows` must already be filtered to `status == "ok"` and split."""

    def __init__(self, rows: list[dict], cfg: DictConfig, *, augment: bool = False):
        self.cfg = cfg
        self.rows = list(rows)
        self.latent_fps = latent_fps(cfg)
        self.oversample = int(cfg.roll.oversample)
        self.training_augmentation = bool(augment)
        self.programs = normalize_programs([int(program) for program in cfg.prep.programs])
        self.pitch_low = int(cfg.roll.pitch_low)
        self.pitch_high = int(cfg.roll.pitch_high)
        self.min_notes = int(cfg.data.min_notes)

        augment_cfg = cfg.augment
        self.segment_bars = int(cfg.window.segment_bars)
        self.segment_augment_enabled = augment and bool(cfg.window.segment_augment)
        self.baseline_probability = float(augment_cfg.baseline_probability)
        self.picogen_probability = float(augment_cfg.picogen_probability)
        self.style_weights = {
            "normal": float(augment_cfg.normal),
            "chord_thin": float(augment_cfg.chord_thin),
            "octave_move": float(augment_cfg.octave_move),
            "octave_shift": float(augment_cfg.octave_shift),
        }
        self.chord_tone_rate = tuple(float(v) for v in augment_cfg.chord_tone_rate)
        self.tension_rate = tuple(float(v) for v in augment_cfg.tension_rate)
        self.octave_move_rate = tuple(float(v) for v in augment_cfg.octave_move_rate)
        self.octave_shift_choices = tuple(int(v) for v in augment_cfg.octave_shift_choices)
        maximum = int(cfg.window.segment_max_simultaneous_notes)
        self.segment_max_simultaneous_notes = maximum if maximum > 0 else None

        self.picogen_name = str(cfg.data.source.picogen_midi)

    def __len__(self) -> int:
        return len(self.rows)

    # --- conditioning ------------------------------------------------------

    def _candidates(self, row: dict) -> dict[str, list]:
        """The conditioning MIDIs this item can draw from, parsed.

        Deliberately *not* memoised in the worker. `cached_notes` already serves these from an
        on-disk npz in a few milliseconds, against a GPU that consumes far fewer items per second —
        so even one worker outruns the model without a memo. An in-process dict keyed by track id,
        by contrast, could not evict: the sampler shuffles each row's repeats across the whole
        pass, so a worker touches most of the 4,157 songs and ends up holding the parsed corpus.
        At ~2.8 MB a track that is 11.4 GB per worker, and it is what pushed this box into swap.

        The variants are read only by `segment_augment_notes`. With segment augmentation off the
        baseline is the only entry `_conditioning_notes` looks at, so parsing the rest of the pool
        (`source.excluded_variant_programs` decides how many that is) costs several times the time
        and memory for nothing.
        """
        notes = str(notes_dir_of(self.cfg))
        candidates = {"baseline": cached_notes(midi_path(row, self.cfg), notes)}
        if self.segment_augment_enabled:
            for path in variant_midis(row, self.cfg):
                name = "picogen" if path.name == self.picogen_name else path.stem
                candidates[name] = cached_notes(path, notes)
        return candidates

    def _downbeats(self, row: dict) -> list[float]:
        path = beat_path(row, self.cfg)
        downbeats: list[float] = []
        if path.is_file():
            try:
                downbeats = [float(v) for v in json.loads(path.read_text()).get("downbeats", [])]
            except (json.JSONDecodeError, OSError):
                downbeats = []
        return downbeats

    def _conditioning_notes(self, row: dict) -> list:
        candidates = self._candidates(row)
        if not self.segment_augment_enabled:
            return candidates["baseline"]
        # Seeded from torch's RNG, not entropy: Lightning reseeds per worker per epoch from
        # cfg.seed, so the draw varies across passes while the run stays reproducible from its seed.
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        return segment_augment_notes(
            candidates,
            self._downbeats(row),
            float(row["duration"]),
            bars_per_segment=self.segment_bars,
            baseline_name="baseline",
            picogen_name="picogen",
            baseline_probability=self.baseline_probability,
            picogen_probability=self.picogen_probability,
            min_notes=int(self.cfg.data.min_notes),
            pitch_low=self.pitch_low,
            pitch_high=self.pitch_high,
            programs=self.programs,
            style_weights=self.style_weights,
            chord_tone_rate=self.chord_tone_rate,
            tension_rate=self.tension_rate,
            octave_move_rate=self.octave_move_rate,
            octave_shift_choices=self.octave_shift_choices,
            max_simultaneous_notes=self.segment_max_simultaneous_notes,
            rng=np.random.default_rng(seed),
        )

    # --- items -------------------------------------------------------------

    def _offset(self, row: dict, span: int, attempt: int) -> int:
        if span <= 0:
            return 0
        if self.training_augmentation:
            return int(torch.randint(0, span + 1, (1,)).item())
        # Validation stays deterministic across passes, so successive attempts walk a fixed
        # sequence keyed on the item rather than drawing fresh randomness.
        return deterministic_offset(f"{row['sample_id']}:{attempt}", span)

    def _onset_times(self, notes: list) -> np.ndarray:
        """Sorted onset times of the notes that survive the program and pitch filters.

        Built once per item so `_draw_window` can score a candidate offset with two binary searches
        instead of cropping and re-filtering the whole track. Onsets rather than overlaps: a single
        sustained chord should not make a window look busy.
        """
        usable = usable_notes(notes, self.pitch_low, self.pitch_high, self.programs)
        return np.sort(
            np.fromiter((note.start for note in usable), dtype=np.float64, count=len(usable))
        )

    def _draw_window(self, row: dict, notes: list, frames: int, span: int) -> int:
        """A start frame whose window actually contains something to condition on.

        Roughly 3% of 10 s draws and 0.1% of 30 s ones land on a silent intro or outro, and an
        empty roll paired with real audio teaches the model to invent rather than to arrange. The
        window is *moved* rather than the item dropped: `__getitem__` is called for a slot the
        sampler has already committed to, so returning nothing would make batch size vary and,
        when a whole batch fails, produce an empty batch.

        After `attempts` tries the densest one seen is used regardless — a track that is genuinely
        this sparse everywhere has no better window to offer.
        """
        attempts = int(self.cfg.window.draw_attempts)
        if attempts < 1 or span <= 0:
            return self._offset(row, span, 0)

        onsets = self._onset_times(notes)
        length = frames / self.latent_fps
        best_start, best_count = 0, -1
        for attempt in range(attempts):
            start = self._offset(row, span, attempt)
            seconds = start / self.latent_fps
            lo, hi = np.searchsorted(onsets, (seconds, seconds + length))
            count = int(hi - lo)
            if count >= self.min_notes:
                return start
            if count > best_count:
                best_start, best_count = start, count
        return best_start

    def __getitem__(self, item: tuple[int, float]) -> dict:
        row_index, target_seconds = item
        row = self.rows[row_index]
        # mmap, and only the window is ever made resident. A whole-track latent is ~1.4 MB, so a
        # worker that materialised every track it touched would hold several gigabytes.
        latent = np.load(latent_path(row, self.cfg), mmap_mode="r")
        total_frames = latent.shape[-1]
        frames = latent_frames(self.cfg, float(target_seconds))
        if frames > total_frames:
            raise RuntimeError(
                f"{row['sample_id']} is {total_frames} latent frames but the sampler asked for "
                f"{frames}. `_load_splits` is meant to have dropped it — a padded item would be "
                "attended to as if it were real, because the DiT ignores the padding mask."
            )
        span = max(0, total_frames - frames)

        # The conditioning notes are drawn once for the whole track, then the window is placed
        # against them — the source and style draws are per bar, not per window, so re-running them
        # for every attempted offset would cost milliseconds a try and change nothing.
        notes = self._conditioning_notes(row)
        start_frame = self._draw_window(row, notes, frames, span)
        start_seconds = start_frame / self.latent_fps
        seconds = frames / self.latent_fps

        # `[64, T]` on disk, `[T, 64]` for the model: ACE-Step is time-major throughout.
        window = np.ascontiguousarray(latent[:, start_frame : start_frame + frames].T)
        roll = roll_from_midi(
            midi_path(row, self.cfg),
            self.cfg,
            frames * self.oversample,
            start_seconds,
            notes=notes,
        )
        return {
            "sample_id": row["sample_id"],
            "latent": torch.from_numpy(window.astype(np.float32)),
            "roll": torch.from_numpy(roll.astype(np.float32)),
            "seconds": float(seconds),
            "start_seconds": float(start_seconds),
        }

    def sample_rows(self, count: int, explicit_ids: list[str] | None = None) -> list[dict]:
        """A fixed, deterministic subset, so listening exports compare by ear across checkpoints."""
        by_id = {row["sample_id"]: row for row in self.rows}
        if explicit_ids:
            missing = [name for name in explicit_ids if name not in by_id]
            if missing:
                raise ValueError(f"Requested sample ids absent from this split: {missing}")
            return [by_id[name] for name in explicit_ids]
        return [by_id[name] for name in sorted(by_id)[:count]]


class LengthBatchSampler(Sampler):
    """Decides every item's length before `WindowDataset.__getitem__` sees it.

    One length per batch, drawn uniformly between `min_seconds` and the current curriculum ceiling
    (and fixed at `max_seconds` for validation). Per batch rather than per item, so a batch never
    contains two lengths and therefore never contains padding — the constraint the DiT's discarded
    attention mask imposes. Length diversity still comes from the draw varying across batches.

    `repeats` puts each row into a pass that many times, each occurrence with its own independent
    length, offset and augmentation draw: ten repeats is ten different segments of the same song,
    not ten copies of one item.
    """

    def __init__(
        self,
        num_rows: int,
        *,
        min_seconds: float,
        max_seconds: float,
        fps: float,
        patch: int,
        batch_size: int,
        seed: int = 0,
        repeats: int = 1,
        fixed: bool = False,
        get_step=None,
        ramp_seconds: float = 0.0,
        ramp_every_steps: int = 0,
        hold_steps: int = 0,
    ):
        self.num_rows = num_rows
        self.min_seconds = float(min_seconds)
        self.max_seconds = float(max_seconds)
        self.fps = float(fps)
        self.patch = max(1, int(patch))
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.repeats = max(1, int(repeats))
        self.fixed = bool(fixed)
        # `get_step` reads the trainer's global step. The batch sampler is consumed in the main
        # process, so it can see it directly — no dataloader rebuild, and no epoch bookkeeping
        # that would break the moment `max_epochs=-1` made epochs meaningless here.
        self.get_step = get_step
        self.ramp_seconds = float(ramp_seconds)
        self.ramp_every_steps = int(ramp_every_steps)
        self.hold_steps = int(hold_steps)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def ceiling(self) -> float:
        """The longest window currently reachable.

        This holds at `min_seconds` for `hold_steps`, rises by `ramp_seconds` every
        `ramp_every_steps` optimizer steps, and then clamps at `max_seconds`. The floor does not move: the range *widens*, so a
        model that has learned short windows keeps being asked for them rather than having them
        taken away.

        Prefetching means the sampler runs a few batches ahead of the trainer, so a stage boundary
        is crossed a fraction of a second early. At a 10,000-step granularity that is noise.
        """
        if self.ramp_seconds <= 0 or self.ramp_every_steps < 1 or self.get_step is None:
            return self.max_seconds
        # Same arithmetic as `curriculum_ceiling`, expressed against this sampler's own fields
        # because it may be constructed standalone (tests do) without a full config to hand. The
        # two are pinned equal by `test_the_sampler_and_the_listening_export_agree`.
        step = int(self.get_step())
        if step < self.hold_steps:
            return self.min_seconds
        stage = (
            1 + (step - self.hold_steps) // self.ramp_every_steps
            if self.hold_steps
            else step // self.ramp_every_steps
        )
        return float(min(self.min_seconds + self.ramp_seconds * stage, self.max_seconds))

    def _draw(self, rng: np.random.Generator) -> float:
        ceiling = self.ceiling()
        if self.fixed or ceiling <= self.min_seconds:
            frames = int(round(ceiling * self.fps))
            frames -= frames % self.patch
            return max(self.patch, frames) / self.fps
        # Quantised to a whole number of DiT patches here rather than in the dataset, so every
        # item in the batch resolves to the identical frame count by construction.
        seconds = float(rng.uniform(self.min_seconds, ceiling))
        frames = int(round(seconds * self.fps))
        frames -= frames % self.patch
        return max(self.patch, frames) / self.fps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.repeat(np.arange(self.num_rows), self.repeats)
        rng.shuffle(order)
        order = order.tolist()
        for start in range(0, len(order), self.batch_size):
            target = self._draw(rng)
            yield [(index, target) for index in order[start : start + self.batch_size]]

    def __len__(self) -> int:
        return math.ceil(self.num_rows * self.repeats / self.batch_size)


def collate(batch: list[dict]) -> dict:
    """Stack. Every item is the same length by construction, and this asserts it."""
    frames = {item["latent"].shape[0] for item in batch}
    if len(frames) != 1:
        raise RuntimeError(
            f"a batch carried {sorted(frames)} latent frames. Batches must be homogeneous: the "
            "DiT discards the padding mask, so a short item's tail would be attended to as real."
        )
    length = frames.pop()
    oversample = batch[0]["roll"].shape[-1] // length
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "latent": torch.stack([item["latent"] for item in batch]),
        "roll": torch.stack([item["roll"] for item in batch]),
        # All ones, and kept anyway: the loss and the roll encoder both take a mask, and a tensor
        # that is always true is cheaper to carry than two code paths.
        "mask": torch.ones(len(batch), length, dtype=torch.bool),
        "seconds": torch.tensor([item["seconds"] for item in batch], dtype=torch.float32),
        "start_seconds": torch.tensor(
            [item["start_seconds"] for item in batch], dtype=torch.float32
        ),
        "oversample": oversample,
    }


class P2PADataModule(pl.LightningDataModule):
    """Splits, samplers and dataloaders."""

    def __init__(self, cfg: DictConfig, root: Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_ds: WindowDataset | None = None
        self.val_ds: WindowDataset | None = None

    def _load_splits(self) -> None:
        rows = [row for row in read_jsonl(manifest_path(self.cfg)) if row.get("status") == "ok"]
        if not rows:
            raise RuntimeError("No 'ok' rows in the manifest; run p2pa-prep first")

        # Nothing shorter than the longest window this run will ever draw. The corpus is full pop
        # songs, so this drops a handful of outliers — and it is what makes "no padding, ever" a
        # property of the data rather than a hope.
        floor = float(self.cfg.length.max_seconds)
        long_enough = [row for row in rows if float(row["duration"]) >= floor]
        dropped = len(rows) - len(long_enough)

        by_split: dict[str, list[dict]] = {}
        for row in long_enough:
            by_split.setdefault(row["split"], []).append(row)
        for name in ("train", "validation"):
            if not by_split.get(name):
                raise RuntimeError(f"The '{name}' split is empty out of {len(long_enough)} rows")
        print(
            f"[data] train={len(by_split['train'])} validation={len(by_split['validation'])} "
            f"tracks ({len(rows)} 'ok', {dropped} shorter than {floor:g}s dropped)",
            flush=True,
        )
        self.train_ds = WindowDataset(by_split["train"], self.cfg, augment=True)
        self.val_ds = WindowDataset(by_split["validation"], self.cfg)

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is None:
            self._load_splits()

    def _sampler(self, dataset, *, validation: bool, seed: int, repeats: int) -> LengthBatchSampler:
        length = self.cfg.length
        return LengthBatchSampler(
            len(dataset),
            # Validation is always at `max_seconds`: a metric that measures a different length
            # every time it is computed is not a metric.
            min_seconds=float(length.min_seconds),
            max_seconds=float(length.max_seconds),
            fps=latent_fps(self.cfg),
            patch=int(self.cfg.ace.patch_size),
            batch_size=int(self.cfg.data.batch_size),
            seed=seed,
            repeats=repeats,
            fixed=validation,
            get_step=None if validation else (
                lambda: int(self.trainer.global_step) if self.trainer else 0
            ),
            ramp_seconds=0.0 if validation else float(length.ramp_seconds),
            ramp_every_steps=0 if validation else int(length.ramp_every_steps),
            hold_steps=0 if validation else int(length.hold_steps),
        )

    def _dataloader(self, dataset, sampler) -> DataLoader:
        workers = int(self.cfg.data.num_workers)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=workers,
            worker_init_fn=worker_init,
            # Bounded resident items: workers x prefetch x batch. Both defaults are deliberately
            # small — see `configs/data/p2pdata.yaml`.
            prefetch_factor=int(self.cfg.data.prefetch_factor) if workers else None,
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.persistent_workers) and workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        self.setup()
        return self._dataloader(
            self.train_ds,
            self._sampler(
                self.train_ds,
                validation=False,
                seed=int(self.cfg.seed),
                repeats=int(self.cfg.window.segments_per_song),
            ),
        )

    def val_dataloader(self) -> DataLoader:
        self.setup()
        # Always one repeat: validation never augments and its offsets are deterministic, so a
        # second pass would render an identical item.
        return self._dataloader(
            self.val_ds,
            self._sampler(self.val_ds, validation=True, seed=int(self.cfg.seed) + 1, repeats=1),
        )
