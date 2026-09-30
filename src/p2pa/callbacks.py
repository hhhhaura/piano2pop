"""Callbacks: the freeze window, the resource guards, an EMA, and periodic listening exports.

Three of these exist because of specific failures. `UnfreezeBackbone` implements the freeze
window that replaces `p2p-stable`'s two-run phase handoff. `ResourceGuard` watches host memory
across the whole process tree, which is what actually fell over. `LossSpikeGuard` names the input
behind a spike instead of leaving it as a jump in the curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .audio import decode_window, encode_mp3, render_midi, require_executable, save_mp3
from .config import latent_fps, latent_frames
from .paths import audio_path, midi_path, notes_dir, soundfont_path
from .pianoroll import crop_notes, write_midi


class UnfreezeBackbone(pl.Callback):
    """Hold the backbone still for the first `finetune.freeze_steps`, then let it move.

    Only `requires_grad` is flipped here. The learning rate is already zero for that parameter
    group until the same step (`configure_optimizers`), so the two agree and a resume anywhere in
    the run lands in the right state without either having to record anything: the decision is a
    pure function of `global_step`.

    Why a freeze window at all, in a project that has no phase handoff: the roll encoder's output
    projection starts at exactly zero, so its condition starts out saying nothing. Letting 2.39B
    pretrained parameters chase a meaningless condition for the first few thousand steps is how a
    finetune forgets what it knew before it learns anything new.
    """

    def __init__(self, freeze_steps: int) -> None:
        self.freeze_steps = int(freeze_steps)

    def _sync(self, trainer, module) -> None:
        should_freeze = int(trainer.global_step) < self.freeze_steps
        if should_freeze == module.backbone_frozen:
            return
        module.apply_freeze(should_freeze)
        module.train(module.training)
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(
            f"[p2pa] step {trainer.global_step}: backbone "
            f"{'frozen' if should_freeze else 'unfrozen'}, {trainable / 1e6:.1f}M trainable",
            flush=True,
        )

    def on_train_start(self, trainer, module) -> None:
        self._sync(trainer, module)

    def on_train_batch_start(self, trainer, module, batch, batch_idx) -> None:
        self._sync(trainer, module)


class ResourceGuard(pl.Callback):
    """Log peak host and device memory, and stop before the OOM killer does.

    Resident set is summed over the process tree, not read from the parent. The run that took this
    box into swap did it in the dataloader workers, whose memory does not appear in the trainer's
    own RSS at all — measuring only the parent would have shown nothing wrong right up to the point
    the machine stopped responding.
    """

    def __init__(self, limit_gb: float, every_n_steps: int = 20) -> None:
        self.limit_gb = float(limit_gb)
        self.every = max(1, int(every_n_steps))
        self.peak_rss = 0.0

    def _tree_memory_gb(self) -> float:
        """Physical memory actually held by this run, summed over its process tree.

        **PSS, not RSS.** Dataloader workers are *forked*, so they share the parent's pages
        copy-on-write — and RSS counts a shared page in full in every process that maps it. With a
        parent holding a 27 GB checkpoint and four workers forked from it, summing RSS reports
        135 GB for 27 GB of real memory. Measured here at 5.0x on a five-process tree.

        That is not a small inaccuracy in a guard whose entire job is to decide when memory is a
        problem: it stopped two legitimate `full` resumes, first at "66 GB" and then at "118 GB",
        neither of which was real. PSS divides each shared page by the number of processes mapping
        it, so a forked tree sums to what is genuinely resident.

        `memory_full_info()` reads `/proc/pid/smaps`, which is slower than `memory_info()` — hence
        every `log_every_n_steps` rather than every step. It is Linux-only; elsewhere, and for any
        process that disappears mid-walk, this falls back to RSS and over-reports rather than
        under-reports, which is the safe direction for a guard.
        """
        import psutil

        process = psutil.Process()
        total = 0
        for target in [process, *process.children(recursive=True)]:
            try:
                total += target.memory_full_info().pss
            except (psutil.Error, AttributeError):
                try:
                    total += target.memory_info().rss
                except psutil.Error:
                    continue
        return total / 1024**3

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if int(trainer.global_step) % self.every:
            return
        used = self._tree_memory_gb()
        self.peak_rss = max(self.peak_rss, used)
        module.log("sys/host_gb", used, prog_bar=False)
        if torch.cuda.is_available():
            module.log("sys/vram_gb", torch.cuda.max_memory_allocated() / 1024**3)
        if self.limit_gb > 0 and used > self.limit_gb:
            raise MemoryError(
                f"host memory {used:.1f} GB (PSS across this process tree) exceeds "
                f"trainer.rss_limit_gb={self.limit_gb:g}. Lower data.num_workers or "
                "data.prefetch_factor rather than raising the limit — the limit is what stops "
                "this run taking the host down with it."
            )


class LossSpikeGuard(pl.Callback):
    """Name the input behind a loss spike, and drop the step.

    `p2p-stable` lost a run to a single corrupted MIDI whose window produced a loss orders of
    magnitude above the rest, and the only evidence was a vertical line on a TensorBoard curve.
    `prep.reject_outliers` is meant to catch such a track before training ever sees it; this is the
    second line, and the one that says *which* track when it does not.

    The running reference is a median over a fixed-size window, so it cannot be dragged upward by
    the very spikes it is meant to detect.
    """

    def __init__(self, factor: float, warmup: int, window: int = 200) -> None:
        self.factor = float(factor)
        self.warmup = int(warmup)
        self.window = int(window)
        self.history: list[float] = []
        self.skipped = 0
        self._skip_current = False
        self._skip_accumulation = False
        self._batch_info = ""

    def on_train_batch_start(self, trainer, module, batch, batch_idx) -> None:
        self._skip_current = False
        self._batch_info = (
            f"sample_ids={list(batch['sample_id'])} "
            f"seconds={float(batch['seconds'][0]):.1f} "
            f"start={[round(float(value), 1) for value in batch['start_seconds']]}"
        )

    def on_before_backward(self, trainer, module, loss) -> None:
        value = float(loss.detach())
        if self.factor <= 0 or len(self.history) < self.warmup:
            self.history.append(value)
            self.history = self.history[-self.window:]
            return
        reference = float(torch.tensor(self.history).median())
        if reference > 0 and value > self.factor * reference:
            self._skip_current = True
            self._skip_accumulation = True
            self.skipped += 1
            print(
                f"[p2pa] step {trainer.global_step}: loss {value:.4f} is "
                f"{value / reference:.1f}x the running median {reference:.4f}; "
                f"discarding its optimizer update. {self._batch_info}",
                flush=True,
            )
            module.log("train/spikes", float(self.skipped))
            return
        # A spike cannot drag its own reference upward: only accepted batches enter the window.
        self.history.append(value)
        self.history = self.history[-self.window:]

    def on_before_optimizer_step(self, trainer, module, optimizer) -> None:
        if self._skip_accumulation:
            # AdamW applies weight decay only to parameters with a gradient. Clearing to None
            # therefore makes this a true no-update step rather than a zero-gradient decay step.
            optimizer.zero_grad(set_to_none=True)
            self._skip_accumulation = False

    def state_dict(self) -> dict:
        return {"history": self.history, "skipped": self.skipped}

    def load_state_dict(self, state_dict: dict) -> None:
        self.history = [float(value) for value in state_dict.get("history", [])][-self.window:]
        self.skipped = int(state_dict.get("skipped", 0))


class EMACallback(pl.Callback):
    """An exponential moving average of the trainable weights, for `backbone=scratch`.

    Validation runs under the averaged weights, so top-k selection and the listening exports see
    them; `model.apply_ema` copies them in when a checkpoint is loaded for sampling. The finetune
    does not keep one: a shadow of 2.39B weights is another 9.6 GB.
    """

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

    def _seed(self, module) -> None:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and name not in self.shadow:
                self.shadow[name] = parameter.detach().clone().float()

    def on_train_start(self, trainer, module) -> None:
        # A resumed run has already restored the shadow from the checkpoint, where Lightning maps
        # every tensor onto the CPU — so it is moved onto the parameters' device here, because the
        # update below is in place and would otherwise add a cuda tensor to a cpu one.
        restored, self.shadow = self.shadow, {}
        own = dict(module.named_parameters())
        for name, value in restored.items():
            if name in own:
                self.shadow[name] = value.to(device=own[name].device, dtype=torch.float32)
        self._seed(module)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        self._seed(module)
        # Warm up the average over the first steps rather than starting at the init weights, which
        # would otherwise dominate the shadow for thousands of steps.
        decay = min(self.decay, (1 + trainer.global_step) / (10 + trainer.global_step))
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(parameter.detach().float(), alpha=1 - decay)

    @torch.no_grad()
    def _swap_in(self, module) -> None:
        self.backup = {}
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name].to(parameter.dtype))

    @torch.no_grad()
    def _swap_out(self, module) -> None:
        for name, parameter in module.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}

    def on_validation_start(self, trainer, module) -> None:
        if self.shadow and not trainer.sanity_checking:
            self._swap_in(module)

    def on_validation_end(self, trainer, module) -> None:
        if self.backup:
            self._swap_out(module)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.shadow = {k: v.float() for k, v in state_dict.get("shadow", {}).items()}


class DryRunReport(pl.Callback):
    """Measure a handful of steps and print what a real run would cost, then stop.

    Cheap insurance against the two ways a 250k-step launch wastes a day: it does not fit, or it
    fits and is four times slower than assumed. Both are visible in twenty steps.
    """

    def __init__(self, steps: int) -> None:
        self.steps = int(steps)
        self.started = 0.0

    def on_train_start(self, trainer, module) -> None:
        import time

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.started = time.time()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        import time

        step = int(trainer.global_step)
        if step < self.steps:
            return
        elapsed = time.time() - self.started
        vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        rss = ResourceGuard(0.0)._tree_memory_gb()
        per_step = elapsed / max(1, step)
        total = int(trainer.max_steps) * per_step
        print(
            "\n[p2pa] dry run\n"
            f"       {step} steps in {elapsed:.1f}s  ->  {per_step * 1000:.0f} ms/step\n"
            f"       peak VRAM {vram:.1f} GB, peak host RSS {rss:.1f} GB "
            f"(limit {float(module.cfg.trainer.rss_limit_gb):g} GB)\n"
            f"       batch {batch['latent'].shape[0]} x {batch['latent'].shape[1]} frames "
            f"({float(batch['seconds'][0]):.0f}s)\n"
            f"       {int(trainer.max_steps)} steps would take {total / 3600:.1f} h",
            flush=True,
        )
        trainer.should_stop = True


class SampleCallback(pl.Callback):
    """Write fixed corpus references plus real-pianist Kong listening samples.

    Four tracks per song, because three of them are the only way to read the fourth. `codec.mp3` is
    the target through the VAE and back, which is the ceiling on anything the model can produce;
    `piano.mp3` is what the model was actually told; `target.mp3` is what it was asked for.

    The Kong subset is deliberately listening-only: it has no target instrumental and never enters
    validation loss or checkpoint selection. Every export failure is swallowed. A missing cover,
    fluidsynth, or bad ffmpeg call must not end a 250k-step run over a listening artifact.
    """

    def __init__(self, cfg: DictConfig, run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.latent_fps = latent_fps(cfg)
        self._last = -1

    def _should_run(self, trainer) -> bool:
        if trainer.sanity_checking or trainer.global_rank != 0:
            return False
        every = max(1, int(self.cfg.trainer.sample_every_n_steps))
        bucket = int(trainer.global_step) // every
        if bucket == 0 or bucket == self._last:
            return False
        self._last = bucket
        return True

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, module) -> None:
        if not self._should_run(trainer):
            return
        dataset = getattr(trainer.datamodule, "val_ds", None)
        if dataset is None:
            return

        # The length the model has actually been trained at, not the length the run will finish
        # at. Under `length=curriculum` those differ for the first 100,000 steps, and generating a
        # 30 s export from a model that has only ever seen 10 s windows measures extrapolation —
        # it sounds broken for a reason that has nothing to do with whether training is working.
        #
        # Deliberately *not* the same choice as `val/loss`, which stays pinned at `max_seconds` in
        # every mode. That number has to mean one thing across all 54 cells to be comparable; a
        # listening export only has to be worth listening to.
        from .dataset import curriculum_ceiling

        seconds = curriculum_ceiling(self.cfg, int(trainer.global_step))
        from .covers import tempo_balanced_corpus_rows

        count = int(self.cfg.trainer.sample_items)
        explicit_ids = list(self.cfg.trainer.sample_ids)
        if explicit_ids:
            selected = [
                {"row": row, "tempo_class": "selected", "bpm": None}
                for row in dataset.sample_rows(count, explicit_ids)
            ]
        else:
            selected = tempo_balanced_corpus_rows(self.cfg, dataset.rows, count)
        out_root = self.run_dir / "val_samples" / f"step_{int(trainer.global_step):07d}"
        out_root.mkdir(parents=True, exist_ok=True)

        index = []
        was_training = module.training
        try:
            for position, selection in enumerate(selected):
                row = selection["row"]
                tempo_class = selection["tempo_class"]
                directory = out_root / f"corpus_{tempo_class}_{position:02d}_{row['sample_id']}"
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    record = self._export(module, dataset, row, seconds, directory)
                    record["source"] = "corpus"
                except Exception as error:  # noqa: BLE001 - listening must not stop training
                    record = self._error_record(row, "corpus", error)
                record["tempo_class"] = tempo_class
                record["bpm"] = selection["bpm"]
                index.append(record)

            from .covers import available_covers

            kong_rows = available_covers(
                self.cfg,
                int(self.cfg.trainer.kong_sample_items),
                list(self.cfg.trainer.kong_sample_ids) or None,
            )
            bpm_by_id = dict(zip(
                list(self.cfg.trainer.kong_sample_ids),
                [float(value) for value in self.cfg.trainer.kong_sample_bpms],
                strict=True,
            ))
            measured = sorted(
                (bpm_by_id[row["sample_id"]], row["sample_id"])
                for row in kong_rows if row["sample_id"] in bpm_by_id
            )
            tempo_by_id = {
                track_id: "slow" if index == 0 else "fast" if index == len(measured) - 1 else "medium"
                for index, (_, track_id) in enumerate(measured)
            }
            for position, row in enumerate(kong_rows):
                tempo_class = tempo_by_id.get(row["sample_id"], "selected")
                directory = out_root / f"kong_{tempo_class}_{position:02d}_{row['sample_id']}"
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    record = self._export_kong(module, row, seconds, directory)
                except Exception as error:  # noqa: BLE001 - listening must not stop training
                    record = self._error_record(row, "kong", error)
                record["tempo_class"] = tempo_class
                record["bpm"] = bpm_by_id.get(row["sample_id"])
                index.append(record)
        finally:
            module.train(was_training)
        (out_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))

    @staticmethod
    def _error_record(row: dict, source: str, error: Exception) -> dict:
        return {
            "sample_id": row["sample_id"],
            "source": source,
            "error": f"{type(error).__name__}: {error}",
        }

    def _export(self, module, dataset, row: dict, seconds: float, directory: Path) -> dict:
        from .sample import decode_latent, sample_latent

        row_index = dataset.rows.index(row)
        item = dataset[(row_index, seconds)]
        frames = item["latent"].shape[0]
        batch = {
            "sample_id": [item["sample_id"]],
            "latent": item["latent"][None],
            "roll": item["roll"][None],
            "mask": torch.ones(1, frames, dtype=torch.bool),
            "seconds": torch.tensor([item["seconds"]]),
            "start_seconds": torch.tensor([item["start_seconds"]]),
        }

        generated = sample_latent(
            module,
            batch,
            steps=int(self.cfg.model.inference_steps),
            guidance=float(self.cfg.model.guidance),
            shift=float(self.cfg.model.shift),
        )
        bitrate = int(self.cfg.trainer.mp3_bitrate_kbps)
        rate = int(self.cfg.ace.sample_rate)

        audio = decode_latent(module, generated)
        save_mp3(audio, rate, directory / "generated.mp3", str(self.cfg.data.ffmpeg), bitrate)
        codec = decode_latent(module, batch["latent"].to(module.device))
        save_mp3(codec, rate, directory / "codec.mp3", str(self.cfg.data.ffmpeg), bitrate)

        try:
            source = decode_window(
                audio_path(row, self.cfg),
                float(item["start_seconds"]),
                float(item["seconds"]),
                rate,
                channels=int(self.cfg.ace.audio_channels),
                ffmpeg=str(self.cfg.data.ffmpeg),
            )
            save_mp3(source, rate, directory / "target.mp3", str(self.cfg.data.ffmpeg), bitrate)
        except Exception:  # noqa: BLE001 - the reference is a convenience, not the deliverable
            pass

        self._write_piano(row, item, directory, bitrate)
        return {
            "sample_id": row["sample_id"],
            "seconds": float(item["seconds"]),
            "start_seconds": float(item["start_seconds"]),
            "frames": int(frames),
        }

    def _export_kong(self, module, row: dict, seconds: float, directory: Path) -> dict:
        """Generate from Kong MIDI and retain the matching real pianist performance as context."""
        from .covers import listening_window_seed, load_listening_condition
        from .sample import build_batch, decode_latent, sample_latent

        notes, _, start = load_listening_condition(self.cfg, row, seconds)
        frames = latent_frames(self.cfg, seconds)
        batch = build_batch(self.cfg, notes, frames, [start])
        generated = sample_latent(
            module,
            batch,
            steps=int(self.cfg.model.inference_steps),
            guidance=float(self.cfg.model.guidance),
            shift=float(self.cfg.model.shift),
            seed=listening_window_seed(str(row["sample_id"])),
        )
        bitrate = int(self.cfg.trainer.mp3_bitrate_kbps)
        rate = int(self.cfg.ace.sample_rate)
        save_mp3(
            decode_latent(module, generated),
            rate,
            directory / "generated.mp3",
            str(self.cfg.data.ffmpeg),
            bitrate,
        )
        piano = decode_window(
            row["audio"], start, seconds, rate,
            channels=int(self.cfg.ace.audio_channels), ffmpeg=str(self.cfg.data.ffmpeg),
        )
        save_mp3(piano, rate, directory / "piano.mp3", str(self.cfg.data.ffmpeg), bitrate)
        write_midi(crop_notes(notes, start, seconds), directory / "piano.mid")
        return {
            "sample_id": row["sample_id"],
            "source": "kong",
            "seconds": seconds,
            "start_seconds": start,
            "frames": frames,
            "reference": "real pianist cover recording",
        }

    def _write_piano(self, row: dict, item: dict, directory: Path, bitrate: int) -> None:
        """What the model was told, as MIDI and as audio. Failure here is never fatal."""
        try:
            from .pianoroll import cached_notes

            notes = cached_notes(midi_path(row, self.cfg), str(notes_dir(self.cfg)))
            start = float(item["start_seconds"])
            write_midi(crop_notes(notes, start, float(item["seconds"])), directory / "piano.mid")
        except Exception:  # noqa: BLE001
            return
        try:
            require_executable(str(self.cfg.data.fluidsynth))
            wav = directory / "piano.wav"
            render_midi(
                directory / "piano.mid",
                wav,
                str(soundfont_path(self.cfg)),
                int(self.cfg.ace.sample_rate),
                fluidsynth=str(self.cfg.data.fluidsynth),
            )
            encode_mp3(wav, directory / "piano.mp3", str(self.cfg.data.ffmpeg), bitrate)
            wav.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - fluidsynth is optional
            pass
