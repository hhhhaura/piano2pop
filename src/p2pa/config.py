"""Structured configuration, and the contract every run is checked against before it costs anything.

Hydra composes `configs/` over the dataclasses below, so a key that is not declared here cannot be
set from the command line. `validate_config` runs first in every entrypoint: the expensive failures
in this project are all shape mismatches and resource mistakes that otherwise surface an hour into
a 500k-step run.

The paper's training mode is a full finetune of ACE-Step v1.5 with a convolutional roll encoder in
its `src_latents` slot, and its one axis is `setting`, which picks the conditioning augmentation.
`backbone=scratch` swaps the pretrained DiT for a small one trained from random initialisation
(`scratch.py`; not in the paper). `trainer.devices` must name exactly one device; see
`_validate_trainer`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

BACKBONES = ("acestep", "scratch")

# Manifest rows gained composition identity and provenance in v2.  Cache identity is kept
# separate: changing a bookkeeping schema must not force 3 GB of byte-identical VAE latents to be
# regenerated.
SCHEMA_VERSION = 2
CACHE_SCHEMA_VERSION = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Where generated data lives.

    `$P2PA_DATA_ROOT` names it — the directory holding `.cache`, `runs`, `tensorboards` and
    `data/p2pdata`, which is where `p2pa-unpack` puts them and what `scripts/setup.sh` writes into
    `env.sh`. That one indirection is the whole of the portability story for paths, on every
    machine; the project directory is only the fallback for a checkout that has generated nothing
    yet, which is what the test suite runs against.
    """
    override = os.environ.get("P2PA_DATA_ROOT", "").strip()
    return Path(override).expanduser().resolve() if override else project_root()


def config_dir() -> Path:
    return project_root() / "configs"


def resolve_path(value: str | Path) -> Path:
    """Anchor a relative path at the data root, so `.cache/...` follows the symlink or the env var."""
    path = Path(value)
    return path if path.is_absolute() else data_root() / path


@dataclass
class SourceConfig:
    # Names the manifest and is folded into every cache fingerprint, so two corpora cannot collide.
    name: str = "p2pdata"
    # The actual bytes behind `instrumental_name`. This is part of the latent fingerprint because
    # changing source-separation output while keeping track ids and filenames must invalidate every
    # target latent rather than silently mixing two targets in one run.
    audio_revision: str = "pop2piano_separated_v1"
    root: str = "data/p2pdata"
    audio_subdir: str = "audio"
    # Each track is a directory of stems; this is the one that is the training target.
    instrumental_name: str = "instrumental.mp3"
    midi_subdir: str = "midi"
    # The conditioning MIDI inside each track's midi directory. Everything else there is a variant.
    baseline_midi: str = "full-piano.mid"
    picogen_midi: str = "full-picogen.mid"
    beats_subdir: str = "beats"
    # Variant filenames are `<stem mix>-<decode target>.mid`. Drop these decode targets from the
    # pool: organ and strings are the two muscriptor variants whose right hand is not a piano
    # right hand — sustained block voicings rather than struck notes — so conditioning on them
    # teaches the model an articulation the piano side never produces. The files stay on disk;
    # this only removes them from the draw, so an ablation can put them back.
    excluded_variant_programs: list[str] = field(default_factory=lambda: ["organ", "strings"])


@dataclass
class DataConfig:
    manifest_dir: str = ".cache/manifests"
    latent_dir: str = ".cache/latents"
    prompt_dir: str = ".cache/prompt"
    notes_dir: str = ".cache/notes"
    # Real pianist cover recordings plus Kong transcriptions. `p2pa-unpack` writes this directory
    # at the data root; on the original workstation `p2pa.covers` recognizes the legacy location.
    covers_root: str = "covers"

    # Drop a drawn conditioning window with fewer usable notes than this.
    min_notes: int = 24

    train_fraction: float = 0.95
    validation_fraction: float = 0.03

    batch_size: int = 4
    # Four, not eight. Every worker holds its own arena, its own note-cache reads and its own BLAS
    # pool; this box has been taken down by the product of those before. See `dataset.py`.
    num_workers: int = 4
    # Bounded on purpose: `num_workers * prefetch_factor * batch_size` items are resident at once.
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    fluidsynth: str = "fluidsynth"
    # Relative, so `resolve_path` anchors it at the data root like every other artifact. An
    # absolute path into one machine's cache resolves nowhere else, and because the training-time
    # piano render is caught by design, it would fail silently.
    # Not in RESUME_CRITICAL, so changing it does not disturb a run in flight.
    soundfont: str = "assets/MuseScore_General.sf2"

    source: SourceConfig = field(default_factory=SourceConfig)


@dataclass
class AceConfig:
    """The frozen facts of the backbone, cross-checked against the checkpoint's own config."""

    # The DiT. `trust_remote_code` — the checkpoint ships its own modelling code.
    model_name: str = "ACE-Step/acestep-v15-base"
    # The Oobleck VAE and the frozen text encoder both live in the main ACE-Step repo.
    assets_repo: str = "ACE-Step/Ace-Step1.5"
    vae_subfolder: str = "vae"
    text_encoder_subfolder: str = "Qwen3-Embedding-0.6B"

    sample_rate: int = 48000
    audio_channels: int = 2
    # `audio_acoustic_hidden_dim` in the checkpoint's config: the VAE's latent width.
    latent_channels: int = 64
    # Samples per latent frame. 48000 / 1920 = exactly 25 Hz.
    hop_length: int = 1920
    # `in_channels`: src_latents(64) + chunk_masks(64) + x_t(64). Asserted at load.
    in_channels: int = 192
    # `proj_in` is a stride-`patch_size` Conv1d, so a window must be a whole number of patches.
    patch_size: int = 2
    # Qwen3-Embedding-0.6B's width, and what `AceStepConditionEncoder.text_projector` expects.
    prompt_dim: int = 1024
    # ACE-Step tokenizes the DiT prompt at 256. Every run feeds the same null prompt, encoded once.
    prompt_max_length: int = 256
    # The model's own classifier-free-guidance dropout rate, applied to the text/lyric/timbre pack
    # only. `src_latents` is never dropped, so sampling has a single guidance scale.
    cfg_ratio: float = 0.15
    # Reference-audio length the timbre encoder is built around (`timbre_fix_frame`).
    refer_frames: int = 750
    # Chunked VAE encode. A whole 4-minute track in one pass peaks at 36.5 GB of VRAM, measured —
    # unusable on a shared box. These two are verified length-exact and bit-close to a single pass
    # by `tests/test_prepare.py::test_chunked_encode_matches_one_pass`.
    chunk_frames: int = 512
    chunk_overlap_frames: int = 32


@dataclass
class RollConfig:
    pitch_low: int = 21
    pitch_high: int = 108
    # One roll frame per latent frame: the conv encoder has no strided stem to fold sub-frames with.
    oversample: int = 1
    frames_per_second: float = 48000 / 1920  # 25.0
    # Onset planes carry velocity; sustain planes are binary presence.
    velocity_in_onset: bool = True


@dataclass
class PrepConfig:
    shard: int = 0
    num_shards: int = 1
    reindex: bool = False
    encode_device: str = "cuda"
    # GM program 0 only, matching the corpus builder's own hard mask.
    programs: list[int] = field(default_factory=lambda: [0])
    retry_rejected: bool = False
    # Which stages `p2pa-prep` runs, in order. Named rather than positional so a re-run can do one
    # of them — `prep.stages=[screen]` re-applies the outlier thresholds without touching latents.
    stages: list[str] = field(
        default_factory=lambda: list(PREP_STAGES)
    )
    # Print the ranked densest conditioning MIDIs after the screen stage.
    report_outliers: bool = False

    # The corrupted-input fix. A track whose conditioning MIDI or whose target latent is
    # pathological is marked `status="rejected"` during prep rather than being left to spike the
    # loss 40k steps in.
    #
    # The caps below are set from the corpus's own measured distribution, not from intuition —
    # `p2pa-prep prep.stages=[screen] prep.report_outliers=true` prints it. Over these 4,382
    # tracks: notes/second reaches 21.3, simultaneous notes 58, the longest single note 22.4 s,
    # and duration_ratio bottoms out at 0.65. Every one of those is a continuous tail of dense
    # arrangements and early-ending transcriptions, not damage. So each cap sits *above* the
    # observed maximum: this stage exists to catch a decode that exploded, not to trim the busiest
    # 3% of a healthy corpus, and a filter that rejects 116 perfectly good tracks is worse than no
    # filter at all.
    reject_outliers: bool = True
    max_notes_per_second: float = 40.0
    max_simultaneous_notes: int = 64
    max_note_seconds: float = 45.0
    # |midi_span / audio_duration - 1| beyond this is a MIDI that does not describe its audio. A
    # ratio below 1 is usually just an instrumental outro the transcriber left empty, which the
    # window draw already routes around, so this is deliberately loose.
    max_duration_ratio_error: float = 0.4
    # The target side, and the one that actually predicts an MSE spike. These latents measure
    # std 0.80-1.09 across the whole corpus and |x| well inside this; anything past it is not
    # music, it is a bad encode.
    max_latent_abs: float = 25.0
    # On top of the absolute caps: reject beyond this many median-absolute-deviations from the
    # corpus median on notes/second. Robust, so one bad track cannot widen its own threshold.
    outlier_mad_z: float = 12.0
    # Track ids never to train on, whatever the statistics say.
    blocklist: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """The roll encoder, and how the finetuned model samples.

    The encoder is `p2p`'s dilation-free residual 1-D convolution stack. Its output projection is
    zero-initialised and biased at the checkpoint's own `silence_latent`, so an untrained encoder
    emits exactly the "no reference audio" latent and a fresh model is stock ACE-Step text2music.
    """

    # `acestep` (the paper's) or `scratch`. Chosen with the `backbone` config group, which also
    # sets the batch size and sampling numbers that go with it.
    backbone: str = "acestep"

    cond_dim: int = 512
    # Counts the stem: 5 is one stem plus four residual blocks.
    cond_layers: int = 5
    # Odd, so each convolution stays centred on its frame.
    cond_kernel: int = 5

    inference_steps: int = 30
    guidance: float = 7.0
    # ACE-Step's timestep shift at inference. 1.0 is the base model's own default.
    shift: float = 1.0


@dataclass
class FinetuneConfig:
    """A full finetune of all 2.39B DiT weights, in float32 masters.

    The backbone is frozen for the first `freeze_steps` optimizer steps while only the
    zero-initialised roll encoder trains, then unfrozen in place — no second run, no handoff.
    """

    freeze_steps: int = 2000
    # A full checkpoint is 27 GB, most of it AdamW moments; keep one monitored copy plus last.ckpt.
    save_top_k: int = 1
    lr_backbone: float = 1.0e-5
    lr_cond: float = 1.0e-4


@dataclass
class ScratchConfig:
    """`backbone=scratch`: `p2p`'s rectified-flow DiT, trained from random initialisation.

    About 220M parameters at these values. Every weight is new, so there is no freeze window,
    both parameter groups share one learning rate, and an EMA of the weights is kept and used for
    validation and sampling.
    """

    hidden_dim: int = 768
    num_heads: int = 12
    # Must be 0: the condition is frame-aligned and summed in, so there is no context sequence for
    # joint-attention blocks to read.
    mmdit_blocks: int = 0
    dit_blocks: int = 8
    mlp_ratio: float = 4.0
    # Classifier-free-guidance dropout of the roll. The finetune never drops it; this model learns
    # a null for it, which is what lets sampling guide on the roll.
    cond_dropout: float = 0.1
    logit_normal: bool = True
    lr: float = 1.0e-4
    ema_decay: float = 0.999
    save_top_k: int = 3


@dataclass
class AugmentConfig:
    """Per-segment source and style draw; active only with `window.segment_augment`.

    Source: the original transcription with `baseline_probability`, PiCoGen with
    `picogen_probability`, the remainder split evenly over the track's MuScriptor variants.
    Style: one of `normal` / `chord_thin` / `octave_move` / `octave_shift`, by these weights.
    """

    baseline_probability: float = 0.0
    picogen_probability: float = 0.0
    normal: float = 1.0
    chord_thin: float = 0.0
    octave_move: float = 0.0
    octave_shift: float = 0.0
    chord_tone_rate: list[float] = field(default_factory=lambda: [0.05, 0.2])
    tension_rate: list[float] = field(default_factory=lambda: [0.2, 0.6])
    octave_move_rate: list[float] = field(default_factory=lambda: [0.1, 0.5])
    octave_shift_choices: list[int] = field(default_factory=lambda: [-12, 0, 12])


@dataclass
class LengthConfig:
    """How long a training window is, as a function of the step.

    The ceiling holds at `min_seconds` for `hold_steps`, then rises by `ramp_seconds` every
    `ramp_every_steps` until it reaches `max_seconds`. The floor stays put, so the reachable range
    widens rather than slides. Every batch is one length, so no batch carries padding: without
    flash-attn the fallback attention leaves padded keys in the softmax denominator, and the DiT
    discards the attention mask anyway. Validation is always at `max_seconds`.
    """

    min_seconds: float = 20.0
    max_seconds: float = 30.0
    ramp_seconds: float = 5.0
    ramp_every_steps: int = 10_000
    hold_steps: int = 100_000


@dataclass
class WindowConfig:
    """How a window is cut out of a whole song, and what the conditioning for it is built from."""

    segment_bars: int = 4
    # The augmentation switch. Off by default: with it on and both source probabilities at 0, the
    # whole draw splits across the MuScriptor variants and the baseline is never drawn — a run that
    # meant to be unaugmented would quietly condition on generated MIDI.
    segment_augment: bool = False
    segment_max_simultaneous_notes: int = 25
    # A drawn window under data.min_notes — a silent intro or outro — is redrawn elsewhere in the
    # same song up to this many times. The item moves rather than being dropped: dropping from a
    # fixed-size batch would make the batch size vary. 0 disables the check.
    draw_attempts: int = 8
    # Each song contributes this many independently drawn windows to one pass over the corpus.
    # This only sets how often validation and sampling come round.
    segments_per_song: int = 10


@dataclass
class TrainerConfig:
    max_steps: int = 500_000
    weight_decay: float = 1.0e-2
    grad_clip: float = 1.0
    warmup_steps: int = 500
    precision: str = "bf16-mixed"
    accelerator: str = "auto"
    # Exactly one device. `_validate_trainer` rejects anything else — see the note there.
    devices: Any = field(default_factory=lambda: [0])
    # The 2.39B DiT, and the condition enters at `proj_in`, so gradient reaches the encoder through
    # all 24 layers whether or not the backbone is frozen. Checkpointing is the memory lever here,
    # not freezing, which is why it defaults on.
    gradient_checkpointing: bool = True
    log_every_n_steps: int = 20
    auto_resume: bool = True

    val_check_interval: int = 2000
    limit_val_batches: Any = 40
    # Validation loss is averaged over this many fixed timesteps with a per-item seed, so the
    # monitored metric is deterministic instead of a one-sample estimate of an expectation.
    val_timesteps: int = 5

    sample_items: int = 2
    # A separate listening-only OOD subset: real pianist recordings conditioned on their Kong
    # transcriptions. These never enter `val/loss` or checkpoint selection.
    kong_sample_items: int = 2
    # BeatThis-measured pair, ordered slow then fast. Keeping BPM beside the ids makes the
    # listening-panel choice explicit and gives `index.json` useful labels on a portable machine
    # that does not install the beat detector.
    kong_sample_ids: list[str] = field(
        default_factory=lambda: ["0TMG60NMShM", "00wvcg5SmFs"]
    )
    kong_sample_bpms: list[float] = field(default_factory=lambda: [63.8, 130.4])
    sample_every_n_steps: int = 10_000
    sample_ids: list[str] = field(default_factory=list)
    mp3_bitrate_kbps: int = 192

    accumulate_grad_batches: int = 1

    # A rehearsal: run this many steps, print peak VRAM, peak RSS and ms/step, and stop without
    # writing a checkpoint. 0 is a real run. Always do one before committing a GPU for 250k steps.
    dry_run_steps: int = 0

    # --- guards ---------------------------------------------------------------------------
    # Raise, rather than let the OOM killer decide, once this process tree's resident set passes
    # the limit. 0 disables. Measured and logged every `log_every_n_steps`.
    rss_limit_gb: float = 64.0
    # A step whose loss exceeds `factor` x the running median is logged with its sample ids and
    # skipped. A corrupted input becomes an identified track instead of a mystery in the curve.
    loss_spike_factor: float = 8.0
    loss_spike_warmup: int = 200


@dataclass
class TrainConfig:
    exp_name: str = "p2pa_v1"
    seed: int = 42
    runs_dir: str = "runs"
    tensorboard_dir: str = "tensorboards"

    data: DataConfig = field(default_factory=DataConfig)
    ace: AceConfig = field(default_factory=AceConfig)
    roll: RollConfig = field(default_factory=RollConfig)
    prep: PrepConfig = field(default_factory=PrepConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    scratch: ScratchConfig = field(default_factory=ScratchConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    length: LengthConfig = field(default_factory=LengthConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


# --- derived quantities ---------------------------------------------------------------------


def latent_fps(cfg: DictConfig | TrainConfig) -> float:
    """Latent frames per second. 48000 / 1920 = exactly 25."""
    return float(cfg.ace.sample_rate) / float(cfg.ace.hop_length)


def roll_fps(cfg: DictConfig | TrainConfig) -> float:
    return latent_fps(cfg) * int(cfg.roll.oversample)


def latent_frames(cfg: DictConfig | TrainConfig, seconds: float) -> int:
    """Latent frames for a duration, rounded down to a whole number of DiT patches.

    `proj_in` is a stride-`patch_size` convolution and the model zero-pads a ragged tail before it.
    Padding the *target* would make the loss include frames that describe nothing, so windows are
    quantised instead and the pad path is never taken.
    """
    patch = int(cfg.ace.patch_size)
    frames = int(round(float(seconds) * latent_fps(cfg)))
    frames -= frames % patch
    return max(patch, frames)


def latent_samples(cfg: DictConfig | TrainConfig, frames: int) -> int:
    return int(frames) * int(cfg.ace.hop_length)


def window_seconds(cfg: DictConfig | TrainConfig, frames: int) -> float:
    return float(frames) / latent_fps(cfg)


# The stages `p2pa-prep` knows, in the order it runs them.
PREP_STAGES = ("index", "screen", "encode", "notes", "dataset")


def _known(node: Any, schema: Any) -> Any:
    """`node` restricted to the keys `schema` declares, recursively."""
    if not isinstance(node, dict) or not isinstance(schema, dict):
        return node
    return {key: _known(value, schema[key]) for key, value in node.items() if key in schema}


def migrate_config(cfg: DictConfig) -> DictConfig:
    """Merge a config written by an older run over today's defaults, so sampling one keeps working.

    Runs recorded under the older schema carry `phase.*`, with `phase.name` one of `full`, `lora`
    or `scratch`, and keys for modes that no longer exist (LoRA, captions, CLAP, the vocal plane).
    A `full` run's `phase.*` becomes `finetune.*`; a `scratch` run's transformer and optimiser
    keys become `scratch.*`. The rest are dropped, which is exact for every full-finetune and
    from-scratch run: those keys were all at their off values there.
    """
    from omegaconf import OmegaConf

    recorded = OmegaConf.to_container(cfg, resolve=True)
    phase = recorded.pop("phase", None)
    if isinstance(phase, dict) and phase.get("name") == "scratch":
        model = recorded.setdefault("model", {})
        model["backbone"] = "scratch"
        if float(phase["lr_backbone"]) != float(phase["lr_cond"]):
            raise ValueError("a scratch run trains both parameter groups at one learning rate")
        recorded.setdefault("scratch", {
            **{key: model[key] for key in (
                "hidden_dim", "num_heads", "mmdit_blocks", "dit_blocks", "mlp_ratio",
                "cond_dropout", "logit_normal",
            ) if key in model},
            "lr": phase["lr_backbone"],
            "save_top_k": phase["save_top_k"],
            "ema_decay": (recorded.get("trainer") or {}).get("ema_decay", 0.999),
        })
    elif isinstance(phase, dict) and "finetune" not in recorded:
        recorded["finetune"] = phase
    stages = (recorded.get("prep") or {}).get("stages")
    if isinstance(stages, list):
        recorded["prep"]["stages"] = [stage for stage in stages if stage in PREP_STAGES]
    schema = OmegaConf.to_container(OmegaConf.structured(TrainConfig))
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), _known(recorded, schema))
    OmegaConf.set_struct(merged, True)
    return merged


# --- the contract ---------------------------------------------------------------------------


def validate_config(cfg: DictConfig) -> None:
    _validate_paths(cfg)
    _validate_roll(cfg)
    _validate_model(cfg)
    _validate_finetune(cfg)
    _validate_scratch(cfg)
    _validate_length(cfg)
    _validate_trainer(cfg)


def _validate_paths(cfg: DictConfig) -> None:
    root = resolve_path(str(cfg.data.source.root))
    if not root.exists():
        raise ValueError(
            f"data.source.root does not exist: {root}. On a machine that has just unpacked the "
            "data tarballs, set $P2PA_DATA_ROOT to the directory they were unpacked into."
        )
    if not str(cfg.data.source.name):
        raise ValueError("data.source.name must be set; it namespaces every cache path")


def _validate_roll(cfg: DictConfig) -> None:
    if int(cfg.roll.pitch_low) >= int(cfg.roll.pitch_high):
        raise ValueError("roll.pitch_low must be below roll.pitch_high")
    # The conv encoder has no strided stem, so the roll has to arrive at the latent rate already.
    # Without this a 4x roll would be read as 4x the music at 4x the tempo, and nothing downstream
    # would have a shape to complain about.
    if int(cfg.roll.oversample) != 1:
        raise ValueError(f"roll.oversample must be 1, not {cfg.roll.oversample}")
    if abs(float(cfg.roll.frames_per_second) - latent_fps(cfg)) > 1e-9:
        raise ValueError(
            f"roll.frames_per_second is {cfg.roll.frames_per_second}, but the latent rate is "
            f"{latent_fps(cfg):.6f}. Any other rate silently misaligns the condition against the "
            "audio it describes."
        )


def _validate_model(cfg: DictConfig) -> None:
    model = cfg.model
    if int(model.cond_layers) < 1:
        raise ValueError("model.cond_layers counts the stem, so it must be at least 1")
    if int(model.cond_kernel) % 2 == 0:
        raise ValueError(
            f"model.cond_kernel={model.cond_kernel} must be odd, so each convolution stays "
            "centred on its frame"
        )
    if int(model.cond_dim) % 8:
        raise ValueError("model.cond_dim must divide into the encoder's 8 GroupNorm groups")


def _validate_scratch(cfg: DictConfig) -> None:
    backbone = str(cfg.model.backbone)
    if backbone not in BACKBONES:
        raise ValueError(f"model.backbone must be one of {BACKBONES}, not {backbone!r}")
    if backbone != "scratch":
        return
    scratch = cfg.scratch
    if int(scratch.mmdit_blocks) != 0:
        raise ValueError(
            "scratch.mmdit_blocks must be 0: the roll is summed into the residual stream, so there "
            "is no context sequence for joint-attention blocks to read"
        )
    if int(scratch.dit_blocks) < 1:
        raise ValueError("scratch.dit_blocks must be at least 1")
    if int(scratch.hidden_dim) % int(scratch.num_heads):
        raise ValueError("scratch.hidden_dim must divide into scratch.num_heads")
    if not 0.0 <= float(scratch.cond_dropout) < 1.0:
        raise ValueError("scratch.cond_dropout must be in [0, 1)")
    if not 0.0 < float(scratch.ema_decay) <= 1.0:
        raise ValueError("scratch.ema_decay must be in (0, 1]")
    if math.isnan(float(scratch.lr)) or float(scratch.lr) <= 0:
        raise ValueError("scratch.lr must be positive")
    # The scratch sampler guides linearly, not with ACE-Step's momentum-buffered APG, and plain
    # linear guidance at ACE-Step's scale of 7 over-saturates. `backbone=scratch` sets 2.0.
    if float(cfg.model.guidance) > 4.0:
        raise ValueError(
            f"model.guidance={cfg.model.guidance} is meant for ACE-Step's APG; the scratch "
            "backbone uses linear guidance, where that over-saturates. Select `backbone=scratch` "
            "(guidance 2.0) or set model.guidance at most 4."
        )


def _validate_finetune(cfg: DictConfig) -> None:
    freeze = int(cfg.finetune.freeze_steps)
    if freeze < 0:
        raise ValueError("finetune.freeze_steps must not be negative")
    if freeze >= int(cfg.trainer.max_steps):
        raise ValueError(
            f"finetune.freeze_steps={freeze} never ends inside "
            f"trainer.max_steps={cfg.trainer.max_steps}"
        )
    for name in ("lr_cond", "lr_backbone"):
        value = float(cfg.finetune[name])
        if math.isnan(value) or value <= 0:
            raise ValueError(f"finetune.{name} must be positive")


def _validate_length(cfg: DictConfig) -> None:
    length = cfg.length
    minimum = float(length.min_seconds)
    maximum = float(length.max_seconds)
    if minimum <= 0 or maximum < minimum:
        raise ValueError("length needs 0 < min_seconds <= max_seconds")
    if latent_frames(cfg, minimum) < 8:
        raise ValueError(
            f"length.min_seconds={minimum} is only {latent_frames(cfg, minimum)} latent frames"
        )
    if maximum > minimum:
        if float(length.ramp_seconds) <= 0:
            raise ValueError("length.ramp_seconds must be positive when max_seconds > min_seconds")
        if int(length.ramp_every_steps) < 1:
            raise ValueError("length.ramp_every_steps must be at least 1")
        stages = math.ceil((maximum - minimum) / float(length.ramp_seconds))
        reached = int(length.hold_steps) + stages * int(length.ramp_every_steps)
        if reached >= int(cfg.trainer.max_steps):
            raise ValueError(
                f"the window length reaches {maximum:g}s only at step {reached}, at or past "
                f"trainer.max_steps={cfg.trainer.max_steps}. The model would never train at full "
                "length, and validation — always at max_seconds — would measure a length it has "
                "never seen."
            )


def _validate_trainer(cfg: DictConfig) -> None:
    trainer = cfg.trainer
    if int(trainer.val_timesteps) < 1:
        raise ValueError("trainer.val_timesteps must be at least 1")
    if int(trainer.sample_items) < 0 or int(trainer.kong_sample_items) < 0:
        raise ValueError("trainer sample counts must not be negative")
    if len(trainer.kong_sample_ids) != len(trainer.kong_sample_bpms):
        raise ValueError("trainer.kong_sample_ids and kong_sample_bpms must have equal lengths")
    if int(trainer.max_steps) < 1:
        raise ValueError("trainer.max_steps must be at least 1")

    devices = trainer.devices
    # A ListConfig is not a list, and `isinstance(..., list)` quietly said "one device" for
    # `[0, 1]` until a test caught it. Anything with a length that is not a string is a list here.
    if isinstance(devices, str):
        count = -1 if devices == "auto" else 1
    elif hasattr(devices, "__len__"):
        count = len(devices)
    else:
        count = 1
    if count != 1:
        raise ValueError(
            f"trainer.devices must name exactly one GPU, got {devices!r}. Runs are single-GPU on "
            "purpose: the batch sampler draws one window length per batch and is not rank-aware, "
            "so DDP would hand every rank the same list. Run settings side by side on different "
            "GPUs instead of one run across several."
        )

    if int(cfg.data.num_workers) < 0:
        raise ValueError("data.num_workers must not be negative")
    if float(trainer.rss_limit_gb) < 0:
        raise ValueError("trainer.rss_limit_gb must not be negative")
