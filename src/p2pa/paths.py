"""Where every artifact lives, and the fingerprints that make the cache self-invalidating.

Cache paths are content-addressed: the fingerprint hashes the parameters that produced the
artifact, so changing one of them changes the path and `stages.expected_completeness` sees the row
as incomplete again. Judging completeness from a path previously recorded on the row cannot do that.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from .config import CACHE_SCHEMA_VERSION, resolve_path
from .stages import fingerprint


def corpus_key(cfg: DictConfig) -> dict[str, str]:
    """Folded into every fingerprint so two corpora can share a directory without colliding."""
    return {"corpus": str(cfg.data.source.name)}


def manifest_path(cfg: DictConfig) -> Path:
    directory = resolve_path(str(cfg.data.manifest_dir))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cfg.data.source.name}.jsonl"


def inventory_path(cfg: DictConfig) -> Path:
    """The lossless corpus ledger; the operational manifest remains training-ready only."""
    directory = resolve_path(str(cfg.data.manifest_dir))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cfg.data.source.name}.inventory.jsonl"


def soundfont_path(cfg) -> Path:
    """The MuseScore soundfont, under the data root like everything else.

    Resolved rather than used raw because `data.soundfont` is relative by default; an absolute
    override still passes through `resolve_path` unchanged.
    """
    return resolve_path(str(cfg.data.soundfont))


def notes_dir(cfg: DictConfig) -> Path:
    """The parsed-MIDI cache, as an absolute path.

    `cfg.data.notes_dir` is a *relative* string (`.cache/notes`), and callers used to hand it to
    `cached_notes` verbatim — which resolves it against the process working directory. That is the
    project root when Hydra is configured not to chdir and something else entirely when it is, so
    the cache silently moved with the caller: warmed in one place by prep, re-parsed from scratch
    in another by training, and on a cluster it failed outright against a `.cache` that was not
    a directory. Every path in this project resolves through `resolve_path`; this one was the
    exception, and there was no reason for it.
    """
    directory = resolve_path(str(cfg.data.notes_dir))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def corpus_root(cfg: DictConfig) -> Path:
    return resolve_path(str(cfg.data.source.root))


def audio_path(row: dict, cfg: DictConfig) -> Path:
    source = cfg.data.source
    return corpus_root(cfg) / str(source.audio_subdir) / row["shard"] / row["track_id"] / str(
        source.instrumental_name
    )


def midi_dir(row: dict, cfg: DictConfig) -> Path:
    return corpus_root(cfg) / str(cfg.data.source.midi_subdir) / row["shard"] / row["track_id"]


def midi_path(row: dict, cfg: DictConfig) -> Path:
    return midi_dir(row, cfg) / str(cfg.data.source.baseline_midi)


def variant_midis(row: dict, cfg: DictConfig) -> list[Path]:
    """Every alternative conditioning MIDI, excluding the packaged copy of the baseline.

    A variant is named `<stem mix>-<decode target>.mid`; the decode targets in
    `source.excluded_variant_programs` are filtered out here rather than deleted, so the pool is a
    config decision and not a property of the disk. The baseline and picogen are named without a
    decode-target suffix that could collide with one, so neither can be excluded by accident.
    """
    baseline = str(cfg.data.source.baseline_midi)
    picogen = str(cfg.data.source.picogen_midi)
    excluded = {
        str(program) for program in getattr(cfg.data.source, "excluded_variant_programs", [])
    }
    return sorted(
        p for p in midi_dir(row, cfg).glob("*.mid")
        if p.name != baseline
        and (p.name == picogen or p.stem.rsplit("-", 1)[-1] not in excluded)
    )


def beat_path(row: dict, cfg: DictConfig) -> Path:
    return corpus_root(cfg) / str(cfg.data.source.beats_subdir) / row["shard"] / f"{row['track_id']}.json"


def latent_path(row: dict, cfg: DictConfig) -> Path:
    """One whole-track latent per song. Windows are sliced out of it, never cached themselves.

    Fingerprinted on the codec identity only: every row is the whole track, so unlike a per-window
    cache there is no start or length to vary.
    """
    key = fingerprint(
        {
            "schema": CACHE_SCHEMA_VERSION,
            **corpus_key(cfg),
            "audio": "instrumental_fullsong",
            "audio_revision": str(cfg.data.source.audio_revision),
            "autoencoder": f"{cfg.ace.assets_repo}/{cfg.ace.vae_subfolder}",
            "sample_rate": int(cfg.ace.sample_rate),
            "hop_length": int(cfg.ace.hop_length),
            "latent_channels": int(cfg.ace.latent_channels),
        }
    )[:10]
    directory = resolve_path(str(cfg.data.latent_dir)) / row["shard"]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{row['track_id']}-{key}.npy"


def null_prompt_path(cfg: DictConfig) -> Path:
    """The fixed lyric and text blocks every item is trained and sampled under. See `prompt.py`."""
    directory = resolve_path(str(cfg.data.prompt_dir)) / f"L{int(cfg.ace.prompt_max_length)}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cfg.data.source.name}-null.npz"


