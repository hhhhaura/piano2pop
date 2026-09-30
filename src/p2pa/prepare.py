"""Build the training cache: one manifest row and one whole-track ACE-Step latent per song.

There are no windows here. The sampler draws its own length and offset at load time, so what the
cache holds is the whole instrumental encoded once, and slicing is a view over it.

Every stage is resumable, shard-aware and content-addressed through `stages.py`: re-running skips
what is already valid on disk, and changing a parameter that fed a fingerprint changes the path,
which makes the affected rows incomplete again without any explicit invalidation step.

Stages, in order: `index` (one manifest row per track, with its split), `screen` (outlier
rejection), `encode` (the whole-track VAE latents), `notes` (the parsed-MIDI cache) and `dataset`
(the final manifest).

**Outlier rejection.** A single corrupted conditioning MIDI produced the loss spikes that ruined a
`p2p-stable` run, and the only evidence was a vertical line on a curve. This stage measures every
track's MIDI and rejects the pathological ones before training ever sees them, with the reason on
the row. `--report-outliers` prints the ranked offenders, because "which track was it" should be
answerable without re-running anything.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm

from .audio import decode_window, probe_duration
from .config import (
    SCHEMA_VERSION,
    TrainConfig,
    config_dir,
    latent_fps,
    resolve_path,
    validate_config,
)
from .paths import (
    audio_path,
    corpus_root,
    inventory_path,
    latent_path,
    manifest_path,
    midi_path,
    notes_dir,
    variant_midis,
)
from .pianoroll import cached_notes, usable_notes
from .stages import (
    artifact_ready,
    atomic_output,
    path_lock,
    read_jsonl,
    run_stage,
    stable_split,
    write_jsonl,
)

# Below these a track is not worth a row; both are checked on a cheap decode before any GPU work.
MIN_SECONDS = 5.0
MIN_RMS = 1e-4


# --- stage 1: index ---------------------------------------------------------


def index_tracks(cfg: DictConfig) -> dict[str, int]:
    """One row per track that has both an instrumental stem and a baseline conditioning MIDI."""
    root = corpus_root(cfg)
    source = cfg.data.source
    candidates = []
    inventory = read_jsonl(inventory_path(cfg)) if inventory_path(cfg).is_file() else []
    if inventory:
        # Inventory is the authority on identity and duplicates.  Filesystem globbing alone is
        # exactly what reintroduced a second upload of the same composition in the old index.
        for item in inventory:
            # A track is ready when its instrumental and piano transcription both exist; the
            # inventory's own state field records collection, not processing, and is not consulted.
            row = {
                key: item[key]
                for key in (
                    "shard", "track_id", "pop_id", "artist", "song", "name", "language",
                    "identity_confidence", "official_instrumental", "split",
                )
                if key in item
            }
            if audio_path(row, cfg).is_file() and midi_path(row, cfg).is_file():
                candidates.append(row)
    else:
        for track in sorted((root / str(source.audio_subdir)).glob("*/*")):
            if not track.is_dir():
                continue
            row = {"shard": track.parent.name, "track_id": track.name}
            if audio_path(row, cfg).is_file() and midi_path(row, cfg).is_file():
                candidates.append(row)

    # ffprobe reads a container header, so each call is fast but dominated by process startup;
    # 4k of them run in well under a minute spread over threads and several minutes in sequence.
    with ThreadPoolExecutor(max_workers=max(4, int(cfg.data.num_workers) * 4)) as pool:
        durations = list(
            tqdm(
                pool.map(
                    lambda row: probe_duration(audio_path(row, cfg), str(cfg.data.ffprobe)),
                    candidates,
                ),
                total=len(candidates),
                desc="index",
                unit="track",
            )
        )

    rows = []
    for row, duration in zip(candidates, durations):
        if duration < MIN_SECONDS:
            continue
        indexed = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": row["track_id"],
            "track_id": row["track_id"],
            "shard": row["shard"],
            "duration": round(float(duration), 3),
            # The released track list carries the paper's split; anything else is split by name.
            "split": row.get("split") or stable_split(
                row.get("name") or row["track_id"],
                int(cfg.seed),
                float(cfg.data.train_fraction),
                float(cfg.data.validation_fraction),
            ),
            "status": "pending",
        }
        indexed.update({
            key: row[key]
            for key in (
                "pop_id", "artist", "song", "name", "language", "identity_confidence",
                "official_instrumental",
            )
            if key in row
        })
        rows.append(indexed)
    rows.sort(key=lambda row: row["track_id"])

    manifest = manifest_path(cfg)
    with path_lock(manifest):
        existing = read_jsonl(manifest) if manifest.is_file() else []
        if existing and not bool(cfg.prep.reindex):
            # Progress is keyed by sample_id so a re-index never discards finished work.
            carried = {row["sample_id"]: row for row in existing}
            state = (
                "status", "reason", "latent", "latent_frames", "latent_std", "latent_max_abs",
                "latent_finite", "latent_source", "midi_stats",
            )
            for row in rows:
                previous = carried.get(row["sample_id"])
                if previous is not None:
                    row.update({key: previous[key] for key in state if key in previous})
        write_jsonl(manifest, rows)

    splits: dict[str, int] = {}
    for row in rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    return {
        "rows": len(rows),
        "candidates": len(candidates),
        **splits,
    }


# --- stage 2: reject pathological conditioning MIDI -------------------------


def midi_statistics(row: dict, cfg: DictConfig) -> dict:
    """What a corrupted MIDI looks like, measured.

    All five are properties of the *notes*, not of the file format — a MIDI that parses cleanly can
    still be nonsense. `notes_per_second` and `max_simultaneous` catch a decode that emitted a wall
    of pitches; `max_note_seconds` catches a note-off that never arrived, which turns one note into
    a drone across the whole track; `duration_ratio` catches a MIDI whose timeline does not
    describe its audio at all, which is the one that produces a roll aligned to nothing.
    """
    notes = cached_notes(midi_path(row, cfg), str(notes_dir(cfg)))
    kept = usable_notes(notes, int(cfg.roll.pitch_low), int(cfg.roll.pitch_high), None)
    duration = max(1e-6, float(row["duration"]))
    if not kept:
        return {
            "notes": 0, "notes_per_second": 0.0, "max_simultaneous": 0,
            "max_note_seconds": 0.0, "span": 0.0, "duration_ratio": 0.0,
        }

    span = max(note.end for note in kept)
    # A sweep over onsets and offsets: +1 at every start, -1 at every end, running maximum.
    events = sorted(
        [(note.start, 1) for note in kept] + [(note.end, -1) for note in kept]
    )
    live = peak = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return {
        "notes": len(kept),
        "notes_per_second": round(len(kept) / duration, 3),
        "max_simultaneous": int(peak),
        "max_note_seconds": round(max(note.end - note.start for note in kept), 3),
        "span": round(float(span), 3),
        "duration_ratio": round(float(span) / duration, 4),
    }


def outlier_reasons(stats: dict, cfg: DictConfig, mad_threshold: float | None) -> list[str]:
    reasons = []
    if stats["notes"] == 0:
        reasons.append("no_usable_notes")
    if stats["notes_per_second"] > float(cfg.prep.max_notes_per_second):
        reasons.append(f"dense:{stats['notes_per_second']:.1f}nps")
    if stats["max_simultaneous"] > int(cfg.prep.max_simultaneous_notes):
        reasons.append(f"polyphony:{stats['max_simultaneous']}")
    if stats["max_note_seconds"] > float(cfg.prep.max_note_seconds):
        reasons.append(f"stuck_note:{stats['max_note_seconds']:.0f}s")
    error = abs(stats["duration_ratio"] - 1.0)
    if stats["notes"] and error > float(cfg.prep.max_duration_ratio_error):
        reasons.append(f"duration_ratio:{stats['duration_ratio']:.2f}")
    if mad_threshold is not None and stats["notes_per_second"] > mad_threshold:
        reasons.append(f"nps_outlier:{stats['notes_per_second']:.1f}>{mad_threshold:.1f}")
    return reasons


def screen_midi(cfg: DictConfig, *, report: bool = False) -> dict:
    """Measure every track's conditioning MIDI, then reject the pathological ones.

    Two passes on purpose. The absolute caps could be applied per row, but the robust one cannot:
    it needs the corpus median, and a threshold derived from a median and a MAD is exactly what one
    catastrophic outlier must not be allowed to widen.
    """
    manifest = manifest_path(cfg)
    rows = read_jsonl(manifest)
    blocked = {str(name) for name in cfg.prep.blocklist}

    def measure(row: dict) -> dict:
        try:
            return midi_statistics(row, cfg)
        except Exception as error:  # noqa: BLE001 - an unparseable MIDI is itself the finding
            return {"error": f"{type(error).__name__}: {error}", "notes": 0,
                    "notes_per_second": 0.0, "max_simultaneous": 0,
                    "max_note_seconds": 0.0, "span": 0.0, "duration_ratio": 0.0}

    with ThreadPoolExecutor(max_workers=max(4, int(cfg.data.num_workers) * 4)) as pool:
        stats = list(tqdm(pool.map(measure, rows), total=len(rows), desc="screen", unit="track"))

    density = np.array([s["notes_per_second"] for s in stats if s["notes"]], dtype=np.float64)
    mad_threshold = None
    if density.size and float(cfg.prep.outlier_mad_z) > 0:
        median = float(np.median(density))
        mad = float(np.median(np.abs(density - median)))
        if mad > 0:
            # 1.4826 * MAD is the consistent estimator of sigma for a normal distribution.
            mad_threshold = median + float(cfg.prep.outlier_mad_z) * 1.4826 * mad

    counts = {"rejected": 0, "kept": 0, "blocked": 0}
    reasons: dict[str, int] = {}
    for row, stat in zip(rows, stats):
        row["midi_stats"] = stat
        found = outlier_reasons(stat, cfg, mad_threshold) if bool(cfg.prep.reject_outliers) else []
        if row["track_id"] in blocked:
            found = ["blocklist", *found]
            counts["blocked"] += 1
        if found:
            row["status"] = "rejected"
            row["reason"] = "midi_outlier:" + ",".join(found)
            counts["rejected"] += 1
            reasons[found[0].split(":")[0]] = reasons.get(found[0].split(":")[0], 0) + 1
        else:
            counts["kept"] += 1
            if str(row.get("reason", "")).startswith("midi_outlier"):
                # A row rejected by an earlier, stricter threshold is allowed back in.
                row["status"] = "pending"
                row.pop("reason", None)

    with path_lock(manifest):
        write_jsonl(manifest, rows)

    if report:
        ranked = sorted(
            zip(rows, stats), key=lambda pair: pair[1]["notes_per_second"], reverse=True
        )[:20]
        print("\n[screen] the twenty densest conditioning MIDIs in the corpus:", file=sys.stderr)
        print(f"{'track_id':<16} {'nps':>8} {'poly':>5} {'longest':>9} {'dur_ratio':>10}  status",
              file=sys.stderr)
        for row, stat in ranked:
            print(
                f"{row['track_id']:<16} {stat['notes_per_second']:>8.1f} "
                f"{stat['max_simultaneous']:>5} {stat['max_note_seconds']:>8.1f}s "
                f"{stat['duration_ratio']:>10.2f}  {row.get('reason', row['status'])}",
                file=sys.stderr,
            )
    return {**counts, "mad_threshold": mad_threshold, "by_reason": reasons}


# --- stage 3: latents -------------------------------------------------------


def latent_valid(path: Path, cfg: DictConfig) -> bool:
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return array.ndim == 2 and array.shape[0] == int(cfg.ace.latent_channels) and array.shape[1] > 0


def latent_summary(array: np.ndarray) -> dict:
    """std and peak magnitude of a whole-track latent.

    The peak is the one that matters. The loss is a mean square error against this tensor, so a
    single wild frame in a 6,000-frame track is enough to produce the kind of vertical line on a
    curve that cost `p2p-stable` a run — and it is invisible in the std.
    """
    values = np.asarray(array, dtype=np.float32)
    return {
        "latent_std": round(float(values.std()), 6),
        "latent_max_abs": round(float(np.abs(values).max()), 4),
        "latent_finite": bool(np.isfinite(values).all()),
    }


def latent_reason(summary: dict, cfg: DictConfig) -> str:
    if not summary["latent_finite"]:
        return "latent_non_finite"
    if summary["latent_max_abs"] > float(cfg.prep.max_latent_abs):
        return f"latent_outlier:max_abs:{summary['latent_max_abs']:.1f}"
    return ""


def encode_complete(row: dict, cfg: DictConfig) -> bool:
    """Recompute the expected path from the *current* config, then check the artifact is real.

    Content-addressed naming is only self-invalidating if completeness is judged from a freshly
    computed path rather than the one recorded on the row, and presence is only proof if the file
    also parses — an interrupted process leaves truncated `.npy` files behind.
    """
    if row.get("status") == "rejected":
        return not bool(cfg.prep.retry_rejected)
    return artifact_ready(latent_path(row, cfg), lambda path: latent_valid(path, cfg))


def load_vae_for_prep(cfg: DictConfig):
    from .ace import load_vae

    return load_vae(cfg, str(cfg.prep.encode_device))


@torch.no_grad()
def encode_waveform(vae, waveform: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Encode a waveform cropped to an exact multiple of the hop length.

    Cropping first means latent frame `i` covers audio samples `[i * hop, (i + 1) * hop)` with no
    drift, which is what lets the dataset turn a frame offset back into the seconds the roll is cut
    at. The frame count is asserted rather than trusted.

    The distribution *mean* is taken, never a sample, so a cached latent is deterministic — a
    resumed prep must not produce a different target for a track it already encoded.
    """
    hop = int(cfg.ace.hop_length)
    frames = waveform.shape[-1] // hop
    if frames < 1:
        raise ValueError(f"waveform of {waveform.shape[-1]} samples is under one latent frame")
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    cropped = waveform[:, : frames * hop].to(device=device, dtype=dtype)

    chunk = int(cfg.ace.chunk_frames)
    overlap = int(cfg.ace.chunk_overlap_frames)
    pieces = []
    for start in range(0, frames, chunk) if chunk > 0 else [0]:
        stop = min(frames, start + chunk) if chunk > 0 else frames
        # Encode with context on both sides and keep only the middle. The encoder is a stack of
        # strided convolutions, so a frame near a chunk edge is built from a truncated receptive
        # field; the overlap is what makes every *kept* frame identical to the one a single pass
        # would have produced.
        left = max(0, start - overlap)
        right = min(frames, stop + overlap)
        window = cropped[:, left * hop : right * hop]
        encoded = vae.encode(window[None]).latent_dist.mean[0]
        if encoded.shape[-1] != right - left:
            raise RuntimeError(
                f"the VAE returned {encoded.shape[-1]} frames for {(right - left) * hop} samples, "
                f"expected {right - left}"
            )
        pieces.append(encoded[:, start - left : start - left + (stop - start)].float().cpu())
        if stop >= frames:
            break

    latent = torch.cat(pieces, dim=-1)
    if latent.shape[-1] != frames:
        raise RuntimeError(
            f"chunked encode produced {latent.shape[-1]} frames, expected {frames}"
        )
    return latent


def read_audio(row: dict, cfg: DictConfig) -> torch.Tensor:
    return decode_window(
        audio_path(row, cfg),
        0.0,
        float(row["duration"]) + 1.0,
        int(cfg.ace.sample_rate),
        channels=int(cfg.ace.audio_channels),
        ffmpeg=str(cfg.data.ffmpeg),
    )


def encode(cfg: DictConfig) -> dict[str, dict[str, int]]:
    """Encode every usable track's whole instrumental to a `[64, T]` float16 latent."""
    holder: dict[str, object] = {}

    def reject(row: dict, reason: str) -> None:
        row["status"] = "rejected"
        row["reason"] = reason

    def process(row: dict, cfg: DictConfig) -> None:
        if "vae" not in holder:
            holder["vae"] = load_vae_for_prep(cfg)

        try:
            waveform = read_audio(row, cfg)
        except Exception as error:  # noqa: BLE001 - one unreadable file must not stop a shard
            reject(row, f"decode_failed:{type(error).__name__}")
            return

        seconds = waveform.shape[-1] / float(cfg.ace.sample_rate)
        if seconds < MIN_SECONDS:
            reject(row, f"short_audio:{seconds:.2f}")
            return
        rms = float(waveform.float().pow(2).mean().sqrt())
        if rms < MIN_RMS:
            reject(row, f"silent_audio:{rms:.6f}")
            return

        latent = encode_waveform(holder["vae"], waveform, cfg).to(torch.float16).cpu()
        summary = latent_summary(latent.numpy())
        reason = latent_reason(summary, cfg) if bool(cfg.prep.reject_outliers) else ""
        if reason:
            row.update(summary)
            reject(row, reason)
            return

        output = latent_path(row, cfg)
        with atomic_output(output) as temporary:
            # np.save appends `.npy` to a path that does not end in it, which would defeat the
            # atomic rename; writing through the handle keeps the temp name intact.
            with open(temporary, "wb") as handle:
                np.save(handle, latent.numpy())
        row["latent"] = str(output.relative_to(resolve_path(str(cfg.data.latent_dir))))
        row["latent_frames"] = int(latent.shape[-1])
        row.update(summary)
        row["latent_source"] = "encoded"
        row["status"] = "ok"
        row.pop("reason", None)

    return run_stage(
        "encode",
        resolve_path(str(cfg.data.manifest_dir)),
        cfg,
        encode_complete,
        process,
        flush_every=10,
    )


# --- stage 4: warm the note cache -------------------------------------------


def warm_notes(cfg: DictConfig) -> dict[str, int]:
    """Pre-parse every conditioning MIDI, including the sung melody, into `.cache/notes`.

    Not strictly necessary — the dataset populates this cache lazily — but the first thousand steps
    would otherwise spend their time parsing several MIDI files per track and writing the results
    to a contended disk, which reads as a mysteriously slow start rather than as one-time setup.
    """
    rows = read_jsonl(manifest_path(cfg))
    warm_dir = str(notes_dir(cfg))

    def warm(row: dict) -> int:
        paths = [midi_path(row, cfg), *variant_midis(row, cfg)]
        parsed = 0
        for path in paths:
            if not path.is_file():
                continue
            try:
                cached_notes(path, warm_dir)
                parsed += 1
            except Exception:  # noqa: BLE001 - an unreadable variant is the dataset's problem later
                pass
        return parsed

    with ThreadPoolExecutor(max_workers=max(4, int(cfg.data.num_workers) * 4)) as pool:
        parsed = list(tqdm(pool.map(warm, rows), total=len(rows), desc="notes", unit="track"))
    return {"tracks": len(rows), "files": sum(parsed)}


# --- entrypoint -------------------------------------------------------------


def finalize(cfg: DictConfig) -> dict:
    rows = read_jsonl(manifest_path(cfg))
    usable = [row for row in rows if row.get("status") == "ok"]
    rejected: dict[str, int] = {}
    for row in rows:
        if row.get("status") == "rejected":
            key = str(row.get("reason", "unknown")).split(":")[0]
            rejected[key] = rejected.get(key, 0) + 1
    frames = [int(row["latent_frames"]) for row in usable if "latent_frames" in row]
    floor = float(cfg.length.max_seconds)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "corpus": str(cfg.data.source.name),
        "audio_revision": str(cfg.data.source.audio_revision),
        "rows": len(rows),
        "usable": len(usable),
        "rejected": rejected,
        "shorter_than_max_window": sum(1 for row in usable if float(row["duration"]) < floor),
        "latent_channels": int(cfg.ace.latent_channels),
        "latent_fps": latent_fps(cfg),
        "roll_fps": float(cfg.roll.frames_per_second),
        "vae": f"{cfg.ace.assets_repo}/{cfg.ace.vae_subfolder}",
        "total_hours": round(sum(frames) / latent_fps(cfg) / 3600.0, 2) if frames else 0.0,
        "splits": {
            split: sum(1 for row in usable if row["split"] == split)
            for split in ("train", "validation", "test")
        },
    }
    target = resolve_path(str(cfg.data.manifest_dir)) / f"{cfg.data.source.name}.metadata.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    temporary.replace(target)
    return summary


def announce(stage: str, result) -> object:
    print(f"[{stage}] {json.dumps(result, sort_keys=True, default=str)}")
    return result


ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)


@hydra.main(version_base=None, config_path=str(config_dir()), config_name="config")
def main(cfg: DictConfig) -> None:
    validate_config(cfg)
    stages = {
        "index": lambda: index_tracks(cfg),
        "screen": lambda: screen_midi(cfg, report=bool(cfg.prep.report_outliers)),
        "encode": lambda: encode(cfg),
        "notes": lambda: warm_notes(cfg),
        "dataset": lambda: finalize(cfg),
    }
    requested = [str(name) for name in cfg.prep.stages]
    unknown = [name for name in requested if name not in stages]
    if unknown:
        raise ValueError(f"unknown prep.stages entries {unknown}; choose from {sorted(stages)}")
    for name in requested:
        announce(name, stages[name]())


if __name__ == "__main__":
    main()
