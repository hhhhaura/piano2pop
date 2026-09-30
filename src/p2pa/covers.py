"""Portable access to real pianist covers and deterministic Kong listening windows."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from .audio import probe_duration
from .config import project_root, resolve_path
from .paths import notes_dir
from .pianoroll import Note, cached_notes, usable_notes

_CURATED_IDS = project_root() / "corpus" / "kong_cover_ids.txt"


def curated_cover_ids() -> list[str]:
    """The exact 196-cover held-out panel shipped by the eval pack."""
    return [line.strip() for line in _CURATED_IDS.read_text().splitlines() if line.strip()]


def covers_root(cfg) -> Path:
    """The cover recordings, under the data root like everything else."""
    return resolve_path(str(cfg.data.covers_root))


def cover_audio_path(root: Path, track_id: str) -> Path:
    return root / "raw" / track_id / "piano.m4a"


def cover_midi_path(root: Path, track_id: str) -> Path:
    """Kong MIDI in either the packed or the original p2p directory layout."""
    shard = hashlib.sha256(track_id.encode()).hexdigest()[:2]
    packed = root / "cover_midi" / shard / f"{track_id}.mid"
    legacy = root / "mismatch" / "cover_midi" / shard / f"{track_id}.mid"
    return packed if packed.is_file() else legacy


def available_covers(cfg, count: int, requested_ids: list[str] | None = None) -> list[dict]:
    """A stable, well-spread subset whose audio and Kong MIDI are both present."""
    if count <= 0:
        return []
    root = covers_root(cfg)
    raw = root / "raw"
    if requested_ids:
        ids = list(dict.fromkeys(str(value) for value in requested_ids))
    else:
        ids = curated_cover_ids() if raw.is_dir() else []
        # Hash order spreads a small listening panel across the collection while remaining fixed.
        ids.sort(key=lambda value: hashlib.sha256(f"p2pa-listening:{value}".encode()).digest())
    rows = []
    for track_id in ids:
        audio = cover_audio_path(root, track_id)
        midi = cover_midi_path(root, track_id)
        if audio.is_file() and midi.is_file():
            rows.append({"sample_id": track_id, "audio": audio, "midi": midi})
        if len(rows) == count:
            break
    return rows


def bpm_from_beat_cache(path: Path) -> float | None:
    """Robust global BPM from BeatThis beat times, or None for a missing/invalid cache."""
    try:
        beats = [float(value) for value in json.loads(path.read_text()).get("beats", [])]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    if not intervals:
        return None
    return 60.0 / statistics.median(intervals)


def tempo_balanced_corpus_rows(cfg, rows: list[dict], count: int) -> list[dict]:
    """Select tempo quantiles; at count two this is exactly one slow and one fast song."""
    from .paths import beat_path

    measured = []
    for row in rows:
        bpm = bpm_from_beat_cache(beat_path(row, cfg))
        if bpm is not None:
            measured.append((bpm, str(row["sample_id"]), row))
    measured.sort(key=lambda value: (value[0], value[1]))
    count = min(max(0, int(count)), len(measured))
    if count == 0:
        return []
    if count == 1:
        indices = [len(measured) // 2]
    else:
        indices = [round(index * (len(measured) - 1) / (count - 1)) for index in range(count)]
    selected = []
    for position, index in enumerate(indices):
        bpm, _, row = measured[index]
        tempo_class = "slow" if position == 0 else "fast" if position == count - 1 else "medium"
        selected.append({"row": row, "bpm": round(bpm, 1), "tempo_class": tempo_class})
    return selected


def listening_window_seed(track_id: str) -> int:
    raw = hashlib.sha256(f"42:{track_id}:listening".encode()).digest()[:8]
    return int.from_bytes(raw, "little") % (2**31 - 1)


def choose_listening_start(
    track_id: str,
    notes: list[Note],
    duration: float,
    seconds: float,
    min_notes: int,
) -> float:
    """Choose one deterministic dense window, falling back to the densest candidate."""
    span = max(0.0, float(duration) - float(seconds))
    if span <= 0:
        return 0.0
    onsets = sorted(note.start for note in notes)
    best = (-1, 0.0)
    modulus = 2**31 - 1
    for attempt in range(24):
        digest = hashlib.sha256(f"42:{track_id}:{attempt}".encode()).digest()[:8]
        start = (int.from_bytes(digest, "little") % modulus) / modulus * span
        inside = sum(start <= onset < start + seconds for onset in onsets)
        if inside > best[0]:
            best = (inside, start)
        if inside >= min_notes:
            return round(start, 3)
    return round(best[1], 3)


def load_listening_condition(cfg, row: dict, seconds: float) -> tuple[list[Note], float, float]:
    """Filtered Kong notes, cover duration, and the selected window start."""
    notes = cached_notes(row["midi"], str(notes_dir(cfg)))
    notes = usable_notes(
        notes,
        int(cfg.roll.pitch_low),
        int(cfg.roll.pitch_high),
        [int(program) for program in cfg.prep.programs],
    )
    duration = probe_duration(row["audio"], str(cfg.data.ffprobe))
    if duration < seconds:
        raise ValueError(f"cover is {duration:.1f}s, shorter than the {seconds:.1f}s export")
    start = choose_listening_start(
        str(row["sample_id"]), notes, duration, seconds, int(cfg.data.min_notes)
    )
    return notes, duration, start
