"""MIDI to a single frame-aligned piano roll.

Prep hard-masks muscriptor decoding to ``prep.programs`` (default GM 0 = Acoustic Grand) and
bans drums. The roll keeps only those programs, so the training encoder sees one piano program
on one plane — not the full acoustic_piano group {0,1,3,6,7}.
"""
from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
import symusic

DRUM_CHANNEL = 9

# Note times arrive from a tick-to-second conversion, so a note meant to land exactly on a frame
# boundary shows up as 29.999999 or 30.000000004. Without this tolerance the two round in opposite
# directions and a note silently gains or loses a frame.
_FRAME_EPSILON = 1e-6


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    end: float
    velocity: int
    is_drum: bool
    program: int = 0


def parse_notes(path: str | Path) -> list[Note]:
    """Read note events with absolute times in seconds, honouring tempo and program changes.

    symusic (C++, nanobind), not mido: parsing was measured as 92% of `roll_from_midi`'s cost, and
    symusic is the standard fast alternative for exactly this - batch note extraction - across
    symbolic-music ML pipelines. `Score.to("second")` does the same tick-to-second conversion mido
    did per-message, and it splits a file by channel the same way mido's per-channel `active` dict
    implicitly did, so each resulting track carries one channel's `program` and `is_drum` (channel
    9) consistently with the previous implementation.

    Verified against mido on 200 real corpus files (baseline, muscriptor variants, picogen):
    183 exact matches, 0 field mismatches, 0 errors. The other 17 differ by a small note count
    (piano performance files only) - traced to mido's `active[key] = (...)` silently overwriting a
    still-sounding note on a same-pitch retrigger, dropping the earlier note it never emits.
    symusic closes the first note and starts a second, per the MIDI spec; this recovers those
    previously-silent-dropped notes rather than losing any.
    """
    score = symusic.Score(str(path)).to("second")
    notes = [
        Note(note.pitch, note.time, note.end, note.velocity, track.is_drum, track.program)
        for track in score.tracks
        for note in track.notes
    ]
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def cached_notes(path: str | Path, cache_dir: str | Path | None = None) -> list[Note]:
    """`parse_notes`, memoised on disk. Identical output, ~40x faster on a repeat read.

    Parsing dominates the training dataloader: measured on this corpus it is 92% of the work
    `roll_from_midi` does, and because a full-song MIDI is cropped per window, the *same* file is
    re-parsed roughly 23 times an epoch — 58 CPU-minutes per epoch that produce a byte-identical
    result every time.

    Only the parse is cached, never the roll. The roll depends on pitch range, register splits and
    the augmentation draw, all of which change between runs or between epochs; the note list
    depends on nothing but the file. The key includes size and mtime, so editing a MIDI invalidates
    it rather than silently serving stale notes.
    """
    path = Path(path)
    if cache_dir is None:
        return parse_notes(path)
    stat = path.stat()
    key = f"{path.stem}-{stat.st_size}-{int(stat.st_mtime)}.npz"
    target = Path(cache_dir) / path.parent.name / key
    if target.is_file():
        try:
            stored = np.load(target)
            return [Note(int(p), float(s), float(e), int(v), bool(d), int(g))
                    for p, s, e, v, d, g in zip(
                        stored["pitch"], stored["start"], stored["end"],
                        stored["velocity"], stored["is_drum"], stored["program"], strict=True)]
        except Exception:  # noqa: BLE001 - a truncated cache must not be fatal
            target.unlink(missing_ok=True)

    notes = parse_notes(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                pitch=np.array([n.pitch for n in notes], dtype=np.int16),
                # float64, not float32. Times run to ~300 s where float32 resolves only to
                # ~3e-5 s — larger than the _FRAME_EPSILON guard above — so a note landing exactly
                # on a frame boundary could round into the neighbouring frame and the cache would
                # not reproduce `parse_notes` exactly.
                start=np.array([n.start for n in notes], dtype=np.float64),
                end=np.array([n.end for n in notes], dtype=np.float64),
                velocity=np.array([n.velocity for n in notes], dtype=np.int16),
                is_drum=np.array([n.is_drum for n in notes], dtype=bool),
                program=np.array([n.program for n in notes], dtype=np.int16),
            )
        temporary.replace(target)
    except OSError:
        temporary.unlink(missing_ok=True)   # a full disk must not stop training
    return notes


def usable_notes(
    notes: list[Note],
    pitch_low: int,
    pitch_high: int,
    programs: Collection[int] | None = None,
) -> list[Note]:
    """Notes that survive into the single training roll.

    Drops drums, out-of-range pitches, and (when ``programs`` is set) any GM program outside the
    allowlist. Default prep uses ``programs={0}`` only — a single Acoustic Grand plane.
    """
    allowed = None if programs is None else frozenset(programs)
    kept: list[Note] = []
    for note in notes:
        if note.is_drum or not (pitch_low <= note.pitch <= pitch_high):
            continue
        if allowed is not None and note.program not in allowed:
            continue
        kept.append(note)
    return kept


def roll_channels(pitch_low: int, pitch_high: int) -> int:
    return 2 * (pitch_high - pitch_low + 1)


def piano_roll(
    notes: list[Note],
    frames: int,
    frames_per_second: float,
    pitch_low: int,
    pitch_high: int,
    *,
    velocity_in_onset: bool = True,
    offset: float = 0.0,
    programs: Collection[int] | None = None,
) -> np.ndarray:
    """Build a single [2 * pitches, frames] roll: sustain planes then onset planes.

    Sustain is binary presence and onset carries velocity, which keeps the two planes
    complementary — repeated notes at the same pitch stay distinguishable from one held note.
    All allowed programs / channels collapse into this one tensor; there is no per-track stack.
    """
    pitches = pitch_high - pitch_low + 1
    roll = np.zeros((2 * pitches, frames), dtype=np.float32)
    for note in usable_notes(notes, pitch_low, pitch_high, programs):
        index = note.pitch - pitch_low
        start = math.floor((note.start - offset) * frames_per_second + _FRAME_EPSILON)
        end = math.ceil((note.end - offset) * frames_per_second - _FRAME_EPSILON)
        end = max(end, start + 1)
        lo, hi = max(0, start), min(frames, end)
        if hi <= lo:
            continue
        roll[index, lo:hi] = 1.0
        if 0 <= start < frames:
            roll[pitches + index, start] = note.velocity / 127.0 if velocity_in_onset else 1.0
    return roll


def _notes_cache(cfg):
    """The parsed-MIDI cache for this config, as an absolute path.

    `cfg.data.notes_dir` is the relative string `.cache/notes`. Read raw it resolves against the
    process working directory, so the cache follows the caller around: warmed in one place, missed
    in another, and on a cluster it collided with an unrelated `.cache` and killed the dataloader.
    Imported lazily because `paths` reaches back into this module's siblings.
    """
    from .paths import notes_dir

    directory = getattr(cfg.data, "notes_dir", None)
    return str(notes_dir(cfg)) if directory else None


def roll_from_midi(path: str | Path, cfg, frames: int, offset: float = 0.0,
                   transform=None, notes: list[Note] | None = None) -> np.ndarray:
    """Training path: one roll from ``prep.programs`` only (default GM 0).

    ``offset`` is where the window starts inside the MIDI file. It is 0 for a transcribed corpus,
    where each MIDI *is* the crop, and the window's start time for a corpus that supplies one
    full-length MIDI per song — cutting the roll out of it costs nothing and avoids writing a
    second copy of every window to disk.

    ``transform`` is applied to the parsed notes before they are rasterised — the one point where
    the conditioning still exists as notes rather than as pixels, which is what an augmentation
    like the segment augmentation needs. It runs before the program and pitch filters, so a policy
    that moves a note out of range simply loses it, exactly as a transcription would have.

    ``notes``, when given, is used instead of a fresh ``cached_notes`` read. A weighted
    conditioning draw already has to parse the candidate once to check it is dense enough before
    accepting it (`P2PDataset._conditioning_midi`); without this, an accepted candidate was parsed
    again here for an identical result - `cached_notes` avoiding the raw MIDI re-parse still means
    a second disk read of its cache file every time.
    """
    from .instrument_groups import normalize_programs

    programs = normalize_programs([int(program) for program in cfg.prep.programs])
    if notes is None:
        notes = cached_notes(path, _notes_cache(cfg))
    if transform is not None:
        notes = transform(notes)
    return piano_roll(
        notes,
        frames,
        float(cfg.roll.frames_per_second),
        int(cfg.roll.pitch_low),
        int(cfg.roll.pitch_high),
        velocity_in_onset=bool(cfg.roll.velocity_in_onset),
        offset=offset,
        programs=programs,
    )


def write_midi(
    notes: list[Note],
    path: str | Path,
    *,
    program: int = 0,
    ticks_per_beat: int = 480,
    tempo: int = 500000,
) -> None:
    """Write notes to a single-track MIDI at a fixed tempo — the inverse of ``parse_notes``."""
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.Message("program_change", channel=0, program=int(program), time=0))

    events: list[tuple[float, int, int, int]] = []
    for note in notes:
        # note_off sorts before note_on at the same tick, so a repeated pitch is not swallowed by
        # its own predecessor's release.
        events.append((note.start, 1, note.pitch, int(note.velocity)))
        events.append((note.end, 0, note.pitch, 0))
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    previous = 0
    for when, kind, pitch, velocity in events:
        tick = round(mido.second2tick(when, ticks_per_beat, tempo))
        track.append(mido.Message(
            "note_on" if kind else "note_off",
            channel=0, note=int(pitch), velocity=velocity, time=max(0, tick - previous),
        ))
        previous = tick
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))


def crop_notes(notes: list[Note], start: float, seconds: float) -> list[Note]:
    """Notes overlapping ``[start, start + seconds)``, rebased so the window begins at zero."""
    kept = [
        Note(
            note.pitch,
            max(note.start, start) - start,
            min(note.end, start + seconds) - start,
            note.velocity, note.is_drum, note.program,
        )
        for note in notes
        if note.end > start and note.start < start + seconds
    ]
    return sorted(kept, key=lambda note: (note.start, note.pitch))


def crop_notes_absolute(notes: list[Note], start: float, end: float) -> list[Note]:
    """Notes overlapping ``[start, end)``, clipped to those bounds but kept at absolute song
    time — unlike ``crop_notes``, nothing is rebased to zero. For splicing several sources'
    segments into one continuous note list (8-bar segment augmentation), where each segment's
    notes need to land at their own true position, not each restart at time zero."""
    kept = [
        Note(
            note.pitch,
            max(note.start, start),
            min(note.end, end),
            note.velocity, note.is_drum, note.program,
        )
        for note in notes
        if note.end > start and note.start < end
    ]
    return sorted(kept, key=lambda note: (note.start, note.pitch))


def total_roll_channels(cfg) -> int:
    """Channels the conditioning encoder is built for: sustain and onset planes per pitch."""
    return roll_channels(int(cfg.roll.pitch_low), int(cfg.roll.pitch_high))
