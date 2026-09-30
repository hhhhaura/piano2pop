"""Pure helpers for mapping PiCoGen's generated bar grid onto detected song beats."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedNote:
    pitch: int
    start: float
    end: float
    velocity: int


def detected_bar_anchors(beats: list[float], downbeats: list[float]) -> list[float]:
    """Song-time anchors matching the bar intervals used by PiCoGen's decoder."""
    beat_array = np.asarray(beats, dtype=np.float64)
    if beat_array.size < 2 or not np.all(np.diff(beat_array) > 0):
        raise ValueError("beats must contain at least two strictly increasing times")
    if not downbeats:
        raise ValueError("no downbeats")
    indices = [int(np.argmin(np.abs(beat_array - float(value)))) for value in downbeats]
    indices = list(dict.fromkeys(indices))
    if indices[-1] < len(beat_array) - 1:
        indices.append(len(beat_array) - 1)
    anchors = [float(beat_array[index]) for index in indices]
    if len(anchors) < 2 or not np.all(np.diff(anchors) > 0):
        raise ValueError("downbeats do not define increasing bar intervals")
    return anchors


def reconcile_terminal_bar(
    source_bar_ticks: list[int],
    song_bar_seconds: list[float],
    beats: list[float],
    downbeats: list[float],
) -> tuple[list[int], bool]:
    """Remove PiCoGen's extra terminal bar only when its appended beat is a duplicate.

    Upstream ``infer.decode`` appends ``len(beats)-1`` when the final downbeat index is
    ``< len(beats)``. Since a valid index is always smaller than the length, it also appends when
    the final downbeat already maps to the final beat, creating a condition interval with zero
    beats. PiCoGen nevertheless emits a bar for it. That bar has no corresponding song time and is
    intentionally collapsed at the final anchor during alignment.
    """
    if len(source_bar_ticks) == len(song_bar_seconds):
        return source_bar_ticks, False
    beat_array = np.asarray(beats, dtype=np.float64)
    if beat_array.size and downbeats:
        final_downbeat_index = int(
            np.argmin(np.abs(beat_array - float(downbeats[-1])))
        )
        if (
            len(source_bar_ticks) == len(song_bar_seconds) + 1
            and final_downbeat_index == len(beat_array) - 1
        ):
            return source_bar_ticks[:-1], True
    raise ValueError(
        "generated/detected bar mismatch: "
        f"{len(source_bar_ticks) - 1} != {len(song_bar_seconds) - 1}"
    )


def align_notes_to_bars(
    notes: list[tuple[int, int, int, int]],
    source_bar_ticks: list[int],
    song_bar_seconds: list[float],
    *,
    song_duration: float,
    minimum_duration: float = 0.01,
) -> list[AlignedNote]:
    """Warp raw ``(pitch,start_tick,end_tick,velocity)`` notes by corresponding bar anchors."""
    source = np.asarray(source_bar_ticks, dtype=np.float64)
    target = np.asarray(song_bar_seconds, dtype=np.float64)
    if source.size != target.size:
        raise ValueError(
            f"generated/detected bar mismatch: {source.size - 1} != {target.size - 1}"
        )
    if source.size < 2 or not np.all(np.diff(source) > 0):
        raise ValueError("source bar ticks must be strictly increasing")
    if not np.all(np.diff(target) > 0):
        raise ValueError("song bar times must be strictly increasing")
    limit = min(float(song_duration), float(target[-1]))
    aligned = []
    for pitch, start_tick, end_tick, velocity in notes:
        start = float(np.interp(float(start_tick), source, target))
        end = float(np.interp(float(end_tick), source, target))
        start = min(max(start, 0.0), limit)
        end = min(max(end, start + minimum_duration), limit)
        if end > start:
            aligned.append(AlignedNote(int(pitch), start, end, int(velocity)))
    return merge_unisons(aligned)


def merge_unisons(notes: list[AlignedNote]) -> list[AlignedNote]:
    """Merge overlapping same-pitch notes so downstream MIDI parsing cannot swallow note-offs."""
    merged: list[AlignedNote] = []
    for pitch in sorted({note.pitch for note in notes}):
        group = sorted((note for note in notes if note.pitch == pitch), key=lambda n: n.start)
        if not group:
            continue
        current = group[0]
        for note in group[1:]:
            if note.start <= current.end:
                current = AlignedNote(
                    pitch,
                    current.start,
                    max(current.end, note.end),
                    max(current.velocity, note.velocity),
                )
            else:
                merged.append(current)
                current = note
        merged.append(current)
    return sorted(merged, key=lambda note: (note.start, note.pitch))
