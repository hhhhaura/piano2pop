"""Conditioning augmentation: a per-segment source and style draw over a song's transcriptions.

Every `window.segment_bars` bars of a song independently draw

* a **source** — the original transcription (`baseline`), PiCoGen's piano cover, or one of the
  track's MuScriptor variants — and
* a **style** — `normal` leaves the notes as they are; `chord_thin` deletes notes from each onset
  simultaneity at a rate keyed on the note's interval above the local bass (the bass is never
  dropped, chord-tone intervals survive more often than tension/passing ones); `octave_move`
  displaces an inner note by one octave when that keeps it strictly between the bass and the top
  note; `octave_shift` transposes the whole segment by a fixed whole-octave offset —

and the segments are spliced back into one note list. Pure functions throughout, so they are
unit-testable without a corpus. Bar boundaries come from the track's own beat file.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .pianoroll import Note, crop_notes_absolute, usable_notes

CHORD_WINDOW = 0.05
MODES = ("normal", "chord_thin", "octave_move", "octave_shift")

# Semitone classes above the local bass. This is a tertian-stack heuristic, not a real chord/key
# analysis (there is no chord-quality or key input anywhere in this corpus): intervals a
# triad/seventh chord would plausibly contain in any quality (root duplicate, 3rd, 5th, 7th)
# against everything else (2nd/4th/6th-ish tensions and passing tones).
CHORD_TONE_CLASSES = frozenset({0, 3, 4, 7, 8, 10, 11})
TENSION_CLASSES = frozenset({1, 2, 5, 6, 9})


def _simultaneities(ordered: list[Note]) -> list[list[int]]:
    """Cluster indices into ``ordered`` (start-sorted) into onset groups within CHORD_WINDOW."""
    clusters: list[list[int]] = []
    current = [0]
    for index in range(1, len(ordered)):
        if ordered[index].start - ordered[current[0]].start <= CHORD_WINDOW:
            current.append(index)
        else:
            clusters.append(current)
            current = [index]
    clusters.append(current)
    return clusters


def policy_chord_thin(pitches, bass, rng, chord_tone_rate=(0.05, 0.2), tension_rate=(0.2, 0.6)):
    """Keep the bass always; drop other notes at a rate keyed by interval above the bass."""
    chord_probability = rng.uniform(*chord_tone_rate)
    tension_probability = rng.uniform(*tension_rate)
    root = int(pitches[bass])
    keep = []
    for index, pitch in enumerate(pitches):
        if index == bass:
            keep.append(index)
            continue
        interval = (int(pitch) - root) % 12
        probability = tension_probability if interval in TENSION_CLASSES else chord_probability
        if rng.random() >= probability:
            keep.append(index)
    return keep


def policy_octave_move(pitches, bass, rng, rate=(0.1, 0.5)):
    """Per-note octave offsets for notes strictly between the bass and the top note.

    Each eligible note independently draws against one per-simultaneity probability; it only
    moves when landing an octave away keeps it strictly inside (bass, top). Returns an array of
    semitone offsets (0 where nothing moves), parallel to ``pitches``.
    """
    probability = rng.uniform(*rate)
    top = int(np.argmax(pitches))
    bass_pitch, top_pitch = int(pitches[bass]), int(pitches[top])
    offsets = np.zeros(len(pitches), dtype=int)
    for index, pitch in enumerate(pitches):
        if index in (bass, top) or rng.random() >= probability:
            continue
        pitch = int(pitch)
        candidates = [offset for offset in (12, -12) if bass_pitch < pitch + offset < top_pitch]
        if candidates:
            offsets[index] = candidates[int(rng.integers(len(candidates)))]
    return offsets


def reduce_notes(
    notes: list[Note], mode: str, rng, *,
    chord_tone_rate=(0.05, 0.2), tension_rate=(0.2, 0.6),
    octave_move_rate=(0.1, 0.5), octave_shift_choices=(-12, 0, 12),
) -> list[Note]:
    """Apply one of ``MODES`` to a note list."""
    if mode not in MODES:
        raise ValueError(f"unknown augmentation mode {mode!r}; expected one of {MODES}")
    if mode == "normal" or not notes:
        return notes

    if mode == "octave_shift":
        shift = int(octave_shift_choices[int(rng.integers(len(octave_shift_choices)))])
        if shift == 0:
            return notes
        return sorted(
            (
                Note(note.pitch + shift, note.start, note.end, note.velocity, note.is_drum,
                     note.program)
                for note in notes
            ),
            key=lambda note: (note.start, note.pitch),
        )

    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    out: list[Note] = []
    for members in _simultaneities(ordered):
        pitches = np.array([ordered[index].pitch for index in members], dtype=int)
        bass = int(np.argmin(pitches))
        if mode == "chord_thin":
            keep = policy_chord_thin(pitches, bass, rng, chord_tone_rate, tension_rate)
            out.extend(ordered[members[position]] for position in sorted(keep))
        else:  # octave_move
            offsets = policy_octave_move(pitches, bass, rng, octave_move_rate)
            for position, offset in enumerate(offsets):
                note = ordered[members[position]]
                out.append(note if offset == 0 else Note(
                    note.pitch + int(offset), note.start, note.end, note.velocity,
                    note.is_drum, note.program,
                ))
    return sorted(out, key=lambda note: (note.start, note.pitch))


def bars_to_segments(
    downbeats: Sequence[float], bars_per_segment: int, duration: float
) -> list[tuple[float, float]]:
    """Group every `bars_per_segment` consecutive bars into one ``(start, end)`` segment,
    covering the whole ``[0, duration)`` track.

    A "bar" is the interval between two consecutive detected downbeats; any intro before the
    first downbeat becomes part of the first segment (not dropped), and any tail after the last
    downbeat becomes part of the last segment. A final segment with fewer than
    `bars_per_segment` bars remaining is still emitted, shorter rather than absorbed elsewhere.
    """
    if bars_per_segment < 1:
        raise ValueError("bars_per_segment must be >= 1")
    if duration <= 0:
        return []
    interior = [float(value) for value in downbeats if 0.0 < value < duration]
    boundaries = sorted({0.0, *interior, float(duration)})
    return [
        (boundaries[index], boundaries[min(index + bars_per_segment, len(boundaries) - 1)])
        for index in range(0, len(boundaries) - 1, bars_per_segment)
    ]


def _max_simultaneous(notes: Sequence[Note]) -> int:
    """The highest number of notes sounding at the same instant.

    A note ending exactly when another starts is treated as a hand-off, not an overlap - the
    ending note's release is processed before the new note's onset at a tied timestamp, matching
    how `crop_notes`/`piano_roll` already treat a shared boundary elsewhere in this module.
    """
    if not notes:
        return 0
    events = sorted(
        [(note.start, 1) for note in notes] + [(note.end, -1) for note in notes],
        key=lambda event: (event[0], event[1]),
    )
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _weighted_choice(entries: Sequence[tuple[str, float]], rng: np.random.Generator) -> str:
    names = [name for name, _ in entries]
    weights = np.array([max(0.0, weight) for _, weight in entries], dtype=float)
    total = weights.sum()
    if total <= 0:
        return names[0]
    return str(rng.choice(np.array(names, dtype=object), p=weights / total))


def draw_segment_source(
    candidates: dict[str, list[Note]],
    segment: tuple[float, float],
    *,
    baseline_name: str,
    picogen_name: str,
    baseline_probability: float,
    picogen_probability: float,
    min_notes: int,
    pitch_low: int,
    pitch_high: int,
    programs,
    rng: np.random.Generator,
    max_simultaneous_notes: int | None = None,
) -> list[Note]:
    """One segment's notes, absolute-time cropped from a weighted source draw.

    Categories are the baseline, PiCoGen, and the remaining MuScriptor variants split evenly. A
    missing category folds its share back onto the baseline.

    A drawn source too sparse *in this segment specifically*, or (when `max_simultaneous_notes`
    is set) too dense - more notes stacked at once than that, typically a generated variant
    hallucinating a messy chord cluster rather than anything the baseline actually plays - does
    not fall back straight to the baseline: a segment can be genuinely well-covered by a
    different variant even when the dice landed on a bad one, so the remaining candidates are
    tried next, highest-weight first, and the first that is neither too sparse nor too dense in
    this segment is used instead. Only when none of them qualify does it fall back to the
    baseline (unconditionally - the baseline is the "normal case" and is never itself filtered on
    density in either direction: it is what the song actually does there).
    """
    start, end = segment
    others = [name for name in candidates if name not in (baseline_name, picogen_name)]
    entries: list[tuple[str, float]] = [(baseline_name, baseline_probability)]
    if picogen_name in candidates:
        entries.append((picogen_name, picogen_probability))
    else:
        entries[0] = (baseline_name, entries[0][1] + picogen_probability)
    remaining = max(0.0, 1.0 - baseline_probability - picogen_probability)
    if others:
        share = remaining / len(others)
        entries.extend((name, share) for name in others)
    else:
        entries[0] = (entries[0][0], entries[0][1] + remaining)

    def _qualifies(name: str) -> tuple[list[Note], bool]:
        cropped = crop_notes_absolute(candidates[name], start, end)
        usable = usable_notes(cropped, pitch_low, pitch_high, programs)
        ok = len(usable) >= min_notes and (
            max_simultaneous_notes is None or _max_simultaneous(usable) <= max_simultaneous_notes
        )
        return cropped, ok

    chosen = _weighted_choice(entries, rng)
    cropped, ok = _qualifies(chosen)
    if chosen == baseline_name or ok:
        return cropped

    fallback_order = [
        name for name, _ in sorted(entries, key=lambda entry: -entry[1])
        if name not in (chosen, baseline_name)
    ]
    for name in fallback_order:
        cropped, ok = _qualifies(name)
        if ok:
            return cropped

    return crop_notes_absolute(candidates[baseline_name], start, end)


def segment_augment_notes(
    candidates: dict[str, list[Note]],
    downbeats: Sequence[float],
    duration: float,
    *,
    bars_per_segment: int,
    baseline_name: str,
    picogen_name: str,
    baseline_probability: float,
    picogen_probability: float,
    min_notes: int,
    pitch_low: int,
    pitch_high: int,
    programs,
    style_weights: dict[str, float],
    chord_tone_rate=(0.05, 0.2),
    tension_rate=(0.2, 0.6),
    octave_move_rate=(0.1, 0.5),
    octave_shift_choices=(-12, 0, 12),
    max_simultaneous_notes: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[Note]:
    """The full per-segment source + style draw for one song, spliced into one note list.

    `style_weights` is the relative mixture of `normal`/`chord_thin`/`octave_move`/`octave_shift`,
    drawn once per segment. A note overlapping a
    segment boundary is clipped there, same convention `crop_notes`/`crop_notes_absolute` use
    elsewhere; no cross-fade/legato smoothing across the cut.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if baseline_name not in candidates:
        raise ValueError(f"candidates must include the baseline ({baseline_name!r})")
    style_entries = list(style_weights.items())

    out: list[Note] = []
    for segment in bars_to_segments(downbeats, bars_per_segment, duration):
        notes = draw_segment_source(
            candidates, segment,
            baseline_name=baseline_name, picogen_name=picogen_name,
            baseline_probability=baseline_probability, picogen_probability=picogen_probability,
            min_notes=min_notes, pitch_low=pitch_low, pitch_high=pitch_high, programs=programs,
            rng=rng, max_simultaneous_notes=max_simultaneous_notes,
        )
        mode = _weighted_choice(style_entries, rng)
        notes = reduce_notes(
            notes, mode, rng, chord_tone_rate=chord_tone_rate, tension_rate=tension_rate,
            octave_move_rate=octave_move_rate, octave_shift_choices=octave_shift_choices,
        )
        out.extend(notes)
    return sorted(out, key=lambda note: (note.start, note.pitch))
