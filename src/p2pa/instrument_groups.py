"""GM program allowlists for piano-cover MIDI.

``prep.instruments`` only soft-conditions muscriptor. The hard constraint is ``prep.programs`` —
default ``[0]`` (Acoustic Grand). The full muscriptor ``acoustic_piano`` group is
``{0, 1, 3, 6, 7}``; we deliberately keep only program 0 so decode and the training roll are a
single piano program, not five.
"""
from __future__ import annotations

from collections.abc import Sequence


def normalize_programs(programs: Sequence[int]) -> frozenset[int]:
    allowed = frozenset(int(program) for program in programs)
    if not allowed:
        raise ValueError("prep.programs must list at least one GM program")
    bad = sorted(program for program in allowed if not 0 <= program <= 127)
    if bad:
        raise ValueError(f"GM programs must be in [0, 127], got {bad}")
    return allowed
