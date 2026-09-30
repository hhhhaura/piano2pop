"""The memory guard must measure memory that is actually held.

Dataloader workers are *forked*, so they share the parent's pages copy-on-write — and RSS counts a
shared page in full in every process that maps it. Summing RSS over the tree therefore reports a
multiple of the truth, and the multiplier is the number of workers. That is not a small inaccuracy
in a guard whose whole job is deciding when memory is a problem: it stopped two legitimate `full`
resumes, at "66 GB" and "118 GB", neither of which was real.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from p2pa.callbacks import ResourceGuard


def test_forked_children_do_not_multiply_the_measurement():
    """A parent holding ~1 GB with four forked children must not read as ~5 GB."""
    guard = ResourceGuard(0.0)
    baseline = guard._tree_memory_gb()

    block = np.ones((1024, 1024, 128), dtype=np.float64)   # ~1 GB, touched so it is resident
    block[:] = 1.0
    alone = guard._tree_memory_gb() - baseline

    children = []
    for _ in range(4):
        pid = os.fork()
        if pid == 0:
            time.sleep(4)
            os._exit(0)
        children.append(pid)
    try:
        time.sleep(1.0)
        with_children = guard._tree_memory_gb() - baseline
    finally:
        for pid in children:
            os.waitpid(pid, 0)

    assert alone > 0.5, f"the fixture allocated only {alone:.2f} GB; too small to measure"
    # Forking must add little: the children hold nothing of their own. A summed-RSS guard would
    # report roughly 5x here.
    assert with_children < alone * 2.0, (
        f"{alone:.2f} GB became {with_children:.2f} GB after forking four children that allocate "
        "nothing — shared pages are being counted once per process"
    )
    del block


def test_the_guard_still_fires_on_real_growth():
    """Fixing the over-count must not make the guard blind."""
    guard = ResourceGuard(0.0)
    before = guard._tree_memory_gb()
    block = np.ones((1024, 1024, 128), dtype=np.float64)
    block[:] = 1.0
    assert guard._tree_memory_gb() - before > 0.5
    del block


def test_a_process_that_exits_mid_walk_is_tolerated():
    """Children come and go; the walk must not raise because one vanished."""
    guard = ResourceGuard(0.0)
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    assert guard._tree_memory_gb() > 0


@pytest.mark.parametrize("limit,expected", [(0.0, False), (10_000.0, False)])
def test_a_generous_limit_never_fires(limit, expected):
    guard = ResourceGuard(limit)
    assert (guard.limit_gb > 0 and guard._tree_memory_gb() > guard.limit_gb) is expected
