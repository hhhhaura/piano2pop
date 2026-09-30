"""`auto_resume` must not trust that a checkpoint file is readable.

A `full` checkpoint is 27 GB, so writing one takes minutes — and a wall clock or a `scancel`
during that window leaves a truncated zip. This is the ordinary outcome of stopping a job at the
wrong moment, not an exotic failure, and it blocked the resume of a run 96,000 steps in:

    RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
"""

from __future__ import annotations

import pytest
import torch

from p2pa.train import newest_valid_checkpoint


def write(path, payload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload or {"state_dict": {}}, path)
    return path


def truncate(path):
    """Chop the tail, which is where a zip's central directory lives."""
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    return path


def test_a_healthy_last_is_preferred(tmp_path):
    write(tmp_path / "last.ckpt")
    write(tmp_path / "5000.ckpt")
    assert newest_valid_checkpoint(tmp_path).name == "last.ckpt"


def test_a_truncated_last_falls_back_to_the_newest_numbered(tmp_path):
    truncate(write(tmp_path / "last.ckpt"))
    write(tmp_path / "90000.ckpt")
    write(tmp_path / "96000.ckpt")
    # Newest by *step*, not mtime: a top-k save can rewrite an older file's timestamp.
    assert newest_valid_checkpoint(tmp_path).name == "96000.ckpt"


def test_it_keeps_falling_back_through_damaged_files(tmp_path):
    truncate(write(tmp_path / "last.ckpt"))
    truncate(write(tmp_path / "96000.ckpt"))
    write(tmp_path / "90000.ckpt")
    assert newest_valid_checkpoint(tmp_path).name == "90000.ckpt"


def test_all_damaged_is_refused_rather_than_silently_starting_over(tmp_path):
    truncate(write(tmp_path / "last.ckpt"))
    truncate(write(tmp_path / "90000.ckpt"))
    with pytest.raises(SystemExit, match="unreadable"):
        newest_valid_checkpoint(tmp_path)


def test_a_fresh_run_has_nothing_to_resume(tmp_path):
    assert newest_valid_checkpoint(tmp_path) is None
    assert newest_valid_checkpoint(tmp_path / "does-not-exist") is None
