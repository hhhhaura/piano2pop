"""Cache paths must not depend on the process working directory.

Every path in this project resolves through `resolve_path`, against `$P2PA_DATA_ROOT` or the
project root — except that `cfg.data.notes_dir` was read raw in twelve places, so the parsed-MIDI
cache resolved against the CWD instead. It survived here because the CWD happened to be the
project root; on a cluster it hit an unrelated `.cache`, took the dataloader down, and would
otherwise have quietly re-parsed a corpus that prep had already warmed somewhere else.

These tests exercise the real entry points from a *different* working directory, which is the one
thing that would have caught it.
"""

from __future__ import annotations

import os

import numpy as np

from p2pa.paths import notes_dir
from p2pa.pianoroll import roll_from_midi, write_midi
from tests.conftest import scale_notes


def test_notes_dir_is_absolute_and_follows_the_data_root(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("P2PA_DATA_ROOT", str(tmp_path))
    # The `cfg` fixture points this at an absolute temp dir so tests never touch the real cache;
    # the relative default is what production uses and what this test is about.
    cfg.data.notes_dir = ".cache/notes"
    resolved = notes_dir(cfg)
    assert resolved.is_absolute()
    assert resolved == tmp_path / ".cache" / "notes"

    monkeypatch.chdir(tmp_path.parent)
    assert notes_dir(cfg) == resolved, "the cache moved when the working directory did"


def test_roll_building_writes_its_cache_to_the_data_root(cfg, tmp_path, monkeypatch):
    """`roll_from_midi` is what the dataloader actually calls."""
    root = tmp_path / "root"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("P2PA_DATA_ROOT", str(root))
    cfg.data.notes_dir = ".cache/notes"

    midi = tmp_path / "songs" / "track" / "full-piano.mid"
    midi.parent.mkdir(parents=True)
    write_midi(scale_notes(40), midi)

    # Run from a directory that is neither the project nor the data root.
    monkeypatch.chdir(elsewhere)
    roll = roll_from_midi(midi, cfg, 400, 0.0)

    assert roll.shape[-1] == 400
    assert np.isfinite(roll).all()
    # The cache landed under the data root...
    assert list((root / ".cache" / "notes").rglob("*.npz")), "nothing was cached under the root"
    # ...and nothing was scattered into the working directory.
    assert not (elsewhere / ".cache").exists(), (
        "a cache was written relative to the working directory; "
        f"found {[p.name for p in elsewhere.iterdir()]}"
    )


def test_a_stray_cache_file_in_the_cwd_is_harmless(cfg, tmp_path, monkeypatch):
    """The exact cluster failure: an unrelated `.cache` in the working directory.

    Resolving against the data root means a file of that name in the CWD is simply irrelevant.
    Before the fix this raised `FileExistsError: '.cache'` from inside a dataloader worker.
    """
    root = tmp_path / "root"
    root.mkdir()
    working = tmp_path / "working"
    working.mkdir()
    (working / ".cache").write_text("not a directory")

    monkeypatch.setenv("P2PA_DATA_ROOT", str(root))
    midi = tmp_path / "t.mid"
    write_midi(scale_notes(20), midi)

    monkeypatch.chdir(working)
    roll = roll_from_midi(midi, cfg, 200, 0.0)
    assert roll.shape[-1] == 200
    assert (working / ".cache").is_file(), "the stray file should be untouched"


def test_os_getcwd_is_not_consulted(cfg, tmp_path, monkeypatch):
    """A guard with teeth: fail if anything resolves a cache path via the CWD."""
    monkeypatch.setenv("P2PA_DATA_ROOT", str(tmp_path))
    cfg.data.notes_dir = ".cache/notes"
    calls = []
    real = os.getcwd

    def spy():
        calls.append(1)
        return real()

    monkeypatch.setattr(os, "getcwd", spy)
    notes_dir(cfg)
    assert not calls, "notes_dir consulted the working directory"
