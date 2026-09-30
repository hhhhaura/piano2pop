from __future__ import annotations

import numpy as np
import torch

from p2pa.pianoroll import Note, write_midi
from p2pa.prepare import (
    latent_reason,
    latent_summary,
    midi_statistics,
    outlier_reasons,
)
from tests.conftest import scale_notes


def statistics_for(cfg, tmp_path, notes, duration: float) -> dict:
    directory = tmp_path / "midi" / "ab" / "track"
    directory.mkdir(parents=True, exist_ok=True)
    write_midi(list(notes), directory / str(cfg.data.source.baseline_midi))
    cfg.data.source.root = str(tmp_path)
    cfg.data.source.midi_subdir = "midi"
    row = {"shard": "ab", "track_id": "track", "duration": duration}
    return midi_statistics(row, cfg)


def test_a_healthy_transcription_is_kept(cfg, tmp_path):
    stats = statistics_for(cfg, tmp_path, scale_notes(200), duration=100.0)
    assert stats["notes"] == 200
    assert outlier_reasons(stats, cfg, None) == []


def test_a_stuck_note_is_caught(cfg, tmp_path):
    """A note-off that never arrived turns one note into a drone across the whole track."""
    notes = [*scale_notes(100), Note(pitch=40, start=0.0, end=200.0, velocity=80, is_drum=False)]
    stats = statistics_for(cfg, tmp_path, notes, duration=200.0)
    assert any(reason.startswith("stuck_note") for reason in outlier_reasons(stats, cfg, None))


def test_a_wall_of_pitches_is_caught(cfg, tmp_path):
    """What a decode that exploded looks like: every pitch sounding at once."""
    notes = [
        Note(pitch=pitch, start=0.0, end=5.0, velocity=80, is_drum=False)
        for pitch in range(21, 109)
    ]
    stats = statistics_for(cfg, tmp_path, notes, duration=60.0)
    assert stats["max_simultaneous"] == 88
    assert any(reason.startswith("polyphony") for reason in outlier_reasons(stats, cfg, None))


def test_a_midi_that_does_not_describe_its_audio_is_caught(cfg, tmp_path):
    stats = statistics_for(cfg, tmp_path, scale_notes(40), duration=300.0)
    assert any(reason.startswith("duration_ratio") for reason in outlier_reasons(stats, cfg, None))


def test_an_empty_transcription_is_caught(cfg, tmp_path):
    stats = statistics_for(cfg, tmp_path, [], duration=100.0)
    assert "no_usable_notes" in outlier_reasons(stats, cfg, None)


def test_the_thresholds_sit_above_this_corpus(cfg, tmp_path):
    """The measured corpus maxima, as a regression guard.

    Every one of these is a real track. If a future edit tightens a cap below them, this fails —
    which is the point: an over-eager filter that rejected 116 healthy tracks is the mistake this
    stage has already made once.
    """
    dense = {"notes": 4000, "notes_per_second": 21.4, "max_simultaneous": 58,
             "max_note_seconds": 22.4, "span": 200.0, "duration_ratio": 0.65}
    assert outlier_reasons(dense, cfg, None) == []


def test_a_blown_up_latent_is_caught(cfg):
    healthy = latent_summary(np.random.randn(64, 500).astype(np.float32))
    assert latent_reason(healthy, cfg) == ""
    blown = np.random.randn(64, 500).astype(np.float32)
    blown[3, 17] = 1e4
    assert latent_reason(latent_summary(blown), cfg).startswith("latent_outlier")
    nan = np.full((64, 8), np.nan, dtype=np.float32)
    assert latent_reason(latent_summary(nan), cfg) == "latent_non_finite"


def test_the_peak_is_what_catches_a_single_bad_frame(cfg):
    """One wild frame in a 6,000-frame track is invisible in the std and fatal to an MSE loss."""
    array = np.random.randn(64, 6000).astype(np.float32)
    array[:, 4242] = 200.0
    summary = latent_summary(array)
    assert summary["latent_std"] < 3.0
    assert latent_reason(summary, cfg) != ""


def test_a_latent_summary_is_json_safe(cfg):
    import json

    summary = latent_summary(torch.randn(64, 32).numpy())
    json.dumps(summary)
