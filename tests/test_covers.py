from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import torch

from p2pa.callbacks import SampleCallback
from p2pa.covers import (
    available_covers,
    choose_listening_start,
    cover_midi_path,
    curated_cover_ids,
    tempo_balanced_corpus_rows,
)
from p2pa.pianoroll import Note


def _touch_cover(root, track_id: str, *, packed: bool = True) -> None:
    audio = root / "raw" / track_id / "piano.m4a"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio")
    shard = hashlib.sha256(track_id.encode()).hexdigest()[:2]
    middle = ("cover_midi",) if packed else ("mismatch", "cover_midi")
    midi = root.joinpath(*middle, shard, f"{track_id}.mid")
    midi.parent.mkdir(parents=True, exist_ok=True)
    midi.write_bytes(b"midi")


def test_cover_discovery_supports_packed_layout_and_is_deterministic(cfg, tmp_path, monkeypatch):
    root = tmp_path / "portable"
    cfg.data.covers_root = "covers"
    monkeypatch.setenv("P2PA_DATA_ROOT", str(root))
    covers = root / "covers"
    for track_id in ("cover-c", "cover-a", "cover-b"):
        _touch_cover(covers, track_id)

    requested = ["cover-a", "cover-b", "cover-c"]
    first = available_covers(cfg, 2, requested)
    second = available_covers(cfg, 2, requested)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len(first) == 2
    assert all(cover_midi_path(covers, row["sample_id"]).is_file() for row in first)


def test_curated_cover_panel_is_fixed_and_contains_the_listening_pair(cfg):
    ids = curated_cover_ids()
    assert len(ids) == len(set(ids)) == 196
    assert set(cfg.trainer.kong_sample_ids) <= set(ids)


def test_dense_listening_window_is_stable():
    notes = [
        Note(60, 50.0 + index * 0.1, 50.05 + index * 0.1, 80, False, 0)
        for index in range(40)
    ]
    start = choose_listening_start("cover", notes, duration=100.0, seconds=30.0, min_notes=24)
    assert start == choose_listening_start(
        "cover", notes, duration=100.0, seconds=30.0, min_notes=24
    )
    assert sum(start <= note.start < start + 30.0 for note in notes) >= 24


def test_corpus_listening_pair_is_slowest_and_fastest(cfg, tmp_path):
    cfg.data.source.root = str(tmp_path / "corpus")
    rows = []
    for track_id, interval in (("slow", 1.0), ("middle", 0.75), ("fast", 0.5)):
        row = {"sample_id": track_id, "track_id": track_id, "shard": "00"}
        path = tmp_path / "corpus" / "beats" / "00" / f"{track_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"beats": [index * interval for index in range(20)]}))
        rows.append(row)

    selected = tempo_balanced_corpus_rows(cfg, rows, 2)
    assert [(item["tempo_class"], item["row"]["sample_id"], item["bpm"]) for item in selected] == [
        ("slow", "slow", 60.0),
        ("fast", "fast", 120.0),
    ]


def test_sample_callback_adds_kong_without_touching_validation_dataset(
    cfg, tmp_path, monkeypatch
):
    cfg.trainer.sample_items = 1
    cfg.trainer.kong_sample_items = 1
    cfg.trainer.sample_every_n_steps = 10
    row = {"sample_id": "corpus-id"}

    class Dataset:
        rows = [row]

        def sample_rows(self, count, ids):
            assert count == 1 and ids is None
            return [row]

    trainer = SimpleNamespace(
        sanity_checking=False,
        global_rank=0,
        global_step=10,
        datamodule=SimpleNamespace(val_ds=Dataset()),
    )
    module = torch.nn.Linear(1, 1).train()
    callback = SampleCallback(cfg, tmp_path)
    monkeypatch.setattr(
        "p2pa.covers.available_covers",
        lambda *_args, **_kwargs: [{"sample_id": "kong-id"}],
    )
    monkeypatch.setattr(
        "p2pa.covers.tempo_balanced_corpus_rows",
        lambda *_args, **_kwargs: [{"row": row, "tempo_class": "slow", "bpm": 60.0}],
    )
    monkeypatch.setattr(
        callback,
        "_export",
        lambda *_args: {"sample_id": "corpus-id"},
    )

    def export_kong(model, *_args):
        model.eval()
        return {"sample_id": "kong-id", "source": "kong"}

    monkeypatch.setattr(callback, "_export_kong", export_kong)
    callback.on_validation_epoch_end(trainer, module)

    index = json.loads((tmp_path / "val_samples" / "step_0000010" / "index.json").read_text())
    assert [(record["source"], record["sample_id"]) for record in index] == [
        ("corpus", "corpus-id"),
        ("kong", "kong-id"),
    ]
    assert module.training
