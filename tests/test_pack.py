from __future__ import annotations

import hashlib
import json

from p2pa.pack import eval_members, merge_tree


def test_eval_pack_uses_the_portable_cover_layout(cfg, tmp_path, monkeypatch):
    root = tmp_path / "data-root"
    monkeypatch.setenv("P2PA_DATA_ROOT", str(root))
    monkeypatch.setattr("p2pa.covers.curated_cover_ids", lambda: ["cover-id"])
    row = {
        "sample_id": "corpus-id",
        "track_id": "corpus-id",
        "shard": "00",
        "status": "ok",
        "split": "validation",
    }
    manifest = root / ".cache" / "manifests" / "p2pdata.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(row) + "\n")
    audio = root / "data" / "p2pdata" / "audio" / "00" / "corpus-id" / "instrumental.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")

    track_id = "cover-id"
    recording = root / "covers" / "raw" / track_id / "piano.m4a"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"cover")
    shard = hashlib.sha256(track_id.encode()).hexdigest()[:2]
    midi = root / "covers" / "cover_midi" / shard / f"{track_id}.mid"
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b"midi")
    unpaired = root / "covers" / "raw" / "no-kong" / "piano.m4a"
    unpaired.parent.mkdir(parents=True)
    unpaired.write_bytes(b"must not be packed")

    arcnames = {arcname for _, arcname in eval_members(cfg)}
    assert arcnames == {
        "corpus/audio/00/corpus-id/instrumental.mp3",
        "covers/raw/cover-id/piano.m4a",
        f"covers/cover_midi/{shard}/cover-id.mid",
    }


def test_unpack_tree_merge_is_idempotent_and_does_not_overwrite(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "kept.txt").write_text("old")
    first = tmp_path / "first"
    first.mkdir()
    (first / "kept.txt").write_text("new")
    (first / "added.txt").write_text("added")

    merge_tree(first, target)
    assert (target / "kept.txt").read_text() == "old"
    assert (target / "added.txt").read_text() == "added"

    second = tmp_path / "second"
    second.mkdir()
    (second / "added.txt").write_text("duplicate")
    merge_tree(second, target)
    assert (target / "added.txt").read_text() == "added"
