from p2pa.corpus_process import (
    STEMSETS,
    VARIANT_TARGETS,
    _audio_dir,
    _banned_tokens,
    _chunk_windows,
    _midi_path,
    _variant_choices,
    shard_rows,
)


def test_hard_mask_allows_only_gm_zero_and_no_drums() -> None:
    banned = set(_banned_tokens([0]))
    assert 1135 not in banned
    assert all(token in banned for token in range(1136, 1265))
    assert all(token in banned for token in range(1265, 1393))


def test_guitar_variant_mask_allows_only_its_three_representative_programs() -> None:
    banned = set(_banned_tokens([24, 26, 29]))
    assert all(1135 + program not in banned for program in (24, 26, 29))
    assert 1135 in banned
    assert 1135 + 25 in banned
    assert all(token in banned for token in range(1265, 1393))


def test_overlapping_chunks_assign_each_onset_to_one_core() -> None:
    windows = _chunk_windows(61.0)
    assert windows == [
        (0.0, 30.0, 0.0, 27.5),
        (25.0, 30.0, 27.5, 52.5),
        (50.0, 11.0, 52.5, 61.0),
    ]


def test_every_stem_mixture_is_decoded_as_piano() -> None:
    """Function-selective transcription: all four mixtures, piano only, for every track."""
    choices = _variant_choices("track-a")
    assert choices == _variant_choices("track-b")
    assert choices == [(stemset, "piano") for stemset in ("full6", "nobass", "harmonic", "pianobass")]
    assert set(STEMSETS) == {"full6", "nobass", "harmonic", "pianobass"}
    assert set(VARIANT_TARGETS) == {"piano"} and VARIANT_TARGETS["piano"]["programs"] == (0,)


def test_six_modulo_shards_are_disjoint_and_complete() -> None:
    rows = [
        {
            "track_id": f"track-{index:02d}",
            "asset_state": "raw_only",
            "reason": "awaiting_separation_and_transcription",
        }
        for index in range(25)
    ]
    shards = [shard_rows(rows, index, 6) for index in range(6)]
    identifiers = [row["track_id"] for shard in shards for row in shard]
    assert len(identifiers) == len(set(identifiers)) == len(rows)
    assert set(identifiers) == {row["track_id"] for row in rows}


def test_twelve_modulo_shards_are_disjoint_and_complete() -> None:
    rows = [
        {
            "track_id": f"track-{index:02d}",
            "asset_state": "raw_only",
            "reason": "awaiting_separation_and_transcription",
        }
        for index in range(50)
    ]
    shards = [shard_rows(rows, index, 12) for index in range(12)]
    identifiers = [row["track_id"] for shard in shards for row in shard]
    assert len(identifiers) == len(set(identifiers)) == len(rows)
    assert set(identifiers) == {row["track_id"] for row in rows}


def test_ineligible_inventory_rows_never_enter_processing() -> None:
    rows = [
        {"track_id": "ready", "asset_state": "ready", "reason": ""},
        {"track_id": "duplicate", "asset_state": "duplicate", "reason": "duplicate_composition"},
        {"track_id": "raw", "asset_state": "raw_only",
         "reason": "awaiting_separation_and_transcription"},
    ]
    assert [row["track_id"] for row in shard_rows(rows, 0, 1)] == ["raw"]


def _publish_track(root, row, *, complete: bool) -> None:
    """Write one track's separated stems and, optionally, its whole deterministic MIDI set."""
    from p2pa.corpus_process import SEPARATION_OUTPUTS
    from p2pa.pianoroll import Note, write_midi

    audio = _audio_dir(row, root)
    audio.mkdir(parents=True, exist_ok=True)
    for name in SEPARATION_OUTPUTS:
        (audio / name).write_bytes(b"\x00" * 1024)
    midi_dir = _midi_path(row, root).parent
    midi_dir.mkdir(parents=True, exist_ok=True)
    notes = [Note(pitch=60, start=0.0, end=0.5, velocity=80, is_drum=False, program=0)]
    targets = [_midi_path(row, root)] + [
        midi_dir / f"{stemset}-{decode_target}.mid"
        for stemset, decode_target in _variant_choices(row["track_id"])
    ]
    for target in targets if complete else targets[:-1]:
        write_midi(notes, target, program=0)






