from __future__ import annotations

import pytest

from p2pa.audio import atomic_media_output


def test_atomic_media_output_keeps_suffix_and_publishes(tmp_path):
    target = tmp_path / "song.mp3"
    with atomic_media_output(target) as temporary:
        assert temporary.suffix == ".mp3"
        assert temporary != target
        temporary.write_bytes(b"complete")
    assert target.read_bytes() == b"complete"
    assert not temporary.exists()


def test_atomic_media_output_cleans_an_interrupted_encode(tmp_path):
    target = tmp_path / "song.wav"
    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_media_output(target) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("interrupted")
    assert not target.exists()
    assert not temporary.exists()
