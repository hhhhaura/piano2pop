"""Build the corpus inventory from the released track list and the audio you downloaded.

    p2pa-corpus-index --tracks corpus/tracks.csv --raw $P2PA_DATA_ROOT/data/raw

`corpus/tracks.csv` lists the paper's 8,876 tracks: `track_id, youtube_id, artist, song, split`.
Place each song's audio (the full mix, as uploaded) at `<raw>/<youtube_id>.<ext>`, with any
extension ffmpeg reads. This writes `.cache/manifests/p2pdata.inventory.jsonl`, the file
`p2pa-corpus-process` and `p2pa-prep` read: one row per track that has its audio, carrying the
paper's split so `p2pa-prep` reproduces it exactly. Tracks without audio are reported, not fatal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .config import resolve_path

AUDIO_EXTENSIONS = (".m4a", ".mp3", ".webm", ".opus", ".wav", ".flac", ".ogg", ".aac")


def shard_of(track_id: str) -> str:
    """The corpus's two-hex-digit shard directory for a track."""
    return hashlib.sha256(track_id.encode()).hexdigest()[:2]


def find_audio(raw: Path, youtube_id: str) -> Path | None:
    for extension in AUDIO_EXTENSIONS:
        path = raw / f"{youtube_id}{extension}"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def build(tracks: Path, raw: Path) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    with tracks.open(newline="") as handle:
        for track in csv.DictReader(handle):
            audio = find_audio(raw, track["youtube_id"])
            if audio is None:
                missing.append(track["youtube_id"])
                continue
            rows.append({
                "track_id": track["track_id"],
                "shard": shard_of(track["track_id"]),
                "pop_id": track["youtube_id"],
                "artist": track["artist"],
                "song": track["song"],
                "name": track["track_id"],
                "split": track["split"],
                # What `p2pa-corpus-process separate` selects; `p2pa-prep` later keeps the rows
                # whose instrumental and piano transcription both exist.
                "asset_state": "raw_only",
                "reason": "awaiting_separation_and_transcription",
                "artifacts": {"raw_audio": str(audio)},
            })
    rows.sort(key=lambda row: row["track_id"])
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", type=Path, default=Path(__file__).resolve().parents[2] / "corpus" / "tracks.csv")
    parser.add_argument("--raw", type=Path, default=resolve_path("data/raw"))
    parser.add_argument("--output", type=Path,
                        default=resolve_path(".cache/manifests/p2pdata.inventory.jsonl"))
    args = parser.parse_args()
    rows, missing = build(args.tracks, args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(args.output)
    print(json.dumps({"inventory": str(args.output), "tracks_with_audio": len(rows),
                      "missing_audio": len(missing), "first_missing": missing[:5]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
