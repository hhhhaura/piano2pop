"""Lay out downloaded test-set audio exactly as the paper's evaluation read it.

    python sing2piano/trim.py --raw /path/to/downloads

For every pair in `audio_manifest.json`, place each listed YouTube upload (the piano cover, the
song, and where listed the official instrumental) at `<raw>/<video_id>.<ext>`. This cuts from each
file the seconds the paper's download cut — the channel's ident and the leading silence, so the
three files of a pair start at the same musical moment — and writes

    sing2piano/sing2piano_audio/<piano_id>/{piano,song,instrumental}.m4a
    sing2piano/sing2piano_audio/manifest.json

which `process.py` reads. An m4a input is cut by stream copy, as the original was; anything else is
re-encoded to AAC. A different download of the same upload can differ by a codec frame, so allow a
few tens of milliseconds of difference from the paper's timeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "sing2piano_audio"
EXTENSIONS = (".m4a", ".webm", ".opus", ".mp3", ".wav", ".flac", ".ogg", ".aac")


def find(raw: Path, video_id: str) -> Path | None:
    for extension in EXTENSIONS:
        path = raw / f"{video_id}{extension}"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def cut(source: Path, target: Path, seconds: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.m4a")
    codec = ["-c", "copy"] if source.suffix == ".m4a" else ["-vn", "-c:a", "aac", "-b:a", "192k"]
    head = ["-ss", f"{seconds:.6f}"] if seconds > 0 else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", *head, "-i", str(source), *codec, str(temporary)],
                   check=True)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="directory of <video_id>.<ext> files")
    args = parser.parse_args()
    manifest = json.loads((HERE / "audio_manifest.json").read_text())
    results, missing = [], []
    for pair in manifest["results"]:
        files = {}
        for name, entry in pair["files"].items():
            if entry.get("status") != "downloaded":
                continue
            target = AUDIO / pair["piano_id"] / name
            if not target.is_file():
                source = find(args.raw, entry["video_id"])
                if source is None:
                    missing.append(entry["video_id"])
                    continue
                cut(source, target, float(entry.get("trimmed", 0.0)))
            files[name] = {"video_id": entry["video_id"], "status": "downloaded",
                           "trimmed": entry.get("trimmed", 0.0), "duration": probe_duration(target)}
        results.append({"piano_id": pair["piano_id"], "files": files})
    AUDIO.mkdir(parents=True, exist_ok=True)
    (AUDIO / "manifest.json").write_text(json.dumps({"results": results}, indent=1))
    complete = sum(1 for r in results if {"piano.m4a", "song.m4a"} <= set(r["files"]))
    print(json.dumps({"pairs": len(results), "complete_pairs": complete, "missing_files": len(missing),
                      "first_missing": missing[:5]}))
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")


if __name__ == "__main__":
    main()
