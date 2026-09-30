"""Beats and PiCoGen2 piano covers for every separated corpus track.

    python corpus/picogen/run.py beats   --python $PICOGEN_PYTHON --device 0
    python corpus/picogen/run.py picogen --python $PICOGEN_PYTHON --picogen-root $PICOGEN_ROOT --device 0

Both read `.cache/manifests/p2pdata.inventory.jsonl` and each track's separated instrumental,
`data/p2pdata/audio/<shard>/<track_id>/instrumental.mp3`, under `$P2PA_DATA_ROOT`, and run a worker
inside the PiCoGen environment (`corpus/picogen/setup.sh`), which has BeatThis and PiCoGen2.

`beats`    BeatThis (`final0`, DBN) beats and downbeats -> `data/p2pdata/beats/<shard>/<id>.json`.
           Every setting with segment augmentation reads these for its four-bar segments.
`picogen`  PiCoGen2 piano covers: BeatThis, SheetSage features, bar-by-bar generation, and each
           generated bar stretched onto the detected bar -> `data/p2pdata/midi/<shard>/<id>/
           full-picogen.mid`. Only the `pico` and `final` settings need these. Its BeatThis output
           is identical to `beats` and is published as the track's beat file too.

Resumable: finished tracks are skipped. `--shard i --num-shards n` splits the corpus across GPUs.
Stdlib only, so it runs under any Python; the heavy lifting is in the worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
MINIMUM_NOTES = 50


def owns(track_id: str, shard: int, num_shards: int) -> bool:
    value = int.from_bytes(hashlib.sha256(track_id.encode()).digest()[:8], "big")
    return value % num_shards == shard


def track_seed(track_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{track_id}".encode()).digest()[:4], "big")


def duration(path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (ValueError, KeyError):
        return 0.0


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True))
    os.replace(temporary, path)


def beats_path(corpus: Path, row: dict) -> Path:
    return corpus / "beats" / row["shard"] / f"{row['track_id']}.json"


def picogen_complete(target: Path, metadata: Path, fingerprint: str) -> bool:
    if not target.is_file() or not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text())
    except (OSError, ValueError):
        return False
    return value.get("fingerprint") == fingerprint and int(value.get("notes", 0)) >= MINIMUM_NOTES


def jobs_for(args, rows: list[dict], corpus: Path, cache: Path) -> list[dict]:
    jobs = []
    for row in rows:
        audio = corpus / "audio" / row["shard"] / row["track_id"] / "instrumental.mp3"
        if not audio.is_file():
            continue
        if args.mode == "beats":
            if not beats_path(corpus, row).is_file():
                jobs.append({"track_id": row["track_id"], "shard": row["shard"], "audio": str(audio)})
            continue
        seed = track_seed(row["track_id"], args.seed)
        stat = audio.stat()
        fingerprint = hashlib.sha256(json.dumps({
            "schema": SCHEMA_VERSION, "alignment": "bar-beat-anchors-v1", "track_id": row["track_id"],
            "audio": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}, "seed": seed,
            "temperature": args.temperature, "model": "picogen2:model_ft_00070000",
        }, sort_keys=True).encode()).hexdigest()[:20]
        target_dir = corpus / "midi" / row["shard"] / row["track_id"]
        target, metadata = target_dir / "full-picogen.mid", target_dir / "full-picogen.json"
        if picogen_complete(target, metadata, fingerprint):
            continue
        work = cache / row["shard"] / row["track_id"] / fingerprint
        jobs.append({
            "track_id": row["track_id"], "shard": row["shard"], "audio": str(audio),
            "duration": duration(audio, args.ffprobe), "seed": seed, "temperature": args.temperature,
            "fingerprint": fingerprint, "minimum_notes": MINIMUM_NOTES,
            "beat_cache": str(work / "beats.json"), "feature_cache": str(work / "sheetsage.npz"),
            "raw_midi": str(work / "raw.mid"), "generation_cache": str(work / "generation.json"),
            "target": str(target), "metadata": str(metadata),
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("beats", "picogen"))
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("PICOGEN_PYTHON", "")),
                        help="the PiCoGen environment's interpreter")
    parser.add_argument("--picogen-root", type=Path, default=Path(os.environ.get("PICOGEN_ROOT", "")),
                        help="the PiCoGen2 checkout (picogen mode)")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("P2PA_DATA_ROOT", ".")))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")
    if not args.python.is_file():
        raise SystemExit("--python (or $PICOGEN_PYTHON) must name the PiCoGen environment's interpreter")

    corpus = args.data_root / "data" / "p2pdata"
    cache = args.data_root / ".cache" / "picogen"
    inventory = args.data_root / ".cache" / "manifests" / "p2pdata.inventory.jsonl"
    rows = [json.loads(line) for line in inventory.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if owns(row["track_id"], args.shard, args.num_shards)]
    jobs = jobs_for(args, rows, corpus, cache)
    print(f"[{args.mode}] shard {args.shard}/{args.num_shards}: {len(rows)} tracks, "
          f"{len(jobs)} to process", flush=True)
    if not jobs:
        return

    by_track = {job["track_id"]: job for job in jobs}
    environment = dict(os.environ)
    if args.device.isdigit():
        environment["CUDA_VISIBLE_DEVICES"] = args.device
    with tempfile.TemporaryDirectory(prefix=f"p2pa_{args.mode}_") as temporary:
        jobs_path = Path(temporary) / "jobs.jsonl"
        jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
        if args.mode == "beats":
            command = [str(args.python), str(HERE / "beats_worker.py"), "--jobs", str(jobs_path),
                       "--device", args.device]
        else:
            command = [str(args.python), str(HERE / "picogen_worker.py"), "--jobs", str(jobs_path),
                       "--picogen-root", str(args.picogen_root), "--device", args.device]
        process = subprocess.Popen(command, cwd=str(HERE), env=environment,
                                   stdout=subprocess.PIPE, text=True)
        ok = failed = 0
        assert process.stdout is not None
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print(f"[worker] {line.rstrip()}", flush=True)
                continue
            if message.get("event") != "result":
                continue
            job = by_track[message["track_id"]]
            row = {"track_id": job["track_id"], "shard": job["shard"]}
            if message.get("ok"):
                ok += 1
                if args.mode == "beats":
                    atomic_json(beats_path(corpus, row),
                                {"beats": message["beats"], "downbeats": message["downbeats"]})
                elif not beats_path(corpus, row).is_file() and Path(job["beat_cache"]).is_file():
                    beats_path(corpus, row).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(job["beat_cache"], beats_path(corpus, row))
            else:
                failed += 1
                print(f"[{args.mode}] {job['track_id']} failed: {message.get('reason')}", flush=True)
            if (ok + failed) % 50 == 0:
                print(f"[{args.mode}] {ok + failed}/{len(jobs)} done ({failed} failed)", flush=True)
        code = process.wait()
    print(json.dumps({"mode": args.mode, "ok": ok, "failed": failed}), flush=True)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
