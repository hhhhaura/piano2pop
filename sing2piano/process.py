"""Stage 7 — one vocal-free audio file per row, and its piano transcriptions.

Three stages, all resumable and shardable across GPUs:

    separate   the target audio      ->  sing2piano_audio/processed/<piano_id>.<ext>
    transcribe that audio (MuScriptor) ->  sing2piano_audio/midi/<piano_id>.mid
    piano      the piano cover (Kong)  ->  sing2piano_audio/midi_piano/<piano_id>.mid

**Where the target comes from.** A row that already has an official instrumental aligned at least
as well as its song file uses it directly: it is the label's own stem, not a separation. Every
other row has its song put through `htdemucs --two-stems vocals`. The instrumental rows are copied
rather than re-encoded, and `processed.jsonl` records which source each row used.

**The transcriptions.** `transcribe` is MuScriptor over the instrumental, through
`p2pa.corpus_process.transcribe_piano` — the same chunking, program-0 decode and onset-energy
velocities as the training corpus — and gives the training-distribution inputs of Base, FST and
RDA. `piano` is Kong et al.'s transcriber over the real piano cover, the real-piano inputs.

    uv run python sing2piano/process.py --stage separate --device 0 --shard 0 --num-shards 2
    uv run python sing2piano/process.py --stage transcribe --device 0
    uv run python sing2piano/process.py --stage piano --device 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from common import HERE, read_jsonl, write_jsonl

AUDIO = HERE / "sing2piano_audio"
PROCESSED = AUDIO / "processed"
MIDI = AUDIO / "midi"
PLAN_JSONL = AUDIO / "processed.jsonl"
VERIFIED_CSV = HERE / "test_verified.csv"


def _env_path(name: str, purpose: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"${name} is not set; it names {purpose}. Set it in env.sh (see README.md).")
    return Path(value).expanduser()


# demucs 4.1.0 and muscriptor share one interpreter (scripts/setup_corpus_tools.sh); Kong's
# piano_transcription_inference is in the PiCoGen environment (corpus/picogen/setup.sh).
LEGACY_PYTHON = _env_path("P2PA_MIR_PYTHON", "the interpreter with demucs 4.1.0 and muscriptor")
KONG_PYTHON = _env_path("PICOGEN_PYTHON", "the PiCoGen environment, which has Kong's transcriber")
MIDI_PIANO = AUDIO / "midi_piano"
DEMUCS_MODEL = "htdemucs"
# htdemucs is a Transformer and refuses a segment longer than the 7.8 s it was trained for.
DEMUCS_SEGMENT = 7
MP3_KBPS = 192


@dataclass
class Row:
    piano_id: str
    name: str
    source: str          # "instrumental" or "separated"
    audio_in: Path       # what to read
    audio_out: Path      # what to write under processed/


def plan(rows: list[dict], durations: dict[str, dict]) -> list[Row]:
    """Decide each row's target audio and where it lands."""
    out = []
    for row in rows:
        pid = row["piano_id"]
        got = durations.get(pid, {})
        piano = got.get("piano.m4a", 0.0)
        song = got.get("song.m4a", 0.0)
        instrumental = got.get("instrumental.m4a", 0.0)
        song_drift = abs(piano - song) if song else float("inf")
        inst_drift = abs(piano - instrumental) if instrumental else float("inf")
        directory = AUDIO / pid
        # "Better" is measured, not assumed: an official instrumental only wins when it tracks the
        # arrangement at least as closely as the song does.
        if row.get("instrumental_id") and inst_drift <= song_drift:
            source, audio_in, ext = "instrumental", directory / "instrumental.m4a", ".m4a"
        elif song:
            source, audio_in, ext = "separated", directory / "song.m4a", ".mp3"
        else:
            continue
        if audio_in.is_file():
            out.append(Row(pid, row.get("name", pid), source, audio_in,
                           PROCESSED / f"{pid}{ext}"))
    return out


def separate(row: Row, device: str) -> dict:
    """demucs `--two-stems vocals`, keeping the accompaniment. Forked from p2p's separate.py.

    demucs wants a torch device string, so a bare index is rejected outright — `-d` gets `cuda`
    and the index is passed the way p2p passes it, through `CUDA_VISIBLE_DEVICES`. Setting only
    one of the two silently strands the job: the wrong card, or the CPU.
    """
    started = time.time()
    environment = dict(os.environ)
    if device.strip().isdigit():
        environment["CUDA_VISIBLE_DEVICES"] = device.strip()
        torch_device = "cuda"
    else:
        torch_device = device or "cuda"
    with tempfile.TemporaryDirectory(prefix="s2p_demucs_") as workspace:
        out = Path(workspace)
        result = subprocess.run(
            [str(LEGACY_PYTHON), "-m", "demucs.separate", "-n", DEMUCS_MODEL,
             "--two-stems", "vocals", "--mp3", "--mp3-bitrate", str(MP3_KBPS),
             "--segment", str(DEMUCS_SEGMENT), "-d", torch_device, "-o", str(out),
             str(row.audio_in)],
            capture_output=True, text=True, check=False, env=environment)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            return {"ok": False, "reason": f"demucs_failed: {tail[-1] if tail else '?'}"[:200]}
        produced = out / DEMUCS_MODEL / row.audio_in.stem / "no_vocals.mp3"
        if not produced.is_file():
            return {"ok": False, "reason": "demucs produced no no_vocals.mp3"}
        row.audio_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = row.audio_out.with_suffix(row.audio_out.suffix + ".tmp")
        shutil.move(str(produced), temporary)
        temporary.replace(row.audio_out)
    return {"ok": True, "reason": "separated", "seconds": round(time.time() - started, 1)}


def adopt(row: Row) -> dict:
    """Copy an official instrumental through unchanged — re-encoding it would only lose bits."""
    row.audio_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = row.audio_out.with_suffix(row.audio_out.suffix + ".tmp")
    shutil.copyfile(row.audio_in, temporary)
    temporary.replace(row.audio_out)
    return {"ok": True, "reason": "adopted"}


def transcribe(rows: list[Row], device: str) -> dict[str, dict]:
    """MuScriptor over each row's instrumental, exactly as the training corpus's baseline is made.

    These are the training-distribution inputs of Base, FST and RDA: `p2pa.corpus_process`'s own
    chunking, program-0 decode and onset-energy velocities, run on the test instrumentals.
    """
    sys.path.insert(0, str(HERE.parent / "src"))
    from p2pa.corpus_process import transcribe_piano

    environment_device = device.strip()
    if environment_device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = environment_device
    items = [(row.piano_id, row.audio_out, MIDI / f"{row.piano_id}.mid")
             for row in rows if row.audio_out.is_file()]
    return transcribe_piano(items, mir_python=LEGACY_PYTHON, cache_root=AUDIO / "cache")


def transcribe_piano(rows: list[Row], device: str) -> None:
    """Kong over each row's piano recording — the conditioning input, not the target.

    `process.py`'s other stages transcribe the *instrumental*, which is p2pdata's convention: its
    conditioning MIDI is muscriptor over a demucs stem. This set exists because its piano side is
    a real arrangement rather than a separation, so the conditioning has to come from that
    recording, and Kong is what reads a solo piano — it regresses velocity and pedal directly
    instead of muscriptor's constant 100.
    """
    MIDI_PIANO.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in rows:
        target = MIDI_PIANO / f"{row.piano_id}.mid"
        if target.is_file() and target.stat().st_size > 0:
            continue
        audio = AUDIO / row.piano_id / "piano.m4a"
        if audio.is_file():
            jobs.append({"piano_id": row.piano_id, "audio": str(audio), "midi": str(target)})
    if not jobs:
        print("every piano recording is already transcribed", file=sys.stderr)
        return

    print(f"{len(jobs)} piano recordings to transcribe", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="s2p_kong_") as workspace:
        jobs_file = Path(workspace) / "jobs.jsonl"
        jobs_file.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        results = MIDI_PIANO / "kong.jsonl"
        environment = dict(os.environ)
        if device.strip().isdigit():
            environment["CUDA_VISIBLE_DEVICES"] = device.strip()
        # Kong's library prints a line per segment — tens of thousands over a full pass. Sent
        # to a file rather than inherited: a long background job that floods its pipe is a job
        # that gets reaped, and the useful progress is in `kong.jsonl` anyway.
        log_path = MIDI_PIANO / "kong.log"
        with log_path.open("a") as log:
            subprocess.run(
                [str(KONG_PYTHON), str(Path(__file__).parent / "kong_worker.py"),
                 "--jobs", str(jobs_file), "--out", str(results), "--device", "cuda"],
                check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)


def shard_of(piano_id: str, num_shards: int) -> int:
    return int(hashlib.sha256(piano_id.encode()).hexdigest()[:8], 16) % num_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("separate", "transcribe", "piano", "all"),
                        default="all")
    parser.add_argument("--device", default="0", help="GPU index for demucs and muscriptor")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not VERIFIED_CSV.is_file():
        raise SystemExit(f"{VERIFIED_CSV} missing")
    manifest = json.loads((AUDIO / "manifest.json").read_text())
    durations = {r["piano_id"]: {k: float(v.get("duration") or 0.0)
                                 for k, v in r["files"].items()
                                 if v.get("status") in {"downloaded", "cached"}}
                 for r in manifest["results"]}
    with VERIFIED_CSV.open() as handle:
        rows = plan(list(csv.DictReader(handle)), durations)
    if args.num_shards > 1:
        rows = [r for r in rows if shard_of(r.piano_id, args.num_shards) == args.shard]
    if args.limit:
        rows = rows[: args.limit]

    if args.stage == "piano":
        transcribe_piano(rows, args.device)
        return

    log = {entry["piano_id"]: entry for entry in read_jsonl(PLAN_JSONL)}
    print(f"{len(rows)} rows in this shard "
          f"({sum(1 for r in rows if r.source == 'instrumental')} adopted, "
          f"{sum(1 for r in rows if r.source == 'separated')} to separate)", file=sys.stderr)

    for index, row in enumerate(rows, 1):
        entry = log.get(row.piano_id, {"piano_id": row.piano_id, "name": row.name,
                                       "source": row.source})
        if args.stage in ("separate", "all") and not row.audio_out.is_file():
            outcome = adopt(row) if row.source == "instrumental" else separate(row, args.device)
            entry.update({"audio": str(row.audio_out.relative_to(AUDIO)), **outcome})
            log[row.piano_id] = entry
            write_jsonl(PLAN_JSONL, list(log.values()))
            if not outcome["ok"]:
                print(f"  {index:4d}/{len(rows)}  FAILED {row.name}: {outcome['reason']}",
                      file=sys.stderr)
                continue
        log[row.piano_id] = entry
        if index % 10 == 0 or index == len(rows):
            print(f"  {index:4d}/{len(rows)}  {row.name[:48]}", file=sys.stderr, flush=True)

    if args.stage in ("transcribe", "all"):
        # One batch, so MuScriptor loads once rather than once per song.
        for piano_id, outcome in transcribe(rows, args.device).items():
            log.setdefault(piano_id, {"piano_id": piano_id}).update({
                "midi_ok": outcome["ok"], "midi_reason": outcome["reason"],
                "notes": outcome.get("notes", 0)})
    write_jsonl(PLAN_JSONL, list(log.values()))
    ok = sum(1 for e in log.values() if e.get("ok"))
    midi = sum(1 for e in log.values() if e.get("midi_ok"))
    print(f"\n{ok} processed, {midi} transcribed", file=sys.stderr)


if __name__ == "__main__":
    main()
