"""Remote-safe separation and transcription over a frozen inventory snapshot.

The orchestrator runs in p2pa's uv environment.  Muscriptor and Demucs run through the absolute
interpreter supplied by ``--mir-python``; this module's worker branch deliberately imports only
their legacy stack.  A normal cluster launch starts six copies of this orchestrator, each with two
workers, all sharing the one GPU exposed to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- legacy muscriptor worker ---------------------------------------------------------------

def _banned_tokens(programs: list[int], max_shift_steps: int = 1001) -> list[int]:
    allowed = frozenset(programs)
    if not allowed or any(not 0 <= value <= 127 for value in allowed):
        raise ValueError(f"invalid GM program allowlist: {sorted(allowed)}")
    program_lo = 3 + max_shift_steps + 128 + 2 + 1
    program_hi = program_lo + 130
    drum_hi = program_hi + 128
    return [program_lo + value for value in range(130) if value not in allowed] + list(
        range(program_hi, drum_hi)
    )


def _muscriptor_worker(argv: list[str]) -> int:
    import torch
    from muscriptor.transcription_model import TranscriptionModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--muscriptor-worker", action="store_true")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)

    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    mine = [job for index, job in enumerate(jobs) if index % args.stride == args.index]
    if not mine:
        return 0
    model = TranscriptionModel.load_model(weights_path=args.model, device=None)
    model._model = model._model.to(torch.float32)
    language_model = model._model
    banned = None
    original_logits = language_model._compute_logits

    def masked_logits(*positional, **keywords):
        logits = original_logits(*positional, **keywords)
        if banned is not None:
            logits.index_fill_(1, banned, float("-inf"))
        return logits

    language_model._compute_logits = masked_logits
    batch_size = args.batch_size if torch.cuda.is_available() else 1
    failures = 0
    print(f"P2PA_MUSCRIPTOR_READY worker={args.index} jobs={len(mine)}", flush=True)
    for position, job in enumerate(mine, 1):
        programs = [int(value) for value in job.get("programs", [0])]
        banned = torch.tensor(
            _banned_tokens(programs), dtype=torch.long, device=language_model.emb.weight.device
        )
        output = Path(job["midi"])
        if output.is_file() and output.stat().st_size > 0:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
        try:
            with tempfile.TemporaryDirectory(prefix="p2pa_mus_") as workspace:
                clip = Path(workspace) / "clip.wav"
                subprocess.run(
                    [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                     "-ss", f"{float(job['start']):.6f}",
                     "-t", f"{float(job['seconds']):.6f}", "-i", str(job["audio"]),
                     "-threads", "1", "-ac", "1", "-ar", "44100", str(clip)],
                    check=True,
                )
                midi = model.transcribe_to_midi(
                    clip,
                    instruments=list(job.get("instruments", ["acoustic_piano"])),
                    batch_size=batch_size,
                )
            if not midi:
                raise RuntimeError("empty MIDI")
            temporary.write_bytes(midi)
            os.replace(temporary, output)
        except Exception as error:  # noqa: BLE001 - one corrupt song is an inventory result
            failures += 1
            temporary.unlink(missing_ok=True)
            print(f"FAILED {job['sample_id']}: {type(error).__name__}: {error}", file=sys.stderr)
        if position % 25 == 0 or position == len(mine):
            print(f"P2PA_PROGRESS {position}/{len(mine)} failures={failures}", flush=True)
    return 0


# --- orchestrator helpers -------------------------------------------------------------------

SEPARATION_REVISION = "htdemucs_vocals_then_6s_48k_v2"
# Batching changes the decode: different batch shapes pad differently, and greedy sampling turns a
# last-bit logit difference into a different note sequence. Three of four probe clips were
# identical between batch 4 and 12 and the fourth differed by 15 notes of ~200, so the revision is
# bumped to record which regime produced a cache entry. Published MIDI is unaffected -- resume is
# keyed on `_valid_piano_midi` of the final file, not on this string.
TRANSCRIPTION_REVISION = "muscriptor_medium_gm0_chunk30_overlap5_b12_v2"

# The first pass publishes vocals and instrumental. The second pass runs htdemucs_6s on
# that instrumental and publishes its five useful non-vocal stems; its vocal residual is dropped.
# The original input is retained as mix.mp3. This list is also the completeness sentinel.
SEPARATION_OUTPUTS = (
    "vocals.mp3", "drums.mp3", "bass.mp3", "other.mp3", "guitar.mp3", "piano.mp3",
    "instrumental.mp3", "mix.mp3",
)


def _separation_complete(target_dir: Path, ffprobe: str) -> bool:
    return all(
        (target_dir / name).is_file() and _media_valid(target_dir / name, ffprobe)
        for name in SEPARATION_OUTPUTS
    )


def _separation_marker(row: dict, cache_root: Path) -> Path:
    return cache_root / "separate" / _fingerprint({
        "revision": SEPARATION_REVISION, "track_id": row["track_id"]
    }) / "complete.json"


def _separation_ready(row: dict, args) -> bool:
    return _separation_marker(row, args.cache_root).is_file() and _separation_complete(
        _audio_dir(row, args.output_root), args.ffprobe
    )


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(path: Path, ffprobe: str) -> dict:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {"sample_rate": int(stream.get("sample_rate") or 0),
            "duration": float(stream.get("duration") or 0.0)}


def _media_valid(path: Path, ffprobe: str, *, sample_rate: int | None = 48000) -> bool:
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        info = _probe(path, ffprobe)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return info["duration"] >= 60.0 and sample_rate in (None, info["sample_rate"])


def _signal_valid(path: Path, ffmpeg: str) -> tuple[bool, str]:
    """Decode the complete target and reject silent or materially flat-clipped audio."""
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "astats=metadata=0:reset=0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return False, "decode_failed"
    rms_values = [
        float(value)
        for value in re.findall(r"RMS level dB:\s*(-?(?:\d+(?:\.\d+)?|inf))", result.stderr)
        if value != "-inf"
    ]
    flat_values = [
        float(value)
        for value in re.findall(r"Flat factor:\s*(-?(?:\d+(?:\.\d+)?|inf))", result.stderr)
        if value != "-inf"
    ]
    if not rms_values or max(rms_values) < -50.0:
        return False, "silent_audio"
    # FFmpeg's flat factor is the dB ratio of repeated min/max samples. A positive value means a
    # conspicuous run of identical full-scale samples, unlike an ordinary mastered peak at 0 dB.
    if flat_values and max(flat_values) > 0.0:
        return False, "flat_clipped_audio"
    return True, ""


def _source_audio(row: dict, source_root: Path) -> Path:
    recorded = row.get("artifacts", {}).get("raw_audio")
    candidate = Path(str(recorded)) if recorded else Path()
    if recorded and candidate.is_file():
        return candidate
    return source_root / "pop2piano" / "raw" / row["track_id"] / "pop.m4a"


def _audio_dir(row: dict, output_root: Path) -> Path:
    return output_root / "p2pdata" / "audio" / row["shard"] / row["track_id"]


def _midi_path(row: dict, output_root: Path) -> Path:
    return output_root / "p2pdata" / "midi" / row["shard"] / row["track_id"] / "full-piano.mid"


STEMSETS = {
    "full6": ("drums", "bass", "other", "guitar", "piano"),
    "nobass": ("drums", "other", "guitar", "piano"),
    "harmonic": ("bass", "other", "guitar", "piano"),
    "pianobass": ("bass", "piano"),
}

# Function-selective transcription decodes every stem mixture as piano (MIDI program 0), exactly
# as the full-instrumental transcription is.
VARIANT_TARGETS = {
    "piano": {"instruments": ("acoustic_piano",), "programs": (0,)},
}


def _variant_choices(track_id: str) -> list[tuple[str, str]]:
    """Every stem mixture, each decoded as piano: `<stemset>-piano.mid`. The same for every track."""
    return [(stemset, "piano") for stemset in STEMSETS]


def _transcription_complete(row: dict, args) -> bool:
    midi_dir = _midi_path(row, args.output_root).parent
    targets = [_midi_path(row, args.output_root)] + [
        midi_dir / f"{stemset}-{decode_target}.mid"
        for stemset, decode_target in _variant_choices(row["track_id"])
    ]
    return all(_valid_piano_midi(path) for path in targets)


def _separation_completion_path(args) -> Path:
    run = _fingerprint({
        "revision": SEPARATION_REVISION, "shard": args.shard,
        "shards": args.num_shards, "limit": args.limit,
        "inventory_sha256": _file_sha256(args.inventory),
    })
    return args.cache_root / "completions" / "separate" / f"shard_{args.shard}_{run}.json"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_ffmpeg(command: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial.{os.getpid()}.{threading.get_ident()}{output.suffix}")
    try:
        subprocess.run([*command, str(temporary)], check=True, capture_output=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg wrote nothing to {temporary}")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _separate_one(row: dict, args) -> dict:
    started = time.time()
    print(f"P2PA_SEPARATE_START track={row['track_id']}", flush=True)
    target_dir = _audio_dir(row, args.output_root)
    target = target_dir / "instrumental.mp3"
    marker = _separation_marker(row, args.cache_root)
    source = _source_audio(row, args.source_root)
    marker_present = marker.is_file()
    if marker_present and _separation_complete(target_dir, args.ffprobe):
        try:
            recorded = json.loads(marker.read_text())
            source_matches = not source.is_file() or recorded.get("source_sha256") == _file_sha256(source)
            if recorded.get("sha256") == _file_sha256(target) and source_matches:
                return {"track_id": row["track_id"], "ok": True, "state": "cached"}
        except (OSError, json.JSONDecodeError):
            pass
    if not source.is_file():
        return {"track_id": row["track_id"], "ok": False, "reason": "missing_raw_audio"}
    with tempfile.TemporaryDirectory(prefix=f"p2pa_demucs_{row['track_id']}_") as workspace:
        workspace_path = Path(workspace)
        vocal_pass = workspace_path / "vocal_pass"
        result = subprocess.run(
            [str(args.mir_python), "-m", "demucs.separate", "-n", "htdemucs",
             "--two-stems", "vocals", "--device", "cuda", "-o", str(vocal_pass), str(source)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            return {"track_id": row["track_id"], "ok": False,
                    "reason": f"demucs_vocal_failed:{tail[-1] if tail else '?'}"[:300]}
        first = vocal_pass / "htdemucs" / source.stem
        vocals = first / "vocals.wav"
        instrumental = first / "no_vocals.wav"
        if not vocals.is_file() or not instrumental.is_file():
            return {"track_id": row["track_id"], "ok": False,
                    "reason": "demucs_missing_vocal_or_instrumental"}
        print(f"P2PA_SEPARATE_VOCALS_DONE track={row['track_id']}", flush=True)

        six_pass = workspace_path / "six_pass"
        result = subprocess.run(
            [str(args.mir_python), "-m", "demucs.separate", "-n", "htdemucs_6s",
             "--device", "cuda", "-o", str(six_pass), str(instrumental)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            return {"track_id": row["track_id"], "ok": False,
                    "reason": f"demucs_6s_failed:{tail[-1] if tail else '?'}"[:300]}
        produced = six_pass / "htdemucs_6s" / instrumental.stem
        stems = {name: produced / f"{name}.wav" for name in
                 ("drums", "bass", "other", "vocals", "guitar", "piano")}
        if not all(path.is_file() for path in stems.values()):
            return {"track_id": row["track_id"], "ok": False, "reason": "demucs_missing_stems"}
        print(f"P2PA_SEPARATE_6S_DONE track={row['track_id']}", flush=True)

        # Publish the first-pass vocal and the five non-vocal second-pass stems. The second-pass
        # vocal is only separation leakage from an already instrumental signal and is discarded.
        publish = {name: stems[name] for name in ("drums", "bass", "other", "guitar", "piano")}
        publish["vocals"] = vocals
        for name, stem in publish.items():
            _atomic_ffmpeg(
                [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(stem),
                 "-ar", "48000", "-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
                target_dir / f"{name}.mp3",
            )
        _atomic_ffmpeg(
            [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-ar", "48000", "-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
            target_dir / "mix.mp3",
        )
        _atomic_ffmpeg(
            [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(instrumental),
             "-ar", "48000",
             "-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
            target,
        )
    if not _media_valid(target, args.ffprobe):
        return {"track_id": row["track_id"], "ok": False, "reason": "invalid_instrumental"}
    signal_ok, signal_reason = _signal_valid(target, args.ffmpeg)
    if not signal_ok:
        target.unlink(missing_ok=True)
        return {"track_id": row["track_id"], "ok": False, "reason": signal_reason}
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps({
        "track_id": row["track_id"], "revision": SEPARATION_REVISION,
        "instrumental": str(target), "sha256": _file_sha256(target),
        "source_sha256": _file_sha256(source),
        "seconds": round(time.time() - started, 3),
    }, sort_keys=True))
    os.replace(temporary, marker)
    print(
        f"P2PA_SEPARATE_PUBLISHED track={row['track_id']} seconds={time.time() - started:.1f}",
        flush=True,
    )
    return {"track_id": row["track_id"], "ok": True, "state": "separated",
            "seconds": round(time.time() - started, 3)}


def _chunk_windows(duration: float, seconds: float = 30.0, overlap: float = 5.0):
    stride, half, start, windows = max(1.0, seconds - overlap), overlap / 2.0, 0.0, []
    while start < duration - 1e-6:
        length = min(seconds, duration - start)
        last = start + length >= duration - 1e-6
        windows.append((start, length, 0.0 if not windows else start + half,
                        duration if last else start + stride + half))
        if last:
            break
        start += stride
    return windows


def _rms_envelope(path: Path, ffmpeg: str):
    import numpy as np
    raw = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", "22050", "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    hop = 220
    frames = audio.size // hop
    if frames < 1:
        return np.full(1, -120.0, dtype=np.float32)
    power = np.maximum(np.mean(audio[:frames * hop].reshape(frames, hop).astype("float64") ** 2,
                               axis=1), 1e-12)
    return (10.0 * np.log10(power)).astype(np.float32)


def _add_velocity(notes, envelope):
    import numpy as np

    from .pianoroll import Note
    if not notes:
        return []
    energy = [float(envelope[min(max(int(note.start / 0.01), 0), envelope.size - 1)])
              for note in notes]
    order = np.argsort(np.asarray(energy), kind="stable")
    ranks = np.empty(len(notes), dtype=np.float64)
    ranks[order] = (np.arange(len(notes)) + 0.5) / len(notes)
    quantiles = (0.0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0)
    velocities = (25, 39, 46, 50, 57, 64, 70, 74, 76, 78, 95)
    return [Note(note.pitch, note.start, note.end,
                 int(np.clip(round(np.interp(ranks[index], quantiles, velocities)), 25, 95)),
                 False, 0) for index, note in enumerate(notes)]


def _stitch_track(
    plan: list[dict], target: Path, audio: Path, ffmpeg: str, programs: tuple[int, ...] = (0,)
) -> dict:
    from .pianoroll import Note, parse_notes, write_midi
    notes = []
    for chunk in plan:
        path = Path(chunk["midi"])
        if not path.is_file():
            return {"ok": False, "reason": f"missing_chunk:{chunk['index']}"}
        try:
            chunk_notes = parse_notes(path)
        except Exception:  # noqa: BLE001 - delete a torn chunk so the next run recreates it
            path.unlink(missing_ok=True)
            return {"ok": False, "reason": f"invalid_chunk:{chunk['index']}"}
        for note in chunk_notes:
            onset = note.start + chunk["start"]
            if chunk["core_start"] - 1e-9 <= onset < chunk["core_end"]:
                notes.append(Note(note.pitch, onset, note.end + chunk["start"], note.velocity,
                                  note.is_drum, note.program))
    allowed = frozenset(programs)
    off_program = sum(note.is_drum or note.program not in allowed for note in notes)
    notes = sorted((note for note in notes if not note.is_drum and note.program in allowed),
                   key=lambda note: (note.start, note.pitch))
    if not notes:
        return {"ok": False, "reason": "all_notes_off_program", "off_program": off_program}
    notes = _add_velocity(notes, _rms_envelope(audio, ffmpeg))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    write_midi(notes, temporary, program=0)
    os.replace(temporary, target)
    parsed = parse_notes(target)
    if not parsed or any(note.is_drum or note.program != 0 for note in parsed):
        target.unlink(missing_ok=True)
        return {"ok": False, "reason": "published_midi_failed_validation"}
    return {"ok": True, "notes": len(parsed), "off_program": off_program}


def _stemset_audio(row: dict, stemset: str, args) -> Path:
    """Build a content-addressed, resumable mix for one variant stemset."""
    audio_dir = _audio_dir(row, args.output_root)
    sources = [audio_dir / f"{name}.mp3" for name in STEMSETS[stemset]]
    missing = [path.name for path in sources if not _media_valid(path, args.ffprobe)]
    if missing:
        raise RuntimeError(f"missing valid stems for {row['track_id']}/{stemset}: {missing}")
    key = _fingerprint({
        "revision": "variant_stemsets_v1",
        "stemset": stemset,
        "sources": [(path.name, _file_sha256(path)) for path in sources],
    })
    target = args.cache_root / "variant_audio" / key / f"{stemset}.mp3"
    if _media_valid(target, args.ffprobe):
        return target
    inputs = [value for path in sources for value in ("-i", str(path))]
    _atomic_ffmpeg(
        [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
         "-filter_complex", f"amix=inputs={len(sources)}:normalize=0", "-ar", "48000",
         "-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
        target,
    )
    if not _media_valid(target, args.ffprobe):
        target.unlink(missing_ok=True)
        raise RuntimeError(f"invalid mixed stemset for {row['track_id']}/{stemset}")
    return target


def _transcribe(rows: list[dict], args) -> list[dict]:
    from .audio import probe_duration
    fingerprint = _fingerprint({
        "revision": TRANSCRIPTION_REVISION, "variants": "k5_stemset_x_program_v1",
        "shard": args.shard,
    })
    root = args.cache_root / "transcribe" / fingerprint
    jobs: list[dict] = []
    plans: dict[str, list[dict]] = {}
    for row in rows:
        track_id = row["track_id"]
        midi_dir = _midi_path(row, args.output_root).parent
        specs = [{
            "name": "baseline", "target": _midi_path(row, args.output_root),
            "audio": _audio_dir(row, args.output_root) / "instrumental.mp3",
            "instruments": ("acoustic_piano",), "programs": (0,),
        }]
        for stemset, decode_target in _variant_choices(track_id):
            settings = VARIANT_TARGETS[decode_target]
            try:
                audio = _stemset_audio(row, stemset, args)
            except RuntimeError as error:
                specs.append({
                    "name": f"{stemset}-{decode_target}",
                    "target": midi_dir / f"{stemset}-{decode_target}.mid",
                    "error": str(error),
                })
                continue
            specs.append({
                "name": f"{stemset}-{decode_target}",
                "target": midi_dir / f"{stemset}-{decode_target}.mid",
                "audio": audio, "instruments": settings["instruments"],
                "programs": settings["programs"],
            })

        plans[track_id] = []
        for spec in specs:
            target = Path(spec["target"])
            if spec.get("error"):
                plans[track_id].append(spec)
                continue
            if _valid_piano_midi(target):
                plans[track_id].append({**spec, "cached": True})
                continue
            if target.is_file():
                target.unlink(missing_ok=True)
            audio = Path(spec["audio"])
            if not _media_valid(audio, args.ffprobe):
                plans[track_id].append({**spec, "error": "missing_valid_audio"})
                continue
            duration = probe_duration(audio, args.ffprobe)
            plan = []
            audio_key = _fingerprint({
                "revision": TRANSCRIPTION_REVISION,
                "audio_sha256": _file_sha256(audio),
                "instruments": spec["instruments"], "programs": spec["programs"],
            })
            for index, (start, seconds, core_start, core_end) in enumerate(
                _chunk_windows(duration)
            ):
                midi = root / audio_key / row["shard"] / track_id / spec["name"] / (
                    f"chunk_{index:04d}.mid"
                )
                jobs.append({
                    "sample_id": f"{track_id}:{spec['name']}:{index}",
                    "audio": str(audio), "start": start, "seconds": seconds,
                    "midi": str(midi), "instruments": spec["instruments"],
                    "programs": spec["programs"],
                })
                plan.append({
                    "index": index, "start": start, "core_start": core_start,
                    "core_end": core_end, "midi": str(midi),
                })
            plans[track_id].append({**spec, "plan": plan})
    if jobs:
        jobs_file = root / f"jobs_shard_{args.shard}.jsonl"
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = jobs_file.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs))
        os.replace(temporary, jobs_file)
        commands = [
            [str(args.mir_python), str(Path(__file__).resolve()), "--muscriptor-worker",
             "--jobs", str(jobs_file), "--index", str(worker), "--stride", str(args.workers),
             # Muscriptor cuts a 30 s clip into six 5 s chunks, so a batch of 4 costs two forward
             # passes per clip. Measured on an idle H200, one worker, four real clips: 16.43 s at
             # batch 4 against 9.23 s at batch 12, a consistent 1.78x. Batch size is an upper
             # bound rather than a reservation, so the headroom above six is free.
             "--model", "medium", "--batch-size", "12", "--ffmpeg", args.ffmpeg]
            for worker in range(args.workers)
        ]
        processes = [subprocess.Popen(command) for command in commands]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"muscriptor workers failed: {codes}")
    results = []
    for track_id, items in plans.items():
        artifacts = {}
        for item in items:
            name = item["name"]
            if item.get("cached"):
                artifacts[name] = {"ok": True, "reason": "cached"}
            elif item.get("error"):
                artifacts[name] = {"ok": False, "reason": item["error"]}
            else:
                artifacts[name] = _stitch_track(
                    item["plan"], Path(item["target"]), Path(item["audio"]), args.ffmpeg,
                    tuple(item["programs"]),
                )
        baseline = artifacts["baseline"]
        variants = {name: result for name, result in artifacts.items() if name != "baseline"}
        results.append({
            "track_id": track_id, "ok": bool(baseline.get("ok")),
            "state": "transcribed" if baseline.get("ok") else "failed",
            "baseline": baseline, "selected": sorted(variants), "variants": variants,
            "variants_ok": sum(bool(result.get("ok")) for result in variants.values()),
            "variants_failed": sum(not bool(result.get("ok")) for result in variants.values()),
        })
    return results


def transcribe_piano(
    items: list[tuple[str, Path, Path]], *, mir_python: Path, cache_root: Path,
    workers: int = 1, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
) -> dict[str, dict]:
    """MuScriptor piano transcription of arbitrary files, exactly as the corpus's baseline is made.

    `items` is `(name, audio, target_midi)`. Same 30 s chunking with 5 s overlap, same MIDI-program
    0 decode, same onset-energy velocities, same stitching as `_transcribe`; used for the test set's
    training-distribution inputs, whose instrumentals are not in the corpus layout. Resumable: a
    valid target is kept.
    """
    from .audio import probe_duration
    root = cache_root / "transcribe_piano" / _fingerprint({"revision": TRANSCRIPTION_REVISION})
    jobs, plans, results = [], {}, {}
    for name, audio, target in items:
        if _valid_piano_midi(target):
            results[name] = {"ok": True, "reason": "cached"}
            continue
        # Any sample rate: MuScriptor's worker decodes through ffmpeg at its own rate, and the test
        # set's instrumentals are not the corpus's 48 kHz separations.
        if not _media_valid(audio, ffprobe, sample_rate=None):
            results[name] = {"ok": False, "reason": "missing_valid_audio"}
            continue
        key = _fingerprint({"revision": TRANSCRIPTION_REVISION, "audio_sha256": _file_sha256(audio),
                            "instruments": ("acoustic_piano",), "programs": (0,)})
        plan = []
        for index, (start, seconds, core_start, core_end) in enumerate(
            _chunk_windows(probe_duration(audio, ffprobe))
        ):
            midi = root / key / f"chunk_{index:04d}.mid"
            jobs.append({"sample_id": f"{name}:{index}", "audio": str(audio), "start": start,
                         "seconds": seconds, "midi": str(midi),
                         "instruments": ("acoustic_piano",), "programs": (0,)})
            plan.append({"index": index, "start": start, "core_start": core_start,
                         "core_end": core_end, "midi": str(midi)})
        plans[name] = (plan, audio, target)
    if jobs:
        jobs_file = root / "jobs.jsonl"
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_file.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs))
        processes = [subprocess.Popen([
            str(mir_python), str(Path(__file__).resolve()), "--muscriptor-worker",
            "--jobs", str(jobs_file), "--index", str(worker), "--stride", str(workers),
            "--model", "medium", "--batch-size", "12", "--ffmpeg", ffmpeg,
        ]) for worker in range(workers)]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"muscriptor workers failed: {codes}")
    for name, (plan, audio, target) in plans.items():
        results[name] = _stitch_track(plan, target, audio, ffmpeg, (0,))
    return results


def _append_ledger(path: Path, result: dict, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _follow_transcribe(rows: list[dict], args, ledger: Path) -> list[dict]:
    """Consume fully separated tracks until the matching separation shard declares completion."""
    lock = threading.Lock()
    results: list[dict] = []
    attempted: set[str] = set()
    for row in rows:
        if _transcription_complete(row, args):
            attempted.add(row["track_id"])
            results.append({
                "track_id": row["track_id"], "ok": True, "state": "cached",
                "variants_ok": 5, "variants_failed": 0,
            })
    completion = _separation_completion_path(args)
    while True:
        ready = [
            row for row in rows
            if row["track_id"] not in attempted
            and _separation_ready(row, args)
        ][:args.follow_batch_size]
        if ready:
            print(
                f"P2PA_TRANSCRIBE_FOLLOW_READY shard={args.shard} batch={len(ready)} "
                f"attempted={len(attempted)}/{len(rows)}",
                flush=True,
            )
            batch = _transcribe(ready, args)
            for result in batch:
                attempted.add(result["track_id"])
                _append_ledger(ledger, result, lock)
                results.append(result)
            continue
        # A transcription run may repartition already-separated data (for example, twelve
        # transcription shards after a six-shard separation run). In that case there is no
        # matching separation completion marker, but every row being complete is conclusive.
        if len(attempted) == len(rows):
            print(
                f"P2PA_TRANSCRIBE_FOLLOW_DONE shard={args.shard} "
                f"attempted={len(attempted)}/{len(rows)} missing=0",
                flush=True,
            )
            return results
        if completion.is_file():
            missing = [row for row in rows if row["track_id"] not in attempted]
            for row in missing:
                result = {
                    "track_id": row["track_id"], "ok": False,
                    "state": "failed", "reason": "separation_finished_without_complete_outputs",
                    "variants_ok": 0, "variants_failed": 5,
                }
                _append_ledger(ledger, result, lock)
                results.append(result)
            print(
                f"P2PA_TRANSCRIBE_FOLLOW_DONE shard={args.shard} "
                f"attempted={len(attempted)}/{len(rows)} missing={len(missing)}",
                flush=True,
            )
            return results
        print(
            f"P2PA_TRANSCRIBE_FOLLOW_WAIT shard={args.shard} "
            f"attempted={len(attempted)}/{len(rows)} poll={args.poll_seconds}s",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def _valid_piano_midi(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from .pianoroll import parse_notes

        notes = parse_notes(path)
    except Exception:  # noqa: BLE001 - verification reports invalid artifacts instead of dying
        return False
    return bool(notes) and all(not note.is_drum and note.program == 0 for note in notes)


def shard_rows(rows: list[dict], shard: int, num_shards: int, limit: int = 0) -> list[dict]:
    """Stable modulo partition used by both launch code and regression tests."""
    eligible = [
        row
        for row in rows
        if row.get("asset_state") in {"raw_only", "rejected"}
        and row.get("reason")
        in {"awaiting_separation_and_transcription", "missing_baseline_midi"}
    ]
    eligible.sort(key=lambda row: row["track_id"])
    selected = [row for index, row in enumerate(eligible) if index % num_shards == shard]
    return selected[:limit] if limit else selected


def transcribe_rows(rows: list[dict], args) -> list[dict]:
    """Residue plus fully separated ready rows whose deterministic K5 set is incomplete."""
    eligible = []
    for row in rows:
        state, reason = row.get("asset_state"), row.get("reason")
        residue = state in {"raw_only", "rejected"} and reason in {
            "awaiting_separation_and_transcription", "missing_baseline_midi",
        }
        ready_backfill = (
            state == "ready" and _separation_ready(row, args)
            and not _transcription_complete(row, args)
        )
        if residue or ready_backfill:
            eligible.append(row)
    eligible.sort(key=lambda row: row["track_id"])
    selected = [
        row for index, row in enumerate(eligible)
        if index % args.num_shards == args.shard
    ]
    return selected[:args.limit] if args.limit else selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("separate", "transcribe", "verify"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--mir-python", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--follow-batch-size", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or args.workers != 2:
        raise SystemExit("cluster corpus processing requires a positive shard count and 2 workers per shard")
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"shard must be in [0, {args.num_shards})")
    if args.follow and args.stage != "transcribe":
        raise SystemExit("--follow is only valid for the transcribe stage")
    if args.follow_batch_size < 1 or args.poll_seconds <= 0:
        raise SystemExit("follow batch size and poll seconds must be positive")
    if not args.mir_python.is_file():
        raise SystemExit(f"missing MIR interpreter {args.mir_python}; run scripts/setup_corpus_tools.sh")

    inventory = _read_jsonl(args.inventory)
    mine = (
        transcribe_rows(inventory, args)
        if args.stage == "transcribe"
        else shard_rows(inventory, args.shard, args.num_shards, args.limit)
    )
    ledger = (
        args.cache_root / "ledgers" / args.stage / f"shards_{args.num_shards}"
        / f"shard_{args.shard}.jsonl"
    )
    if args.stage == "separate":
        lock = threading.Lock()
        results = []
        print(
            f"P2PA_SEPARATION_READY shard={args.shard}/{args.num_shards} "
            f"rows={len(mine)} workers={args.workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_separate_one, row, args): row for row in mine}
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                _append_ledger(ledger, result, lock)
                results.append(result)
                print(
                    f"P2PA_SEPARATION_PROGRESS shard={args.shard} "
                    f"completed={completed}/{len(mine)} ok={sum(bool(r.get('ok')) for r in results)} "
                    f"failed={sum(not bool(r.get('ok')) for r in results)}",
                    flush=True,
                )
    elif args.stage == "transcribe":
        if args.follow:
            results = _follow_transcribe(mine, args, ledger)
        else:
            results = _transcribe(mine, args)
            lock = threading.Lock()
            for result in results:
                _append_ledger(ledger, result, lock)
    else:
        results = []
        for row in mine:
            target = _audio_dir(row, args.output_root) / "instrumental.mp3"
            media_ok = _media_valid(target, args.ffprobe)
            signal_ok, signal_reason = _signal_valid(target, args.ffmpeg) if media_ok else (False, "invalid_audio")
            midi_ok = _valid_piano_midi(_midi_path(row, args.output_root))
            results.append(
                {
                    "track_id": row["track_id"],
                    "ok": media_ok and signal_ok and midi_ok,
                    "audio": "ok" if media_ok and signal_ok else signal_reason,
                    "midi": "ok" if midi_ok else "invalid_midi",
                }
            )
    summary = {"stage": args.stage, "shard": args.shard, "shards": args.num_shards,
               "workers": args.workers, "rows": len(mine),
               "ok": sum(bool(result.get("ok")) for result in results),
               "failed": sum(not bool(result.get("ok")) for result in results)}
    if args.stage == "transcribe":
        summary["variants_ok"] = sum(int(result.get("variants_ok", 0)) for result in results)
        summary["variants_failed"] = sum(
            int(result.get("variants_failed", 0)) for result in results
        )
    if args.stage == "separate":
        _atomic_json(_separation_completion_path(args), {
            "revision": SEPARATION_REVISION, "completed": time.time(), **summary,
        })
    print(json.dumps(summary, sort_keys=True))
    if args.fail_on_error and summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if "--muscriptor-worker" in sys.argv:
        sys.exit(_muscriptor_worker(sys.argv[1:]))
    main()
