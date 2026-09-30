"""Persistent PiCoGen worker, executed by the isolated Python 3.11 environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from picogen_alignment import align_notes_to_bars, detected_bar_anchors, reconcile_terminal_bar
from picogen_fast import picogen2_module


def _configure_single_process_jukebox(torch) -> None:
    """Keep SheetSage's Jukebox encoder out of distributed/NCCL mode.

    Jukebox's legacy adapter defines ``is_available`` as whether this PyTorch build
    includes distributed support, rather than whether a process group is initialized.
    Consequently every independent corpus worker tries to rendezvous on port 29500.
    These workers never communicate, so use the adapter's existing single-process
    fallbacks instead.
    """
    if torch.distributed.is_initialized():
        raise RuntimeError("PiCoGen worker unexpectedly joined a distributed process group")
    import jukebox.utils.dist_adapter as jukebox_dist

    jukebox_dist.is_available = lambda: False


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True))
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _bar_ticks(events, tokenizer) -> list[int]:
    boundaries = [0]
    bar_length = 4
    for start, end in tokenizer.get_bar_ranges(events, from_start=False):
        for event in events[start:end]:
            if event.etype == "bar" and event.value not in {"bar_start", "bar_end", "bar_N"}:
                bar_length = int(event.value.split("_")[1])
        boundaries.append(boundaries[-1] + tokenizer.ticks_per_beat * bar_length)
    return boundaries


def _write_aligned(path: Path, notes) -> None:
    import miditoolkit

    ticks_per_beat = 480
    ticks_per_second = 2 * ticks_per_beat  # fixed 120 BPM; the corpus consumes absolute seconds
    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    midi.tempo_changes = [miditoolkit.TempoChange(120.0, 0)]
    instrument = miditoolkit.Instrument(program=0, is_drum=False, name="piano")
    for note in notes:
        instrument.notes.append(miditoolkit.Note(
            velocity=max(1, min(127, int(note.velocity))),
            pitch=int(note.pitch),
            start=max(0, round(note.start * ticks_per_second)),
            end=max(1, round(note.end * ticks_per_second)),
        ))
    midi.instruments = [instrument]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    midi.dump(str(temporary))
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_or_extract_features(job, beat_detector, sheetsage):
    beat_path = Path(job["beat_cache"])
    feature_path = Path(job["feature_cache"])
    feature_seconds = 0.0
    if beat_path.is_file() and feature_path.is_file():
        beat_info = json.loads(beat_path.read_text())
        stored = np.load(feature_path)
        return beat_info, stored["melody"], stored["harmony"], feature_seconds, True

    started = time.perf_counter()
    beats, downbeats = beat_detector(Path(job["audio"]))
    beat_info = {"beats": beats.tolist(), "downbeats": downbeats.tolist()}
    _atomic_json(beat_path, beat_info)
    output = sheetsage.infer(audio_path=Path(job["audio"]), beat_information=beat_info)
    melody = np.asarray(output["melody_last_hidden_state"])
    harmony = np.asarray(output["harmony_last_hidden_state"])
    _atomic_npz(feature_path, melody=melody, harmony=harmony)
    return beat_info, melody, harmony, time.perf_counter() - started, False


def _generate(job, model, tokenizer, decode, torch, beat_info, melody, harmony):
    raw_path = Path(job["raw_midi"])
    generation_path = Path(job["generation_cache"])
    if raw_path.is_file() and raw_path.stat().st_size > 0 and generation_path.is_file():
        return json.loads(generation_path.read_text()), 0.0, True

    torch.manual_seed(int(job["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(job["seed"]))
    np.random.seed(int(job["seed"]) % (2**32))
    started = time.perf_counter()
    events = decode(
        model=model,
        tokenizer=tokenizer,
        beat_information=beat_info,
        melody_last_embs=melody,
        harmony_last_embs=harmony,
        temperature=float(job["temperature"]),
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + f".tmp.{os.getpid()}")
    tokenizer.events_to_midi(events).dump(str(temporary))
    os.replace(temporary, raw_path)
    generation = {"source_bar_ticks": _bar_ticks(events, tokenizer), "events": len(events)}
    _atomic_json(generation_path, generation)
    return generation, time.perf_counter() - started, False


def _process(job, model, tokenizer, decode, torch, beat_detector, sheetsage, miditoolkit):
    started = time.perf_counter()
    beat_info, melody, harmony, feature_seconds, features_cached = _load_or_extract_features(
        job, beat_detector, sheetsage
    )
    generation, generation_seconds, generation_cached = _generate(
        job, model, tokenizer, decode, torch, beat_info, melody, harmony
    )
    alignment_started = time.perf_counter()
    anchors = detected_bar_anchors(beat_info["beats"], beat_info["downbeats"])
    source_bar_ticks, trimmed_terminal_bar = reconcile_terminal_bar(
        generation["source_bar_ticks"],
        anchors,
        beat_info["beats"],
        beat_info["downbeats"],
    )
    raw = miditoolkit.MidiFile(job["raw_midi"])
    raw_notes = [
        (note.pitch, note.start, note.end, note.velocity)
        for instrument in raw.instruments if not instrument.is_drum
        for note in instrument.notes
    ]
    aligned = align_notes_to_bars(
        raw_notes,
        source_bar_ticks,
        anchors,
        song_duration=float(job["duration"]),
    )
    if len(aligned) < int(job["minimum_notes"]):
        raise RuntimeError(f"too_few_notes_{len(aligned)}")
    _write_aligned(Path(job["target"]), aligned)
    published = miditoolkit.MidiFile(job["target"])
    published_notes = [
        note for instrument in published.instruments if not instrument.is_drum
        and instrument.program == 0 for note in instrument.notes
    ]
    if len(published_notes) != len(aligned):
        raise RuntimeError("published MIDI failed note-count/program validation")
    metadata = {
        "fingerprint": job["fingerprint"],
        "track_id": job["track_id"],
        "shard": job["shard"],
        "seed": int(job["seed"]),
        "temperature": float(job["temperature"]),
        "notes": len(aligned),
        "bars": len(anchors) - 1,
        "raw_notes": len(raw_notes),
        "trimmed_terminal_zero_beat_bar": trimmed_terminal_bar,
        "duration": float(job["duration"]),
        "midi_sha256": _sha256(Path(job["target"])),
    }
    _atomic_json(Path(job["metadata"]), metadata)
    return {
        **metadata,
        "ok": True,
        "reason": "built",
        "features_cached": features_cached,
        "generation_cached": generation_cached,
        "feature_seconds": round(feature_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "alignment_seconds": round(time.perf_counter() - alignment_started, 3),
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--picogen-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument(
        "--fast-decoder", action=argparse.BooleanOptionalAction, default=True,
        help="retain PiCoGen's KV cache across bars and skip discarded condition computation",
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--features-only", action="store_true")
    phase.add_argument("--generate-only", action="store_true")
    parser.add_argument("--watch-features", action="store_true")
    parser.add_argument("--feature-done", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    # Librosa/numba and matplotlib are imported indirectly by mirtoolkit. Their default user-cache
    # locations are not guaranteed writable on compute nodes or inside batch containers.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/p2p_picogen_numba")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/p2p_picogen_matplotlib")
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.picogen_root))
    import miditoolkit
    import piano_transcription_inference
    import torch

    _configure_single_process_jukebox(torch)

    # mirtoolkit v0.1 imports this removed helper from its optional ByteDance transcription
    # adapter at package import time. PiCoGen only needs BeatThis and SheetSage, but the unrelated
    # import would otherwise prevent either from loading (a known mirtoolkit v0.1 mismatch).
    if not hasattr(piano_transcription_inference, "load_audio_stream"):
        def _unused_transcription_loader(*_args, **_kwargs):
            raise RuntimeError("piano_transcription_inference.load_audio_stream is unavailable")

        piano_transcription_inference.load_audio_stream = _unused_transcription_loader
    from beat_this.inference import File2Beats
    from mirtoolkit import sheetsage
    if args.device.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.device.isdigit():
        device = torch.device("cuda")  # CUDA_VISIBLE_DEVICES remaps the requested GPU to zero.
    else:
        device = torch.device(args.device)
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]

    class LazyBeatDetector:
        """Load BeatThis on first cache miss, then retain it for the rest of the shard."""

        def __init__(self):
            self.detector = None

        def __call__(self, audio):
            if self.detector is None:
                self.detector = File2Beats(
                    checkpoint_path="final0", device=device, float16=False, dbn=True
                )
            return self.detector(audio)

    beat_detector = LazyBeatDetector()
    if args.features_only:
        print(json.dumps({
            "event": "ready", "model_load_seconds": 0.0,
            "device": str(device), "component": "features",
        }), flush=True)
        for position, job in enumerate(jobs, 1):
            started = time.perf_counter()
            try:
                _, _, _, feature_seconds, cached = _load_or_extract_features(
                    job, beat_detector, sheetsage
                )
                result = {
                    "track_id": job["track_id"], "shard": job["shard"],
                    "fingerprint": job["fingerprint"], "ok": True,
                    "reason": "features_cached" if cached else "features_built",
                    "feature_seconds": round(feature_seconds, 3),
                    "seconds": round(time.perf_counter() - started, 3),
                }
            except Exception as error:  # noqa: BLE001 - retain per-song resumability
                result = {
                    "track_id": job["track_id"], "shard": job["shard"],
                    "fingerprint": job["fingerprint"], "ok": False,
                    "reason": f"{type(error).__name__}: {error}"[:300],
                }
            print(json.dumps({"event": "result", "position": position, "total": len(jobs),
                              **result}, sort_keys=True), flush=True)
        return

    if args.generate_only and not args.watch_features:
        missing = [job["track_id"] for job in jobs if not (
            Path(job["beat_cache"]).is_file() and Path(job["feature_cache"]).is_file()
        )]
        if missing:
            raise RuntimeError(
                f"--generate-only requires cached features; missing {len(missing)}, "
                f"first={missing[0]}"
            )
    if args.watch_features and not (args.generate_only and args.feature_done):
        raise RuntimeError("--watch-features requires --generate-only and --feature-done")

    infer = picogen2_module("infer")
    PiCoGenDecoder = picogen2_module("model").PiCoGenDecoder
    Tokenizer = picogen2_module("repr").Tokenizer

    load_started = time.perf_counter()
    assets = picogen2_module("assets")
    load_config = picogen2_module("utils").load_config

    config_path = args.config or assets.config_file()
    hyperparameters = load_config(config_path)
    model = PiCoGenDecoder.from_pretrained(
        ckpt_file=args.checkpoint,
        config_file=config_path,
        device=device,
    )
    tokenizer = Tokenizer(
        vocab_file=args.vocab,
        beat_div=hyperparameters.beat_div,
        ticks_per_beat=hyperparameters.ticks_per_beat,
    )
    decode = infer.decode
    if args.fast_decoder:
        from picogen_fast import decode as fast_decode

        decode = fast_decode
    print(json.dumps({
        "event": "ready", "model_load_seconds": round(time.perf_counter() - load_started, 3),
        "device": str(device),
    }), flush=True)

    pending = list(jobs)
    completed = 0
    while pending:
        ready = [job for job in pending if not args.generate_only or (
            Path(job["beat_cache"]).is_file() and Path(job["feature_cache"]).is_file()
        )]
        if not ready:
            if args.watch_features and not args.feature_done.is_file():
                print(json.dumps({
                    "event": "waiting", "pending": len(pending),
                    "poll_seconds": args.poll_seconds,
                }), flush=True)
                time.sleep(args.poll_seconds)
                continue
            for job in pending:
                completed += 1
                result = {
                    "track_id": job["track_id"], "shard": job["shard"],
                    "fingerprint": job["fingerprint"], "ok": False,
                    "reason": "feature_producer_finished_without_cache",
                }
                print(json.dumps({"event": "result", "position": completed,
                                  "total": len(jobs), **result}, sort_keys=True), flush=True)
            break

        for job in ready:
            try:
                result = _process(
                    job, model, tokenizer, decode, torch, beat_detector, sheetsage, miditoolkit
                )
            except Exception as error:  # noqa: BLE001 - one song must not terminate a corpus shard
                result = {
                    "track_id": job["track_id"], "shard": job["shard"],
                    "fingerprint": job["fingerprint"],
                    "ok": False, "reason": f"{type(error).__name__}: {error}"[:300],
                }
            completed += 1
            pending.remove(job)
            print(json.dumps({"event": "result", "position": completed, "total": len(jobs),
                              **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
