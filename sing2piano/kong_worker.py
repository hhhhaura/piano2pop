"""Transcribe piano recordings with Kong et al.'s high-resolution piano transcription model.

Kong, Li, Song, Wan and Wang, *High-resolution Piano Transcription with Pedals by Regressing Onset
and Offset Times* — chosen for its note velocities. It transcribes a solo piano recording, regresses
real velocities and pedal, and folds the pedal into the note offsets, so notes ring the way the
pianist played them.

`piano_transcription_inference` is installed in the PiCoGen environment (`corpus/picogen/setup.sh`),
so this runs there as a subprocess of `process.py` rather than being imported into p2pa's.

Reads a JSONL of `{"audio": ..., "midi": ..., "piano_id": ...}` jobs and appends one result object
per line.

    $PICOGEN_PYTHON kong_worker.py --jobs jobs.jsonl --out results.jsonl --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Kong's model is fixed at 16 kHz mono.
SAMPLE_RATE = 16000


def decode_mono(path: Path, sample_rate: int = SAMPLE_RATE, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode a whole file to mono float32 through an ffmpeg pipe.

    The library's own `load_audio` is pinned to librosa 0.8 internals that no longer exist in 0.11;
    ffmpeg does the same job with no version surface.
    """
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(int(sample_rate)), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def transcribe(model, audio_path: Path, midi_path: Path) -> dict:
    """Piano audio -> MIDI, in the recording's own timebase. Resumable and atomic."""
    if midi_path.is_file() and midi_path.stat().st_size > 0:
        return {"ok": True, "reason": "cached", "notes": 0}
    if not audio_path.is_file():
        return {"ok": False, "reason": "no_piano_audio", "notes": 0}
    try:
        audio = decode_mono(audio_path)
    except Exception as error:  # noqa: BLE001 - one unreadable file must not stop the batch
        return {"ok": False, "reason": f"load_failed: {type(error).__name__}", "notes": 0}
    if audio.size < SAMPLE_RATE:
        return {"ok": False, "reason": "piano_audio_too_short", "notes": 0}
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = midi_path.with_name(midi_path.name + ".tmp")
    try:
        result = model.transcribe(audio, str(temporary))
    except Exception as error:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        return {"ok": False, "reason": f"transcribe_failed: {type(error).__name__}: {error}"[:200],
                "notes": 0}
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        return {"ok": False, "reason": "wrote_nothing", "notes": 0}
    temporary.replace(midi_path)
    return {"ok": True, "reason": "transcribed", "notes": len(result.get("est_note_events", []))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from piano_transcription_inference import PianoTranscription

    model = PianoTranscription(device=args.device, checkpoint_path=None)
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    with args.out.open("a") as handle:
        for job in jobs:
            audio, midi = Path(job["audio"]), Path(job["midi"])
            result = transcribe(model, audio, midi)
            handle.write(json.dumps({"piano_id": job.get("piano_id", audio.parent.name), **result}) + "\n")
            handle.flush()
            print(f"  {audio.parent.name}  {result['reason']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
