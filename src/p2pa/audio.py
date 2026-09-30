from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


@contextmanager
def atomic_media_output(output: str | Path):
    """Publish encoded media atomically while retaining its suffix for ffmpeg/FluidSynth."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial.{os.getpid()}{output.suffix}")
    try:
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Nothing was written to {temporary}")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable '{name}' is not available on PATH")
    return path


def decode_window(
    path: str | Path,
    start: float,
    duration: float,
    sample_rate: int,
    *,
    channels: int = 2,
    ffmpeg: str = "ffmpeg",
) -> torch.Tensor:
    """Decode [start, start + duration) straight into a [channels, samples] tensor.

    Piped rather than written to a temporary WAV: nothing downstream needs the audio on disk,
    only the latent, so a decode-encode pass never touches the filesystem.
    """
    command = [
        require_executable(ffmpeg), "-hide_banner", "-loglevel", "error",
        "-ss", f"{float(start):.6f}", "-t", f"{float(duration):.6f}", "-i", str(path),
        "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio from {path} at [{start}, {start + duration})")
    usable = (samples.size // channels) * channels
    return torch.from_numpy(samples[:usable].copy()).view(-1, channels).t().contiguous()


def render_midi(
    midi_path: str | Path,
    output: str | Path,
    soundfont: str | Path,
    sample_rate: int,
    *,
    fluidsynth: str = "fluidsynth",
) -> None:
    """Synthesize a MIDI file to WAV.

    FluidSynth degrades instead of failing — it prints 'Instrument not found' and still writes a
    file — so its output is inspected rather than only its exit code.
    """
    soundfont = Path(soundfont)
    if not soundfont.is_file():
        raise FileNotFoundError(f"Soundfont not found: {soundfont}")
    with atomic_media_output(output) as temporary:
        result = subprocess.run(
            [require_executable(fluidsynth), "-ni", "-g", "0.8", "-F", str(temporary),
             "-r", str(int(sample_rate)), str(soundfont), str(midi_path)],
            capture_output=True, text=True, check=True,
        )
        noise = f"{result.stdout}\n{result.stderr}".lower()
        for phrase in ("instrument not found", "substitut"):
            if phrase in noise:
                raise RuntimeError(
                    f"FluidSynth substituted a preset rendering {midi_path}: {result.stderr}"
                )


def save_mp3(waveform: torch.Tensor, sample_rate: int, output: str | Path, ffmpeg: str, bitrate: int) -> None:
    """Encode a [channels, samples] tensor to MP3 by piping raw float32 into ffmpeg."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    channels = waveform.size(0)
    interleaved = waveform.detach().float().cpu().clamp(-1, 1).t().contiguous().numpy()
    with atomic_media_output(output) as temporary:
        subprocess.run(
            [require_executable(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "f32le", "-ar", str(int(sample_rate)), "-ac", str(channels), "-i", "pipe:0",
             "-codec:a", "libmp3lame", "-b:a", f"{int(bitrate)}k", str(temporary)],
            input=interleaved.tobytes(), check=True,
        )


def encode_mp3(source: str | Path, output: str | Path, ffmpeg: str, bitrate: int) -> None:
    """Transcode an existing audio file to MP3."""
    with atomic_media_output(output) as temporary:
        subprocess.run(
            [require_executable(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i",
             str(source), "-codec:a", "libmp3lame", "-b:a", f"{int(bitrate)}k", str(temporary)],
            check=True,
        )


def trim_copy(source: str | Path, output: str | Path, seconds: float, ffmpeg: str) -> None:
    """Copy the first `seconds` of an encoded file without re-encoding it.

    Used to export a cached render at the length that was actually encoded: FluidSynth keeps
    writing through its reverb tail, so the file on disk runs past the clip window while the
    condition latent covers only `[0, seconds)`. Stream copy keeps the exported audio bit-exact
    with what the VAE read.
    """
    with atomic_media_output(output) as temporary:
        subprocess.run(
            [require_executable(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i",
             str(source), "-t", f"{float(seconds):.6f}", "-c:a", "copy", str(temporary)],
            check=True,
        )


def probe_duration(path: str | Path, ffprobe: str = "ffprobe") -> float:
    """Length of an audio file in seconds, or 0.0 when it cannot be determined.

    Needed by corpora of full-length songs, where the window has to be placed inside a track whose
    length is not known in advance. Cheap enough to run once per track at index time: ffprobe
    reads the container header, not the audio.
    """
    result = subprocess.run(
        [require_executable(ffprobe), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return 0.0
