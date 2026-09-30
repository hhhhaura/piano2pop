"""Pack the project's data into tar archives, and unpack them somewhere else.

Two packs, because they answer different questions and only one of them is needed to train.

``train``  everything a training run reads: whole-track latents, conditioning MIDI (the original
           transcription and its variants), beats, the fixed-prompt cache, and the manifest.
``eval``   the held-out audio the evaluation suite needs: the instrumental and conditioning MIDI
           for the validation and test splits, plus the real pianist cover recordings and their
           Kong transcriptions. About 3 GB.

The 184 GB stem tree is in neither, and that is the point: training reads cached latents and never
touches audio, and evaluation only ever needs held-out songs. Carrying the corpus would be
carrying 20x the bytes for nothing.

On the far side, `p2pa-unpack` extracts into a directory and prints the one environment variable
that makes every path in the project resolve against it:

    export P2PA_DATA_ROOT=/scratch/p2pa-data

`config.resolve_path` consults that before falling back to the project root, so the symlink layout
this machine uses is not something the H100 has to reproduce.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path

from .config import TrainConfig, data_root, project_root, resolve_path
from .stages import read_jsonl

MANIFEST_NAME = "p2pa_pack.json"


def _members(root: Path, patterns: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    resolved = []
    for source, arcname in patterns:
        if source.exists():
            resolved.append((source, arcname))
        else:
            print(f"[pack] missing, skipped: {source}", flush=True)
    return resolved


def train_members(cfg) -> list[tuple[Path, str]]:
    corpus = resolve_path(str(cfg.data.source.root))
    return _members(data_root(), [
        (resolve_path(str(cfg.data.latent_dir)), "cache/latents"),
        (resolve_path(str(cfg.data.prompt_dir)), "cache/prompt"),
        (resolve_path(str(cfg.data.manifest_dir)), "cache/manifests"),
        (corpus / str(cfg.data.source.midi_subdir), "corpus/midi"),
        (corpus / str(cfg.data.source.beats_subdir), "corpus/beats"),
    ])


def eval_members(cfg, splits: tuple[str, ...] = ("validation", "test")) -> list[tuple[Path, str]]:
    """Held-out instrumentals only, plus the whole real-cover set.

    The covers are small (196 files, ~700 MB) and are the only conditioning in the suite that is
    genuinely out of distribution, so they travel whole rather than sampled.
    """
    from .paths import audio_path, manifest_path

    members: list[tuple[Path, str]] = []
    rows = [
        row for row in read_jsonl(manifest_path(cfg))
        if row.get("status") == "ok" and row.get("split") in splits
    ]
    for row in rows:
        audio = audio_path(row, cfg)
        if audio.is_file():
            members.append((audio, f"corpus/audio/{row['shard']}/{row['track_id']}/{audio.name}"))

    from .covers import cover_audio_path, cover_midi_path, covers_root, curated_cover_ids

    covers = covers_root(cfg)
    # A cover is useful only as a pair from the declared held-out panel. The workstation contains
    # a larger, evolving transcription pool; allowing directory contents to define the benchmark
    # made its membership and claimed size depend on when the pack happened to run.
    for track_id in curated_cover_ids():
        recording = cover_audio_path(covers, track_id)
        midi = cover_midi_path(covers, track_id)
        if not recording.is_file() or not midi.is_file():
            continue
        members.append((recording, f"covers/raw/{track_id}/piano.m4a"))
        # Pack individual files because workstation and portable layouts differ by one directory.
        shard = midi.parent.name
        members.append((midi, f"covers/cover_midi/{shard}/{midi.name}"))
    return _members(data_root(), members)


def write_archive(output: Path, members: list[tuple[Path, str]], metadata: dict) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    temporary = output.with_name(output.name + ".partial")
    # Uncompressed. Latents are float16 noise and mp3/m4a are already compressed, so gzip spends
    # minutes to save single-digit percent; `tar` streams straight to the wire instead.
    with tarfile.open(temporary, "w") as archive:
        for source, arcname in members:
            archive.add(source, arcname=arcname, recursive=True)
        payload = json.dumps(metadata, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mtime = int(time.time())
        import io

        archive.addfile(info, io.BytesIO(payload))
    os.replace(temporary, output)
    return {
        "archive": str(output),
        "bytes": output.stat().st_size,
        "gigabytes": round(output.stat().st_size / 1024**3, 2),
        "seconds": round(time.time() - started, 1),
        "entries": len(members),
    }


def load_config():
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore

    from .config import config_dir

    ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        return compose(config_name="config")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # No `choices=`: with `nargs="*"` argparse validates the *whole default list* against it as
    # if it were one value, so both `default=["train","eval"]` and the empty default rejected the
    # no-argument form — the only form anyone actually types. Validated by hand below instead.
    parser.add_argument("packs", nargs="*", default=[],
                        help="train, eval, or nothing for both")
    parser.add_argument("--out", type=Path, default=project_root() / "dist")
    args = parser.parse_args()

    packs = list(args.packs) or ["train", "eval"]
    unknown = [name for name in packs if name not in ("train", "eval")]
    if unknown:
        raise SystemExit(f"unknown pack {unknown}; choose from train, eval")

    cfg = load_config()
    for name in packs:
        members = train_members(cfg) if name == "train" else eval_members(cfg)
        metadata = {
            "pack": name,
            "corpus": str(cfg.data.source.name),
            "latent_channels": int(cfg.ace.latent_channels),
            "latent_fps": float(cfg.ace.sample_rate) / float(cfg.ace.hop_length),
            "vae": f"{cfg.ace.assets_repo}/{cfg.ace.vae_subfolder}",
            "prompt_encoder": f"{cfg.ace.assets_repo}/{cfg.ace.text_encoder_subfolder}",
            "prompt_max_length": int(cfg.ace.prompt_max_length),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        result = write_archive(args.out / f"p2pa-{name}.tar", members, metadata)
        print(f"[pack] {json.dumps(result, sort_keys=True)}")


LAYOUT = {
    "cache/latents": ".cache/latents",
    "cache/prompt": ".cache/prompt",
    "cache/manifests": ".cache/manifests",
    "corpus/midi": "data/p2pdata/midi",
    "corpus/beats": "data/p2pdata/beats",
    "corpus/audio": "data/p2pdata/audio",
}


def merge_tree(source: Path, target: Path) -> None:
    """Move a staged tree into place without overwriting files from an earlier pack or rerun."""
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.replace(source, target)
        return
    for entry in source.rglob("*"):
        if not entry.is_file():
            continue
        destination = target / entry.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.replace(entry, destination)


def unpack_main() -> None:
    parser = argparse.ArgumentParser(
        description="Unpack a p2pa archive into a data root and verify it against the manifest."
    )
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    staging = args.root / ".p2pa_staging"
    for archive_path in args.archives:
        print(f"[unpack] {archive_path}", flush=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(staging)
        metadata_path = staging / MANIFEST_NAME
        if metadata_path.is_file():
            print(f"[unpack] {metadata_path.read_text()}")
            metadata_path.unlink()

    for source_name, target_name in LAYOUT.items():
        source = staging / source_name
        target = args.root / target_name
        # Merging rather than replacing: the two packs both write under `data/p2pdata`, and an
        # interrupted transfer may also be unpacked again over its already-published prefix.
        merge_tree(source, target)
    merge_tree(staging / "covers", args.root / "covers")

    print(verify(args.root))
    print(
        f"\nexport P2PA_DATA_ROOT={args.root}\n"
        "Every path in the project resolves against that; no symlinks are needed."
    )


def verify(root: Path) -> str:
    """Re-check every manifest row against what actually landed on disk.

    A truncated transfer produces a tar that extracts cleanly and a corpus that is quietly missing
    a thousand latents, which would look like a smaller dataset rather than like a broken one.
    """
    os.environ["P2PA_DATA_ROOT"] = str(root)
    cfg = load_config()
    from .paths import latent_path, manifest_path, midi_path

    rows = [row for row in read_jsonl(manifest_path(cfg)) if row.get("status") == "ok"]
    missing = {"latent": 0, "midi": 0}
    for row in rows:
        missing["latent"] += not latent_path(row, cfg).is_file()
        missing["midi"] += not midi_path(row, cfg).is_file()
    return json.dumps({"usable_rows": len(rows), "missing": missing}, sort_keys=True)


if __name__ == "__main__":
    main()
