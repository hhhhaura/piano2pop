"""Incremental, atomic, shardable preparation-stage helpers.

Every prep stage is the same loop — walk the manifest, skip what is already on disk, publish
atomically — so that loop lives here once. A stage supplies only two functions: `row_complete`,
which recomputes the expected artifact paths from the *current* config, and `process_row`.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

# --- identity ---------------------------------------------------------------

def fingerprint(parts: object) -> str:
    """Stable hash of everything that determines an artifact. Order-independent for dicts."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def stable_split(key: str, seed: int, train_fraction: float, validation_fraction: float) -> str:
    """Assign a split by hashing a group key, so groups never straddle splits."""
    unit = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little") / 2**64
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "test"


# --- jsonl ------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(temp, path)


# --- locking ----------------------------------------------------------------

@contextmanager
def path_lock(path: Path) -> Iterator[None]:
    """Exclusive advisory lock keyed on a sidecar .lock file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def flush_rows(manifest: Path, rows: list[dict], indices: list[int]) -> None:
    """Merge only this shard's changed rows into the on-disk manifest."""
    if not indices:
        return
    with path_lock(manifest):
        on_disk = read_jsonl(manifest)
        if len(on_disk) != len(rows):
            raise RuntimeError(
                f"Manifest length changed under {manifest}: disk={len(on_disk)} memory={len(rows)}. "
                "Do not re-index while a stage is running."
            )
        for index in indices:
            on_disk[index] = rows[index]
        write_jsonl(manifest, on_disk)


# --- completeness -----------------------------------------------------------

def artifact_ready(
    path: object,
    validate: Callable[[Path], bool] | None = None,
    *,
    remove_invalid: bool = True,
) -> bool:
    """True when `path` exists and passes `validate`.

    Presence alone is not proof: an interrupted process leaves truncated files. Invalid
    artifacts are deleted by default so the next pass recomputes them.
    """
    if not (isinstance(path, (str, Path)) and str(path)):
        return False
    path = Path(path)
    if not path.is_file():
        return False
    if validate is None:
        return True
    try:
        valid = bool(validate(path))
    except Exception:  # noqa: BLE001 - an artifact we cannot even read counts as invalid
        valid = False
    if valid:
        return True
    if remove_invalid:
        path.unlink(missing_ok=True)
    return False


def expected_completeness(
    expected_paths: Callable[[dict, object], dict[str, Path]],
    validate: Callable[[Path], bool] | None = None,
) -> Callable[[dict, object], bool]:
    """Build a completeness predicate that recomputes paths from the *current* config.

    This is what makes content-addressed naming self-invalidating: when a parameter changes,
    the expected path changes, the old artifact is not consulted, and the row is redone.
    Judging completeness from a path previously recorded on the row cannot do this.
    """
    def complete(row: dict, cfg: object) -> bool:
        return all(artifact_ready(path, validate) for path in expected_paths(row, cfg).values())

    return complete


# --- atomic publish ---------------------------------------------------------

@contextmanager
def atomic_output(output: Path) -> Iterator[Path]:
    """Yield a temp path; on clean exit publish it atomically to `output`.

    If another process published first, its artifact is equally valid and this one is dropped.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    try:
        yield temp
        if not temp.exists():
            raise RuntimeError(f"Nothing was written to {temp}")
        if output.is_file():
            temp.unlink(missing_ok=True)
        else:
            os.replace(temp, output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise




# --- sharding ---------------------------------------------------------------

def shard_spec(cfg) -> tuple[int, int]:
    shards = int(getattr(cfg.prep, "num_shards", 1) or 1)
    shard = int(getattr(cfg.prep, "shard", 0) or 0)
    if shards < 1:
        raise ValueError("prep.num_shards must be >= 1")
    if not 0 <= shard < shards:
        raise ValueError(f"prep.shard must be in [0, {shards})")
    return shard, shards


# --- stage loop -------------------------------------------------------------

def run_stage(
    stage: str,
    manifest_dir: Path,
    cfg,
    row_complete: Callable[[dict, object], bool],
    process_row: Callable[[dict, object], None],
    *,
    flush_every: int = 50,
    require: Callable[[dict, object], bool] | None = None,
    require_hint: str = "",
) -> dict[str, dict[str, int]]:
    """Run one preparation stage over the selected corpus's manifest, resumably and shard-aware.

    row_complete: takes (row, cfg) so it can recompute expected artifact paths from the
    current config. See `expected_completeness`.
    require: predicate an upstream stage must already satisfy for every row. When it fails,
    the stage refuses to start instead of dying part-way through.

    Only `data.source.name`'s manifest is touched. This once globbed every `*.jsonl` here, which
    was harmless while one corpus existed and destructive once two did: `data=pop2piano` walked
    Jamendo's 167k rows under pop2piano's config, where `provided_midi` points `midi_path` into a
    tree that holds nothing for a Jamendo track, so every row was reset to `pending`. Nothing was
    re-encoded and no latent was lost, but a corpus's bookkeeping was rewritten by a command that
    named a different corpus. A stage now touches exactly the corpus it was asked for.
    """
    name = str(cfg.data.source.name)
    manifest = manifest_dir / f"{name}.jsonl"
    if not manifest.is_file():
        raise RuntimeError(
            f"No manifest for corpus {name!r} at {manifest}; run the indexing stage first."
        )
    manifests = [manifest]

    if require is not None:
        incomplete = [m.name for m in manifests if not all(require(r, cfg) for r in read_jsonl(m))]
        if incomplete:
            raise RuntimeError(
                f"Upstream stage incomplete for {', '.join(incomplete)}. {require_hint}".strip()
            )

    shard, shards = shard_spec(cfg)
    counts: dict[str, dict[str, int]] = {}
    for manifest in manifests:
        rows = read_jsonl(manifest)
        indices = [i for i in range(len(rows)) if i % shards == shard]
        done = skipped = 0
        dirty: list[int] = []
        progress = tqdm(indices, desc=f"{stage}/{manifest.stem}[{shard}/{shards}]", unit="row")
        for index in progress:
            if row_complete(rows[index], cfg):
                skipped += 1
            else:
                process_row(rows[index], cfg)
                done += 1
                dirty.append(index)
                if len(dirty) >= flush_every:
                    flush_rows(manifest, rows, dirty)
                    dirty = []
            progress.set_postfix(done=done, skipped=skipped, refresh=False)
        progress.close()
        flush_rows(manifest, rows, dirty)
        counts[manifest.stem] = {
            "rows": len(rows), "shard_rows": len(indices),
            "done": done, "skipped": skipped, "shard": shard, "shards": shards,
        }
    return counts
