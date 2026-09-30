#!/usr/bin/env python3
"""Pre-download the SheetSage assets PiCoGen needs, so compute nodes never reach the network.

Only the *Jukebox* branch is fetched. `mirtoolkit.sheetsage.infer` defaults to `use_jukebox=True`
and PiCoGen's `infer.py` does not override it, so the handcrafted (mel-spectrogram) models would be
~500 MB of weights nothing loads. Pass --handcrafted if that ever changes.

`retrieve_asset` verifies a checksum and is resumable in the sense that an already-correct file is
returned untouched, so re-running this after an interrupted 10 GB download is cheap and safe.
"""
from __future__ import annotations

import argparse
import sys

JUKEBOX = [
    "JUKEBOX_VQVAE",                        # ~1 GB
    "JUKEBOX_LM",                           # ~10 GB, the long pole
    "SHEETSAGE_V02_JUKEBOX_MELODY_CFG",
    "SHEETSAGE_V02_JUKEBOX_MELODY_STEP",
    "SHEETSAGE_V02_JUKEBOX_MELODY_MODEL",
    "SHEETSAGE_V02_JUKEBOX_HARMONY_CFG",
    "SHEETSAGE_V02_JUKEBOX_HARMONY_STEP",
    "SHEETSAGE_V02_JUKEBOX_HARMONY_MODEL",
]
HANDCRAFTED = [
    "SHEETSAGE_V02_HANDCRAFTED_MOMENTS",
    "SHEETSAGE_V02_HANDCRAFTED_MELODY_CFG",
    "SHEETSAGE_V02_HANDCRAFTED_MELODY_STEP",
    "SHEETSAGE_V02_HANDCRAFTED_MELODY_MODEL",
    "SHEETSAGE_V02_HANDCRAFTED_HARMONY_CFG",
    "SHEETSAGE_V02_HANDCRAFTED_HARMONY_STEP",
    "SHEETSAGE_V02_HANDCRAFTED_HARMONY_MODEL",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handcrafted", action="store_true",
                        help="also fetch the non-Jukebox models")
    args = parser.parse_args()

    from sheetsage import CACHE_DIR
    from sheetsage.assets import retrieve_asset

    tags = JUKEBOX + (HANDCRAFTED if args.handcrafted else [])
    print(f"[assets] cache dir: {CACHE_DIR}")
    failed: list[str] = []
    for position, tag in enumerate(tags, 1):
        print(f"[assets] {position}/{len(tags)} {tag}", flush=True)
        try:
            retrieve_asset(tag)
        except Exception as error:  # noqa: BLE001 - report every failure, not just the first
            print(f"[assets] FAILED {tag}: {type(error).__name__}: {error}", file=sys.stderr)
            failed.append(tag)
    if failed:
        raise SystemExit(f"[assets] {len(failed)} asset(s) failed: {', '.join(failed)}")
    print(f"[assets] all {len(tags)} present in {CACHE_DIR}")


if __name__ == "__main__":
    main()
