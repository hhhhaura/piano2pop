"""Beat/downbeat detection worker for the corpus-wide beats cache.

Runs under the PiCoGen environment (`corpus/picogen/setup.sh`), which has beat_this installed;
`corpus/picogen/run.py beats` dispatches it. Reads a JSONL job list, writes one newline-delimited
JSON result per line to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import torch
    from beat_this.inference import File2Beats

    device = (
        torch.device("cuda")
        if args.device.lower() != "cpu" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    detector = File2Beats(checkpoint_path="final0", device=device, float16=False, dbn=True)
    print(json.dumps({"event": "ready", "device": str(device)}), flush=True)

    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    for job in jobs:
        try:
            beats, downbeats = detector(Path(job["audio"]))
            result = {
                "track_id": job["track_id"], "shard": job["shard"], "ok": True,
                "beats": [float(value) for value in beats],
                "downbeats": [float(value) for value in downbeats],
            }
        except Exception as error:  # noqa: BLE001 - one bad track must not stop the corpus shard
            result = {
                "track_id": job["track_id"], "shard": job["shard"], "ok": False,
                "reason": f"{type(error).__name__}: {error}"[:300],
            }
        print(json.dumps({"event": "result", **result}), flush=True)


if __name__ == "__main__":
    main()
