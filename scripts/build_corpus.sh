#!/usr/bin/env bash
# Build the training corpus from the audio you placed in $P2PA_DATA_ROOT/data/raw: every stage, in
# order, one shard per GPU. Every stage is resumable, so re-running picks up where it stopped.
#
#   GPUS=0,1,2,3 bash scripts/build_corpus.sh
#   PICOGEN=0 bash scripts/build_corpus.sh     # skip PiCoGen covers (only `pico`/`final` use them)
#
# Needs env.sh (scripts/setup.sh), P2PA_MIR_PYTHON (scripts/setup_corpus_tools.sh) and
# PICOGEN_PYTHON (corpus/picogen/setup.sh; BeatThis lives there too).
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
: "${GPUS:=0}"
: "${PICOGEN:=1}"
: "${P2PA_MIR_PYTHON:?run scripts/setup_corpus_tools.sh and add P2PA_MIR_PYTHON to env.sh}"
: "${PICOGEN_PYTHON:?run corpus/picogen/setup.sh and add PICOGEN_PYTHON to env.sh}"
IFS=, read -r -a GPU <<< "$GPUS"
N=${#GPU[@]}
DATA="$P2PA_DATA_ROOT"
INVENTORY="$DATA/.cache/manifests/p2pdata.inventory.jsonl"

# `sharded <command...>`: one copy per GPU, each with --shard i --num-shards N, then wait for all.
sharded() {
  local pids=()
  for i in "${!GPU[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU[$i]}" "$@" --shard "$i" --num-shards "$N" &
    pids+=($!)
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

echo "[corpus] 1/6 index the downloaded audio"
uv run --no-sync p2pa-corpus-index --raw "$DATA/data/raw" --output "$INVENTORY"

for stage in separate transcribe; do
  echo "[corpus] $([[ $stage == separate ]] && echo 2 || echo 3)/6 $stage"
  sharded uv run --no-sync p2pa-corpus-process "$stage" \
    --inventory "$INVENTORY" --source-root "$DATA/data/raw" --output-root "$DATA/data" \
    --cache-root "$DATA/.cache/corpus-process" --mir-python "$P2PA_MIR_PYTHON"
done

echo "[corpus] 4/6 beats (BeatThis on each instrumental)"
sharded python3 corpus/picogen/run.py beats --device 0

if [[ "$PICOGEN" == 1 ]]; then
  echo "[corpus] 5/6 PiCoGen2 piano covers"
  sharded python3 corpus/picogen/run.py picogen --device 0
else
  echo "[corpus] 5/6 PiCoGen2 skipped (PICOGEN=0)"
fi

echo "[corpus] 6/6 training cache: manifest, outlier screen, VAE latents, fixed prompt"
CUDA_VISIBLE_DEVICES="${GPU[0]}" uv run --no-sync p2pa-prep
CUDA_VISIBLE_DEVICES="${GPU[0]}" uv run --no-sync p2pa-prep-text
echo "[corpus] done"
