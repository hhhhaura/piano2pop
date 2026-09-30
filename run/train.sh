#!/usr/bin/env bash
# Train one setting on this workstation: a full finetune of ACE-Step v1.5 for 500k steps.
#
#   run/train.sh var                      # runs/var, GPU 0
#   P2PA_DEVICE=3 run/train.sh pico       # on GPU 3
#   run/train.sh rule exp_name=rule_seed1 seed=1
#   run/train.sh pico backbone=scratch    # the from-scratch DiT instead; runs/scratch_pico
#
# Resumable: re-running the same exp_name continues from its own last.ckpt.
set -euo pipefail
cd "$(dirname "$0")/.."
SETTING="${1:?usage: run/train.sh <base|rule|var|pico|final> [hydra overrides...]}"
shift || true
[[ -f configs/setting/$SETTING.yaml ]] || {
  echo "run/train.sh: no configs/setting/$SETTING.yaml" >&2
  exit 2
}
[[ -f env.sh ]] && source env.sh
NAME="$SETTING"
for arg in "$@"; do
  [[ "$arg" == backbone=scratch ]] && NAME="scratch_$SETTING"
done
exec uv run --no-sync p2pa-train setting="$SETTING" exp_name="$NAME" \
  "trainer.devices=[${P2PA_DEVICE:-0}]" "$@"
