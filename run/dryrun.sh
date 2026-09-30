#!/usr/bin/env bash
# Rehearse a setting for 20 steps and print peak VRAM, peak host RSS and ms/step. Nothing is
# written to disk. Do this before committing a GPU for 500k steps.
#
#   run/dryrun.sh var
#   P2PA_DEVICE=3 run/dryrun.sh base data.batch_size=2
#   run/dryrun.sh pico backbone=scratch
set -euo pipefail
cd "$(dirname "$0")/.."
SETTING="${1:?usage: run/dryrun.sh <setting> [overrides...]}"
shift || true
# Benchmark the steady-state trainable set, not the cheap initial freeze window: frozen, the run
# trains 17.5M parameters; unfrozen it carries float32 masters, gradients and two AdamW moments
# for 2.4B, which is the regime of the other 498,000 steps.
exec run/train.sh "$SETTING" exp_name="dryrun_$SETTING" trainer.dry_run_steps=20 \
  finetune.freeze_steps=0 "$@"
