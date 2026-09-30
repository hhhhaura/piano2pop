#!/usr/bin/env bash
# Cache the Hugging Face models training and sampling need: the ACE-Step v1.5 DiT, its VAE and its
# Qwen3 text encoder. Run once where there is network; afterwards everything runs with
# HF_HUB_OFFLINE=1, e.g. on offline cluster nodes that share HF_HOME.
#
#   bash scripts/fetch_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f env.sh ]] && source env.sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoConfig
print(snapshot_download("ACE-Step/acestep-v15-base"))
print(snapshot_download("ACE-Step/Ace-Step1.5",
                        allow_patterns=["vae/*", "Qwen3-Embedding-0.6B/*"]))
# snapshot_download stores repository files, but trust_remote_code also needs Transformers'
# dynamic-module cache. Warm it while there is network, or offline runs fail at model load.
AutoConfig.from_pretrained("ACE-Step/acestep-v15-base", trust_remote_code=True)
print("ACE-Step remote modeling code cached")
PY
echo "[p2pa] weights cached under $HF_HOME"
