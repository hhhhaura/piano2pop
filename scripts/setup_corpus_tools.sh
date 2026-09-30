#!/usr/bin/env bash
# Build the quarantined Python 3.10 MIR environment with uv and prefetch its model weights.
# Muscriptor/Demucs are intentionally not project dependencies: their NumPy/Torch constraints do
# not belong in p2pa's training environment. The batch jobs always call MIR_PYTHON absolutely.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${P2PA_DATA_ROOT:=$PWD/data-root}"
: "${P2PA_MIR_ENV:=$P2PA_DATA_ROOT/.cache/mir_venv}"
: "${P2PA_MIR_TORCH_BACKEND:=cu124}"
: "${HF_HOME:=$HOME/.cache/huggingface}"
: "${TORCH_HOME:=$HOME/.cache/torch}"
MIR_PYTHON="$P2PA_MIR_ENV/bin/python"

for command in uv ffmpeg ffprobe; do
  command -v "$command" >/dev/null 2>&1 || { echo "missing required command: $command" >&2; exit 1; }
done

# `MuScriptor/muscriptor-medium` is a gated repo. Unauthenticated, its fetch fails with a 401
# only *after* the venv has been built and demucs cached, so the credential check comes first.
if [[ -z "${HF_TOKEN:-}" && ! -s "$HF_HOME/token" ]]; then
  cat >&2 <<EOF
missing Hugging Face credentials: MuScriptor/muscriptor-medium is gated.
Export a token that has access to it, then re-run:
  export HF_TOKEN=hf_...
or log in once, which writes $HF_HOME/token:
  HF_HOME=$HF_HOME uvx --from huggingface_hub hf auth login
EOF
  exit 1
fi
if [[ -n "${HF_TOKEN:-}" ]]; then export HF_TOKEN; fi

mkdir -p "$(dirname "$P2PA_MIR_ENV")" "$HF_HOME" "$TORCH_HOME"
if [[ ! -x "$MIR_PYTHON" ]]; then
  uv venv --python 3.10 "$P2PA_MIR_ENV"
fi
uv pip install --python "$MIR_PYTHON" --torch-backend "$P2PA_MIR_TORCH_BACKEND" \
  "muscriptor==0.2.1" "demucs==4.1.0"

HF_HOME="$HF_HOME" TORCH_HOME="$TORCH_HOME" "$MIR_PYTHON" - <<'PY'
from demucs.pretrained import get_model
from muscriptor.transcription_model import TranscriptionModel

print("[corpus-tools] caching htdemucs_6s")
get_model("htdemucs_6s")
print("[corpus-tools] caching muscriptor medium")
TranscriptionModel.load_model(weights_path="medium", device="cpu")
print("[corpus-tools] both model families load offline")
PY

cat <<EOF
[corpus-tools] ready
export P2PA_MIR_PYTHON=$MIR_PYTHON
export HF_HOME=$HF_HOME
export TORCH_HOME=$TORCH_HOME
EOF
