#!/usr/bin/env bash
# The PiCoGen2 environment: BeatThis, SheetSage and PiCoGen2, in their own Python 3.11 venv.
# Needed for `corpus/picogen/run.py` (beats for every setting; PiCoGen covers for `pico`/`final`).
#
#   bash corpus/picogen/setup.sh
#
# PiCoGen2 itself is cloned from https://github.com/tanchihpin0517/PiCoGen (branch v2) and put on the
# worker's path rather than installed. Idempotent. Every unusual line is load-bearing:
#   python 3.11           what the paper's corpus was built with; PiCoGen2's own pyproject says
#                         <3.11, which is why it is imported from its checkout, not installed.
#   mpi4py-mpich          SheetSage refuses to import without mpi4py; the -mpich wheel vendors MPI.
#   --no-build-isolation  the mirtoolkit SheetSage fork pulls an old Jukebox setup.py that imports
#                         pkg_resources without declaring setuptools as a build dependency.
#   transformers/huggingface_hub 4.41.2 / 0.23.4   newer pairs break the fork's imports.
set -euo pipefail
cd "$(dirname "$0")"

: "${P2PA_DATA_ROOT:?source env.sh first}"
: "${PICOGEN_ROOT:=$P2PA_DATA_ROOT/picogen}"
: "${PICOGEN_SRC:=$PICOGEN_ROOT/PiCoGen}"
: "${PICOGEN_ENV:=$PICOGEN_ROOT/venv}"
: "${SHEETSAGE_CACHE_DIR:=$PICOGEN_ROOT/cache/sheetsage}"
: "${TORCH_HOME:=$PICOGEN_ROOT/cache/torch}"
PY="$PICOGEN_ENV/bin/python"

command -v uv >/dev/null 2>&1 || { echo "missing uv" >&2; exit 1; }
command -v wget >/dev/null 2>&1 || { echo "missing wget (picogen2/assets.py shells out to it)" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "missing git" >&2; exit 1; }
mkdir -p "$SHEETSAGE_CACHE_DIR" "$TORCH_HOME/hub/checkpoints"
if [[ ! -d "$PICOGEN_SRC/picogen2" ]]; then
  echo "[picogen] cloning PiCoGen2 into $PICOGEN_SRC"
  git clone --depth 1 -b v2 https://github.com/tanchihpin0517/PiCoGen.git "$PICOGEN_SRC"
fi

if [[ ! -x "$PY" ]]; then
  echo "[picogen] creating $PICOGEN_ENV (python 3.11)"
  uv venv --python 3.11 "$PICOGEN_ENV"
fi
echo "[picogen] build toolchain"
uv pip install --python "$PY" 'setuptools==80.9.0' wheel cython numpy
uv pip uninstall --python "$PY" mpi4py 2>/dev/null || true
echo "[picogen] PiCoGen stack (the slow one)"
uv pip install --python "$PY" --no-build-isolation \
  torch torchvision torchaudio \
  'transformers==4.41.2' 'huggingface_hub==0.23.4' \
  sentencepiece protobuf omegaconf joblib questionary termcolor ipdb miditoolkit mpi4py-mpich \
  piano_transcription_inference \
  'mirtoolkit @ git+https://github.com/tanchihpin0517/mirtoolkit.git@v0.1.0'

echo "[picogen] SheetSage + Jukebox weights (~10 GB)"
SHEETSAGE_CACHE_DIR="$SHEETSAGE_CACHE_DIR" TORCH_HOME="$TORCH_HOME" "$PY" fetch_assets.py

echo "[picogen] verifying"
PYTHONPATH="$PICOGEN_SRC" SHEETSAGE_CACHE_DIR="$SHEETSAGE_CACHE_DIR" TORCH_HOME="$TORCH_HOME" "$PY" -c \
  'import miditoolkit, piano_transcription_inference as p, torch;
p.load_audio_stream = getattr(p, "load_audio_stream", lambda *_a, **_k: None);
from mirtoolkit import beat_this, sheetsage;
from picogen2.model import PiCoGenDecoder;
print("[picogen] ready:", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())'

cat <<DONE

[picogen] done. Add to env.sh:

  export PICOGEN_PYTHON=$PY
  export PICOGEN_ROOT=$PICOGEN_SRC
  export SHEETSAGE_CACHE_DIR=$SHEETSAGE_CACHE_DIR
  export TORCH_HOME=$TORCH_HOME

then run, once per GPU (i = 0..n-1):

  python corpus/picogen/run.py beats   --shard i --num-shards n --device i
  python corpus/picogen/run.py picogen --shard i --num-shards n --device i
DONE
