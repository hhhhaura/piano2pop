#!/usr/bin/env bash
# Bring the project up on a machine. Run from the project directory.
#
#   bash scripts/setup.sh
#   P2PA_CUDA=cu130 P2PA_DATA_ROOT=/big/disk/p2pa-data bash scripts/setup.sh
#
# Idempotent: an existing environment is reused and an already-unpacked data root is re-verified
# rather than re-extracted.
set -euo pipefail
cd "$(dirname "$0")/.."

# Which CUDA build to install. This is the one dependency that is a property of the *machine*
# rather than of the project, so it is a choice made here and nowhere else. Roughly: driver
# 525-555 -> cu124, 580+ -> cu130. Check the node, not the login host.
: "${P2PA_CUDA:=cu124}"
# Default to a directory beside the project. Everything generated lands here, so prefer a big disk.
: "${P2PA_DATA_ROOT:=$PWD/data-root}"
export P2PA_DATA_ROOT

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'MSG'
no `uv` on PATH. It is the only thing this project needs installed by hand:

    curl -LsSf https://astral.sh/uv/install.sh | sh
    exec $SHELL -l

It manages the Python interpreter too, so nothing else has to be provisioned first.
MSG
  exit 1
fi
echo "[setup] uv $(uv --version | awk '{print $2}'), cuda build $P2PA_CUDA"

echo "[setup] installing p2pa and its dependencies"
uv sync --extra dev --extra "$P2PA_CUDA"

# uv can retain a previously selected CUDA resolution in an existing environment. A stale cu130
# torch can install successfully, then fail much later when torchaudio loads its native extension
# against an older driver. Re-pin the cu124 pair explicitly after sync; this is idempotent and
# leaves cu130 installs untouched.
if [[ "$P2PA_CUDA" == "cu124" ]]; then
  uv pip install --python "$PWD/.venv/bin/python" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --force-reinstall \
    "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124"
fi

uv run --no-sync python - <<'CHECK'
import torch, torchaudio
print(f"[setup] torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"torchaudio {torchaudio.__version__}, cuda.is_available()={torch.cuda.is_available()}")
CHECK
echo "[setup] (False is expected on a machine without a GPU, such as a cluster login node)"

# The packs carry a `p2pa_pack.json` and, once extracted, `p2pa-unpack` re-checks every manifest
# row against what actually landed. A truncated transfer produces a tar that extracts cleanly and
# a corpus quietly missing a thousand latents, so read the line it prints.
if compgen -G 'dist/p2pa-*.tar' >/dev/null; then
  if [[ -f dist/SHA256SUMS ]]; then
    echo "[setup] verifying pack checksums"
    (cd dist && sha256sum -c SHA256SUMS)
  fi

  mkdir -p "$P2PA_DATA_ROOT"

  # Skip the extraction when the data root already verifies. Unpacking is tens of thousands of
  # small files onto a network filesystem — minutes, not seconds — and re-running this script to
  # pick up a fixed env.sh or a reinstalled package should not pay that again. `verify` re-checks
  # every manifest row against the disk, so "already there" means checked, not assumed.
  # Decided in Python and signalled by exit code, not by matching text. An *empty* root also
  # reports zero missing latents — because it has zero rows — so "nothing missing" alone is not
  # completeness; `usable_rows` has to be non-zero too. Matching on `"latent": 0` would have
  # skipped the unpack on a machine that had no data at all.
  if uv run --no-sync python - "$P2PA_DATA_ROOT" <<'CHECK'
import json, sys
from p2pa.pack import verify
report = json.loads(verify(sys.argv[1]))
missing = report["missing"]
complete = report["usable_rows"] > 0 and missing["latent"] == 0 and missing["midi"] == 0
print(json.dumps(report, sort_keys=True))
sys.exit(0 if complete else 1)
CHECK
  then
    echo "[setup] data root already complete, skipping unpack"
  else
    echo "[setup] unpacking into $P2PA_DATA_ROOT (several minutes; ~7.4 GB, ~50k files)"
    uv run --no-sync p2pa-unpack dist/p2pa-train.tar dist/p2pa-eval.tar --root "$P2PA_DATA_ROOT"
  fi
else
  echo "[setup] no packs in dist/ — leaving \$P2PA_DATA_ROOT ($P2PA_DATA_ROOT) as it is"
fi

# The model weights may not be cached yet (scripts/fetch_models.sh is a later, network-only step),
# so a failure here is expected on a first run and is reported rather than fatal.
echo "[setup] running the test suite (offline, no GPU)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_OFFLINE=1 uv run --no-sync pytest -q || echo "[setup] tests incomplete — re-run after scripts/fetch_models.sh"

# One file with the settings this run actually used, so an interactive shell and a batch job
# cannot disagree about them. That disagreement is not hypothetical: `fetch_models.sh` cached the
# model weights under its own HF_HOME, an interactive `pytest` had no HF_HOME, and thirteen tests
# failed looking in an empty ~/.cache — with an error about the network, on a machine that had
# already downloaded everything.
cat > env.sh <<ENV
# Written by scripts/setup.sh. Source before running anything interactively:
#     source env.sh
export P2PA_DATA_ROOT=$P2PA_DATA_ROOT
export P2PA_CUDA=$P2PA_CUDA
export HF_HOME=$HF_HOME
# \`uv run\` syncs the project environment first, with NO extras — and that is a third locked
# resolution, not a no-op. torch is not a base dependency, but seven installed packages require it
# (pytorch-lightning, accelerate, peft, torchmetrics, ...), so uv.lock carries torch three times:
# +cu124 when that extra is active, +cu130 when that one is, and a plain PyPI build when NEITHER
# is. A bare \`uv run\` selects the third, swaps the CUDA torch for the PyPI one — and leaves
# torchaudio alone, because torchaudio has no no-extra resolution. The pair stop sharing an ABI:
#     libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
# It surfaces at the next \`import torchaudio\`, in whatever command happens to run next rather
# than in the one that broke it. This is NOT cu124-specific; cu130 breaks identically.
# Exporting this makes every documented \`uv run\` safe. Re-run scripts/setup.sh to pick up a
# changed dependency — that is what it is for.
export UV_NO_SYNC=1
# Corpus building only (see README.md). Training and sampling need none of these; a script that
# does need one says which and stops.
#   P2PA_MIR_PYTHON        the interpreter with demucs 4.1.0 and muscriptor (scripts/setup_corpus_tools.sh)
export P2PA_MIR_PYTHON=${P2PA_MIR_PYTHON:-}
# The PiCoGen2 environment (corpus/picogen/setup.sh): BeatThis beats, PiCoGen covers, Kong's
# piano transcriber for the test set. PICOGEN_ROOT is the PiCoGen2 checkout (branch v2).
export PICOGEN_PYTHON=${PICOGEN_PYTHON:-}
export PICOGEN_ROOT=${PICOGEN_ROOT:-}
# Deliberately NOT setting HF_HUB_OFFLINE here: scripts/fetch_models.sh, run from a shell that
# sources this, exists to download. On machines without network, set HF_HUB_OFFLINE=1 there, which
# turns a silent network stall into an immediate, honest failure.
ENV
echo "[setup] wrote env.sh"

cat <<NEXT

[setup] ready. Source env.sh FIRST. It carries UV_NO_SYNC=1, without which the next
        \`uv run\` syncs with no extras, swaps this CUDA torch for a PyPI one and
        breaks torchaudio against it:

    source env.sh

Still to do, on a machine that has network:

    bash scripts/fetch_models.sh     # ACE-Step DiT + VAE + Qwen3 text encoder into HF_HOME

Then rehearse before committing a long run:

    run/dryrun.sh var
NEXT
