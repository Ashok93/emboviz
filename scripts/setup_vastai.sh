#!/usr/bin/env bash
# One-shot Vast.ai env setup for PolicyLens.
# Idempotent: safe to re-run. Targets Linux + (optional) CUDA.
#
# Usage (from repo root):
#   bash scripts/setup_vastai.sh
#
# After this completes, run:
#   uv run python scripts/run_first_test.py --episode 0

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] PolicyLens — Vast.ai env bootstrap"
echo "[setup] repo: $REPO_ROOT"

# --- 1. Install uv (fast Python package manager) ------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer puts the binary in ~/.local/bin; make sure it's on PATH for this shell
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[setup] uv already installed: $(uv --version)"
fi

# --- 2. Configure HuggingFace + model caches inside repo ---------------------
# Vast.ai instance disks are ephemeral; we still keep caches in-repo so a single
# `rm -rf hf_cache` cleans everything, and a Vast.ai persistent volume can be
# mounted at this path to survive restarts.
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"
mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"

# --- 3. Sync Python deps ------------------------------------------------------
echo "[setup] syncing dependencies via uv (this is the slow step on first run)..."
uv sync

# --- 4. Verify torch + GPU ---------------------------------------------------
echo "[setup] verifying torch + GPU..."
uv run python - <<'PY'
import torch
print(f"  torch       : {torch.__version__}")
print(f"  cuda avail. : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda device : {torch.cuda.get_device_name(0)}")
    print(f"  cuda mem    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  (no CUDA — will run on CPU; this is fine for Diffusion Policy on PushT, just slower)")
PY

echo "[setup] done."
echo ""
echo "Next: uv run python scripts/run_first_test.py --episode 0"
