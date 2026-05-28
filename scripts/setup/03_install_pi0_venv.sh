#!/usr/bin/env bash
# π0 / pi0_libero venv via Physical-Intelligence's openpi. Their pyproject
# pulls JAX + PyTorch backends; we need both because attention extraction
# requires PyTorch (use_pytorch=True) while baseline inference uses JAX.
set -euo pipefail
source /root/.bashrc.emboviz

# openpi pins lerobot to an old git commit. uv clones it and runs
# `git reset --hard`, which triggers a git-lfs smudge of lerobot's
# tests/artifacts/*.safetensors fixtures — and at least one of those
# LFS blobs has been GC'd from the remote ("remote missing object"),
# so the smudge (and the whole build) fails. Those artifacts are test
# fixtures, NOT part of the built lerobot wheel, so skipping the smudge
# is safe: git-lfs leaves pointer files in place and the wheel builds
# from the (non-LFS) Python sources. See git-lfs docs on GIT_LFS_SKIP_SMUDGE.
export GIT_LFS_SKIP_SMUDGE=1

REPO=/root/repos/openpi
echo "[pi0] cloning Physical-Intelligence/openpi"
if [ ! -d "$REPO" ]; then
    git clone https://github.com/Physical-Intelligence/openpi.git "$REPO"
fi
cd "$REPO"

VENV="$REPO/.venv"
echo "[pi0] creating venv at $VENV"
uv venv "$VENV" --python 3.11

echo "[pi0] installing openpi (pulls JAX + PyTorch backends)"
uv pip install --python "$VENV/bin/python" -e .

echo "[pi0] pinning transformers (4.53.2 — required by openpi's gemma wrapper)"
uv pip install --python "$VENV/bin/python" "transformers==4.53.2"

# IMPORTANT: openpi pins lerobot to a specific git commit (the legacy
# lerobot.common.* import layout). We do NOT install a newer lerobot
# here because openpi's own modules ``import lerobot.common.datasets``
# at import time. Our LeRobotEpisodeSource adapter handles both old
# and new layouts, so the openpi-pinned version works fine.

echo "[pi0] runtime deps used by emboviz diagnostics + reports"
uv pip install --python "$VENV/bin/python" \
    "scipy>=1.11" "pyarrow" "rerun-sdk>=0.22" "jinja2>=3.1"

echo "[pi0] installing emboviz (editable)"
uv pip install --python "$VENV/bin/python" -e /root/emboviz/

echo "[pi0] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, jax, lerobot, emboviz
print('  torch       ', torch.__version__)
print('  transformers', transformers.__version__)
print('  jax         ', jax.__version__)
print('  lerobot     ', lerobot.__version__)
print('  emboviz     ', emboviz.__file__)
"
echo "[pi0] DONE — $VENV/bin/python ready"
echo "[pi0] Note: first inference triggers checkpoint download + Triton autotune (~5-10 min)"
