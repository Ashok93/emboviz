#!/usr/bin/env bash
# π0 / pi0_libero venv via Physical-Intelligence's openpi. Their pyproject
# pulls JAX + PyTorch backends; we need both because attention extraction
# requires PyTorch (use_pytorch=True) while baseline inference uses JAX.
set -euo pipefail
source /root/.bashrc.emboviz

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

echo "[pi0] adding lerobot for pi0_libero dataset"
uv pip install --python "$VENV/bin/python" "lerobot==0.3.2"

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
