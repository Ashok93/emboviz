#!/usr/bin/env bash
# OpenVLA-OFT venv. Uses moojink's fork of OpenVLA with their pinned
# vendored transformers (incompatible with mainline). lerobot is added
# separately because moojink's pyproject doesn't depend on it.
set -euo pipefail
source /root/.bashrc.emboviz

REPO=/root/repos/openvla-oft
echo "[oft] cloning moojink/openvla-oft"
if [ ! -d "$REPO" ]; then
    git clone https://github.com/moojink/openvla-oft.git "$REPO"
fi
cd "$REPO"

VENV="$REPO/.venv"
echo "[oft] creating venv at $VENV"
uv venv "$VENV" --python 3.10

echo "[oft] installing moojink's openvla-oft (pulls their pinned transformers + prismatic)"
uv pip install --python "$VENV/bin/python" -e .

echo "[oft] adding lerobot for dataset loading (LIBERO-spatial)"
uv pip install --python "$VENV/bin/python" "lerobot==0.3.3"

echo "[oft] runtime deps used by emboviz diagnostics + reports"
uv pip install --python "$VENV/bin/python" \
    "scipy>=1.11" "pyarrow" "rerun-sdk>=0.22" "jinja2>=3.1"

echo "[oft] installing emboviz (editable)"
uv pip install --python "$VENV/bin/python" -e /root/emboviz/

echo "[oft] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, lerobot, emboviz
print('  torch       ', torch.__version__)
print('  transformers', transformers.__version__)
print('  lerobot     ', lerobot.__version__)
print('  emboviz     ', emboviz.__file__)
"
echo "[oft] DONE — $VENV/bin/python ready"
