#!/usr/bin/env bash
# OpenVLA adapter — dev pod recipe.
#
# This script does EXACTLY what a user does, per README:
#
#     uv venv .venv-openvla --python 3.10
#     uv pip install 'emboviz[openvla]'
#
# The only difference is the dev pod points pip at the local checkout
# of emboviz via the absolute path, while a user installs from PyPI.
#
# Per CLAUDE.md "Dev path is the user path": NO version pins in here,
# NO CUDA-detection cleverness, NO --index-url juggling. Pyproject's
# ``[openvla]`` extra is the only source of truth for adapter deps.
set -euo pipefail
source /root/.bashrc.emboviz

VENV=/root/venvs/openvla
uv venv "$VENV" --python 3.10
uv pip install --python "$VENV/bin/python" -e "/root/emboviz[openvla]"

echo "[openvla] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, lerobot, emboviz
print('  torch       ', torch.__version__, '  cuda_avail=', torch.cuda.is_available())
print('  transformers', transformers.__version__)
print('  lerobot     ', lerobot.__version__)
print('  emboviz     ', emboviz.__file__)
"
echo "[openvla] DONE — $VENV/bin/python ready"
