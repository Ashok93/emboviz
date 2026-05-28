#!/usr/bin/env bash
# SAM 3 sidecar — dev pod recipe.
#
# Same shape as the user-facing path documented in README:
#
#     uv venv .venv-sam3 --python 3.12
#     uv pip install emboviz-sam3
#     emboviz-sam3 serve --preload &
#
# The sidecar is its own Python 3.12 venv because:
#   - Official facebookresearch/sam3 needs Python 3.12+.
#   - HF's Sam3Model integration needs transformers >= 4.56.
#   - None of the four VLA adapter venvs (OpenVLA 3.10/4.49, OFT 3.10/
#     vendored fork, π0 3.11/4.53, GR00T 3.11/4.57) can host those
#     constraints alongside their pinned adapter deps.
# Isolating SAM 3 in its own process lets every adapter share the same
# default text→mask detector over HTTP.
#
# Per CLAUDE.md "Dev path is the user path": NO version pins here.
# ``sam3_service/pyproject.toml`` owns the SAM 3 runtime deps.
set -euo pipefail
source /root/.bashrc.emboviz

VENV=/root/venvs/sam3
uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" -e "/root/emboviz/sam3_service/"

echo "[sam3] sanity import"
"$VENV/bin/python" -c "
import torch, transformers
print('  torch       ', torch.__version__, '  cuda_avail=', torch.cuda.is_available())
print('  transformers', transformers.__version__)
from transformers import Sam3Model, Sam3Processor
print('  Sam3Model + Sam3Processor importable: OK')
import emboviz_sam3
print('  emboviz-sam3', emboviz_sam3.__version__)
"
echo "[sam3] DONE — start the server with:"
echo "    $VENV/bin/emboviz-sam3 serve --preload"
echo "[sam3] Note: first run downloads facebook/sam3 (~3.4 GB; gated, needs HF_TOKEN)"
