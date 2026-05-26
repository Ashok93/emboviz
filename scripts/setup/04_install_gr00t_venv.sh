#!/usr/bin/env bash
# GR00T-N1.7 venv via NVIDIA's Isaac-GR00T. The cloned repo includes
# demo_data/droid_sample (3 demo episodes). Pin transformers==4.57.3 —
# newer versions broke GroundingDINO API; older versions break Qwen3-VL.
set -euo pipefail
source /root/.bashrc.emboviz

REPO=/root/repos/Isaac-GR00T
echo "[gr00t] cloning NVIDIA/Isaac-GR00T (~1 GB with demo data)"
if [ ! -d "$REPO" ]; then
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$REPO"
fi

VENV=/root/venvs/gr00t
echo "[gr00t] creating venv at $VENV (Python 3.11 — gr00t pyproject requires it)"
uv venv "$VENV" --python 3.11

cd "$REPO"
echo "[gr00t] installing gr00t package"
uv pip install --python "$VENV/bin/python" -e .

echo "[gr00t] pinning transformers (4.57.3)"
uv pip install --python "$VENV/bin/python" "transformers==4.57.3"

echo "[gr00t] runtime deps not pulled by gr00t pyproject"
uv pip install --python "$VENV/bin/python" pandas av decord torchcodec albumentations peft

echo "[gr00t] installing emboviz (editable)"
uv pip install --python "$VENV/bin/python" -e /root/emboviz/

echo "[gr00t] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, gr00t, emboviz
print('  torch       ', torch.__version__)
print('  transformers', transformers.__version__)
print('  gr00t       ', gr00t.__file__)
print('  emboviz     ', emboviz.__file__)
"
echo "[gr00t] DONE — $VENV/bin/python ready"
echo "[gr00t] Note: first inference downloads nvidia/GR00T-N1.7-3B (~6 GB) + nvidia/Cosmos-Reason2-2B (gated, needs HF_TOKEN)"
