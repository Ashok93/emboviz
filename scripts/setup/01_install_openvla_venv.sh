#!/usr/bin/env bash
# OpenVLA-7B venv. Loads model lazily from HF (no local clone needed).
# Pins captured from a working pod 2026-05-26.
set -euo pipefail
source /root/.bashrc.emboviz

VENV=/root/venvs/openvla
echo "[openvla] creating venv at $VENV"
uv venv "$VENV" --python 3.10

echo "[openvla] installing deps"
uv pip install --python "$VENV/bin/python" \
    "torch==2.12.0" \
    "transformers==4.49.0" \
    "lerobot==0.3.2" \
    "accelerate==1.13.0" \
    "diffusers==0.37.1" \
    "huggingface-hub==0.36.2" \
    "tokenizers==0.21.4" \
    "safetensors==0.7.0" \
    "sentencepiece==0.2.1" \
    "timm==0.9.16" \
    "einops==0.8.2" \
    "av==17.0.1" \
    "torchcodec==0.13.0" \
    "pandas==2.3.3" \
    "numpy" "pillow" "matplotlib" "tqdm" \
    "scipy>=1.11" "pyarrow" "rerun-sdk>=0.22" "jinja2>=3.1"

# flash-attn needs --no-build-isolation; slow build (~10 min)
echo "[openvla] installing flash-attn (slow ~10 min build)"
uv pip install --python "$VENV/bin/python" flash-attn --no-build-isolation || \
    echo "[openvla] flash-attn build failed — OpenVLA falls back to sdpa attention. Acceptable."

# emboviz itself in editable mode so our adapter is importable
echo "[openvla] installing emboviz (editable)"
uv pip install --python "$VENV/bin/python" -e /root/emboviz/

echo "[openvla] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, lerobot, emboviz
print('  torch       ', torch.__version__)
print('  transformers', transformers.__version__)
print('  lerobot     ', lerobot.__version__)
print('  emboviz     ', emboviz.__file__)
"
echo "[openvla] DONE — $VENV/bin/python ready"
