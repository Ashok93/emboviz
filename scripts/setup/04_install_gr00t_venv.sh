#!/usr/bin/env bash
# GR00T-N1.7 venv via NVIDIA's Isaac-GR00T. The cloned repo includes
# demo_data/droid_sample (3 demo episodes). Pin transformers==4.57.3 —
# newer versions broke GroundingDINO API; older versions break Qwen3-VL.
#
# Why ``--no-deps`` for the gr00t package itself:
#   gr00t's pyproject lists flash-attn as a required dep. flash-attn's
#   build setup imports torch BEFORE pip has installed it (build isolation
#   ships an empty env), so the build fails with ModuleNotFoundError.
#   We don't actually USE flash-attn — the emboviz gr00t adapter forces
#   eager attention for the extraction path and SDPA otherwise — so we
#   install gr00t without its deps and bring the runtime deps we need
#   in explicitly below.
set -euo pipefail
source /root/.bashrc.emboviz

REPO=/root/repos/Isaac-GR00T
echo "[gr00t] cloning NVIDIA/Isaac-GR00T + pulling git-lfs blobs"
if [ ! -d "$REPO" ]; then
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$REPO"
fi
# demo_data/droid_sample ships as git-lfs pointer files. Without this
# pull they are 131-byte stubs and the dataset adapter fails to read
# the parquets. Bootstrap installed git-lfs system-wide.
( cd "$REPO" && git lfs install --skip-smudge --local && git lfs pull )

VENV=/root/venvs/gr00t
echo "[gr00t] creating venv at $VENV (Python 3.11 — gr00t pyproject requires it)"
uv venv "$VENV" --python 3.11

cd "$REPO"

echo "[gr00t] installing torch + transformers pin FIRST so subsequent"
echo "        editable installs that import torch at build time succeed"
uv pip install --python "$VENV/bin/python" \
    "torch==2.12.0" \
    "transformers==4.57.3"

echo "[gr00t] installing gr00t package (--no-deps to skip flash-attn build)"
uv pip install --python "$VENV/bin/python" --no-deps -e .

echo "[gr00t] runtime deps that gr00t code actually uses at run time"
uv pip install --python "$VENV/bin/python" \
    accelerate peft pandas av decord torchcodec albumentations \
    diffusers einops huggingface-hub safetensors tokenizers \
    sentencepiece tqdm pillow numpy scipy timm dm-tree tyro \
    lmdb msgpack msgpack-numpy termcolor omegaconf jsonlines \
    gymnasium kornia opencv-python-headless

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
