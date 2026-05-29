#!/usr/bin/env bash
# GR00T-N1.7 adapter — dev pod recipe.
#
# Mirrors the user-path exactly (per README):
#
#     uv pip install emboviz emboviz-lerobot emboviz-gr00t
#     emboviz install-gr00t        # handles --no-deps for gr00t package
#
# The ``emboviz install-gr00t`` command's second-pass ``--no-deps``
# install of NVIDIA's gr00t package sidesteps its broken flash-attn
# build dep (whose setup.py imports torch before pip has installed it,
# which fails under build isolation). The adapter falls back to SDPA
# at runtime so flash-attn is never invoked.
#
# We also clone the Isaac-GR00T repo for its ``demo_data/droid_sample``
# (3 sample episodes) — a USER who installed gr00t via pip and wants
# droid_sample does the same clone manually; documented in README.
set -euo pipefail
source /root/.bashrc.emboviz

# Pull the upstream repo for demo_data/droid_sample.
REPO=/root/repos/Isaac-GR00T
if [ ! -d "$REPO" ]; then
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$REPO"
fi
( cd "$REPO" && git lfs install --skip-smudge --local && git lfs pull )

MAIN_VENV=/root/.venv-emboviz
ADAPTER=gr00t

# Shared main venv: created once (guarded), each script adds its shims.
[ -d "$MAIN_VENV" ] || uv venv "$MAIN_VENV" --python 3.11
uv pip install --python "$MAIN_VENV/bin/python" \
    -e /root/emboviz/adapters/emboviz-wire \
    -e /root/emboviz \
    -e /root/emboviz/adapters/emboviz-lerobot \
    -e /root/emboviz/adapters/emboviz-$ADAPTER

EMBOVIZ_VENVS_DIR=/root/venvs "$MAIN_VENV/bin/emboviz" install-$ADAPTER --force

echo "[$ADAPTER] DONE"
echo "Start the worker:"
echo "    /root/venvs/$ADAPTER/bin/emboviz-$ADAPTER serve &"
