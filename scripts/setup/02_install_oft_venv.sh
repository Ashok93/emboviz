#!/usr/bin/env bash
# OpenVLA-OFT adapter — dev pod recipe.
#
# Mirrors the user-path exactly (per README):
#
#     uv pip install emboviz emboviz-oft
#     emboviz install-oft
set -euo pipefail
source /root/.bashrc.emboviz

MAIN_VENV=/root/.venv-emboviz
ADAPTER=oft

uv venv "$MAIN_VENV" --python 3.11
uv pip install --python "$MAIN_VENV/bin/python" \
    -e /root/emboviz \
    -e /root/emboviz/adapters/emboviz-$ADAPTER

EMBOVIZ_VENVS_DIR=/root/venvs "$MAIN_VENV/bin/emboviz" install-$ADAPTER --force

echo "[$ADAPTER] DONE"
echo "Start the worker:"
echo "    /root/venvs/$ADAPTER/bin/emboviz-$ADAPTER serve &"
