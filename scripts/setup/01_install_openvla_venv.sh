#!/usr/bin/env bash
# OpenVLA adapter — dev pod recipe.
#
# This script runs EXACTLY what a user runs (per README):
#
#     uv venv ~/.venv-emboviz --python 3.11
#     uv pip install emboviz emboviz-openvla
#     emboviz install-openvla
#
# The only difference vs PyPI users is the dev pod points pip at the
# local checkouts (-e <path>) instead of PyPI. Pyproject + the
# adapter's AdapterSpec.runtime_pip are the single source of truth.
set -euo pipefail
source /root/.bashrc.emboviz

MAIN_VENV=/root/.venv-emboviz
ADAPTER=openvla

uv venv "$MAIN_VENV" --python 3.11
uv pip install --python "$MAIN_VENV/bin/python" \
    -e /root/emboviz \
    -e /root/emboviz/adapters/emboviz-$ADAPTER

EMBOVIZ_VENVS_DIR=/root/venvs "$MAIN_VENV/bin/emboviz" install-$ADAPTER --force

echo "[$ADAPTER] DONE"
echo "Start the worker:"
echo "    /root/venvs/$ADAPTER/bin/emboviz-$ADAPTER serve &"
