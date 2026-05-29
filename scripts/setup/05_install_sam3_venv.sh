#!/usr/bin/env bash
# SAM 3 adapter — dev pod recipe.
#
# Mirrors the user-path exactly (per README):
#
#     uv pip install emboviz emboviz-lerobot emboviz-sam3
#     emboviz install-sam3
#
# The SAM 3 runtime venv is pinned to Python 3.12 (sam3 reference repo
# requirement) and transformers >= 4.56 (added the ``Sam3Model``
# integration). None of the four VLA adapter venvs can host those
# constraints alongside their pinned adapter deps; ZMQ's bytes wire
# is Python-version-agnostic so SAM 3 stays on 3.12 forever.
set -euo pipefail
source /root/.bashrc.emboviz

MAIN_VENV=/root/.venv-emboviz
ADAPTER=sam3

# Shared main venv: created once (guarded), each script adds its shims.
[ -d "$MAIN_VENV" ] || uv venv "$MAIN_VENV" --python 3.11
uv pip install --python "$MAIN_VENV/bin/python" \
    -e /root/emboviz/adapters/emboviz-wire \
    -e /root/emboviz \
    -e /root/emboviz/adapters/emboviz-lerobot \
    -e /root/emboviz/adapters/emboviz-$ADAPTER

EMBOVIZ_VENVS_DIR=/root/venvs "$MAIN_VENV/bin/emboviz" install-$ADAPTER --force

echo "[$ADAPTER] DONE"
echo "Start the worker (preloads SAM 3 — ~30 s first run):"
echo "    /root/venvs/$ADAPTER/bin/emboviz-$ADAPTER serve &"
echo "Note: first run downloads facebook/sam3 (~3.4 GB; gated, needs HF_TOKEN)"
