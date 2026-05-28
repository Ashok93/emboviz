#!/usr/bin/env bash
# π0 / π0.5 adapter — dev pod recipe.
#
# Mirrors the user-path exactly (per README):
#
#     uv pip install emboviz emboviz-pi0
#     emboviz install-pi0       # handles GIT_LFS_SKIP_SMUDGE=1 transparently
set -euo pipefail
source /root/.bashrc.emboviz

MAIN_VENV=/root/.venv-emboviz
ADAPTER=pi0

uv venv "$MAIN_VENV" --python 3.11
uv pip install --python "$MAIN_VENV/bin/python" \
    -e /root/emboviz \
    -e /root/emboviz/adapters/emboviz-$ADAPTER

EMBOVIZ_VENVS_DIR=/root/venvs "$MAIN_VENV/bin/emboviz" install-$ADAPTER --force

echo "[$ADAPTER] DONE"
echo "Start the worker:"
echo "    /root/venvs/$ADAPTER/bin/emboviz-$ADAPTER serve &"
echo "Note: first inference triggers checkpoint download + Triton autotune (~5-10 min)"
