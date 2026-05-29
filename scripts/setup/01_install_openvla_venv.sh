#!/usr/bin/env bash
# OpenVLA adapter — dev pod recipe.
#
# This script runs EXACTLY what a user runs (per README):
#
#     uv venv ~/.venv-emboviz --python 3.11
#     uv pip install emboviz emboviz-lerobot emboviz-openvla
#     emboviz install-openvla
#
# The only difference vs PyPI users is the dev pod points pip at the
# local checkouts (-e <path>) instead of PyPI. Pyproject + each
# AdapterSpec.runtime_pip are the single source of truth.
#
# The main venv is SHARED across all adapter scripts (created once,
# guarded; each script ADDS its shims). The host install carries NO
# model or dataset libraries — only lightweight shims:
#   • emboviz-wire    — the ZMQ wire contracts
#   • emboviz         — core (no torch, no lerobot, modern rerun)
#   • emboviz-lerobot — the dataset-reader shim; its lerobot lives in the
#                       isolated reader venv (06_install_lerobot_venv.sh,
#                       or lazily materialised on first analyze)
#   • emboviz-<adapter> — this model's shim
# The heavy deps install into per-adapter / reader venvs via the
# ``emboviz install-<name>`` steps.
set -euo pipefail
source /root/.bashrc.emboviz

MAIN_VENV=/root/.venv-emboviz
ADAPTER=openvla

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
