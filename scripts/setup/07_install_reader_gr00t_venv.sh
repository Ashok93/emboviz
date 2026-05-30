#!/usr/bin/env bash
# GR00T-format dataset reader — dev pod recipe.
#
# Mirrors the user-path exactly (per the reader's README):
#
#     uv pip install emboviz emboviz-reader-gr00t
#     emboviz install-reader-gr00t
#
# Builds the ISOLATED reader venv (/root/venvs/reader-gr00t via
# EMBOVIZ_VENVS_DIR) holding lerobot 0.3.x — codebase v2.1. GR00T datasets
# are LeRobot v2.1 + meta/modality.json; lerobot >=0.4 cannot read v2.x, so
# this reader pins the last v2.1-capable lerobot, separate from the v3.0
# ``lerobot`` reader's venv. The reader talks to the host over the SAME ZMQ
# wire as a model worker; ``emboviz analyze`` spawns it automatically for
# any ``dataset.format: gr00t`` config (this script just pre-warms the venv
# so the first run doesn't pay the install).
set -euo pipefail
source /root/.bashrc.emboviz

MAIN_VENV=/root/.venv-emboviz

# Shared main venv: created once (guarded). Install the lightweight host
# shims (no torch, no lerobot in the host).
[ -d "$MAIN_VENV" ] || uv venv "$MAIN_VENV" --python 3.11
uv pip install --python "$MAIN_VENV/bin/python" \
    -e /root/emboviz/adapters/emboviz-wire \
    -e /root/emboviz \
    -e /root/emboviz/adapters/emboviz-reader-gr00t

EMBOVIZ_VENVS_DIR=/root/venvs "$MAIN_VENV/bin/emboviz" install-reader-gr00t --force

echo "[reader-gr00t] DONE"
echo "Reader venv ready: /root/venvs/reader-gr00t (lerobot 0.3.x, codebase v2.1)"
echo "Used automatically when a run config has dataset.format: gr00t"
