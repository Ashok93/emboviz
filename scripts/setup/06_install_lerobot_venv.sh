#!/usr/bin/env bash
# LeRobot dataset reader — dev pod recipe.
#
# Mirrors the user-path exactly (per README):
#
#     uv pip install emboviz emboviz-lerobot
#     emboviz install-lerobot
#
# Builds the ISOLATED reader venv (/root/venvs/lerobot via
# EMBOVIZ_VENVS_DIR) holding the LATEST lerobot (0.5.x) — codebase v3.0,
# the current official format. (Older LeRobot v2.1 + modality.json GR00T
# datasets are read by the separate reader-gr00t venv; see script 07.)
# Core never installs lerobot: its transitive torch / video-decode stack
# stays out of the host. The reader talks to the host over the SAME ZMQ
# wire as a model worker; ``emboviz analyze`` spawns it automatically for
# any ``dataset.format: lerobot`` config (this script just pre-warms the
# venv so the first run doesn't pay the install).
set -euo pipefail
source /root/.bashrc.emboviz

MAIN_VENV=/root/.venv-emboviz

# Shared main venv: created once (guarded). Install the lightweight host
# shims (no torch, no lerobot in the host).
[ -d "$MAIN_VENV" ] || uv venv "$MAIN_VENV" --python 3.11
uv pip install --python "$MAIN_VENV/bin/python" \
    -e /root/emboviz/adapters/emboviz-wire \
    -e /root/emboviz \
    -e /root/emboviz/adapters/emboviz-lerobot

EMBOVIZ_VENVS_DIR=/root/venvs "$MAIN_VENV/bin/emboviz" install-lerobot --force

echo "[lerobot] DONE"
echo "Reader venv ready: /root/venvs/lerobot (lerobot 0.5.x, codebase v3.0)"
echo "Used automatically when a run config has dataset.format: lerobot"
