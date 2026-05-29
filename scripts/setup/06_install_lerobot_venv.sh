#!/usr/bin/env bash
# LeRobot dataset reader — dev pod recipe.
#
# Mirrors the user-path exactly (per README):
#
#     uv pip install emboviz emboviz-lerobot
#     emboviz install-lerobot
#
# Builds the ISOLATED reader venv (/root/venvs/lerobot via
# EMBOVIZ_VENVS_DIR) holding lerobot 0.3.x — codebase v2.1, which reads
# LeRobot v2.0 AND v2.1 datasets. Core never installs lerobot: its
# transitive ``rerun-sdk<0.27`` pin would collide with core's own
# ``rerun>=0.32`` .rrd exporter. The reader talks to the host over the
# SAME ZMQ wire as a model worker; ``emboviz analyze`` spawns it
# automatically for any ``dataset.format: lerobot`` config (this script
# just pre-warms the venv so the first run doesn't pay the install).
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
echo "Reader venv ready: /root/venvs/lerobot (lerobot 0.3.x, codebase v2.1)"
echo "Used automatically when a run config has dataset.format: lerobot"
