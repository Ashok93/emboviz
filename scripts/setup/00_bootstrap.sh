#!/usr/bin/env bash
# Bootstrap a fresh GPU pod: install uv, clone emboviz, set up env.
# Idempotent — safe to re-run.
set -euo pipefail

echo "[bootstrap] installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi
uv --version

echo "[bootstrap] cloning emboviz"
if [ ! -d /root/emboviz ]; then
    git clone https://github.com/Ashok93/emboviz.git /root/emboviz
fi

echo "[bootstrap] setting up bashrc"
cp /root/emboviz/scripts/setup/bashrc.emboviz.sh /root/.bashrc.emboviz
if ! grep -q "source /root/.bashrc.emboviz" /root/.bashrc 2>/dev/null; then
    echo "source /root/.bashrc.emboviz" >> /root/.bashrc
fi
source /root/.bashrc.emboviz

echo "[bootstrap] creating cache dirs"
mkdir -p /root/hf_cache /root/uv_cache /root/outputs /root/venvs /root/repos /root/probes /root/logs

if [ ! -f /root/emboviz/.env ]; then
    echo "[bootstrap] WARNING: /root/emboviz/.env not found."
    echo "[bootstrap]   Create it with HF_TOKEN=<your-token> before running gated-repo models (pi0, gr00t)."
fi

echo "[bootstrap] done. Next: bash scripts/setup/install_all.sh"
