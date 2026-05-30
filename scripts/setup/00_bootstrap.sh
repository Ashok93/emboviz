#!/usr/bin/env bash
# Bootstrap a fresh GPU pod: install uv, set up env. The emboviz code is
# scp'd from the dev checkout (NOT cloned — see below). Idempotent.
set -euo pipefail

echo "[bootstrap] system packages — ffmpeg for torchcodec, git-lfs for Isaac-GR00T demo_data"
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git-lfs curl git

echo "[bootstrap] installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi
uv --version

# We scp the dev checkout to /root/emboviz so we test LOCAL (often unpushed)
# changes — we never clone from GitHub (it would be stale). Fail loudly if
# the code isn't here yet rather than silently proceeding without it.
echo "[bootstrap] checking emboviz code is present (scp'd from dev, never cloned)"
if [ ! -d /root/emboviz ]; then
    echo "[bootstrap] ERROR: /root/emboviz not found." >&2
    echo "[bootstrap]   scp your dev checkout first, e.g.:" >&2
    echo "[bootstrap]     git archive --format=tar.gz HEAD -o /tmp/emboviz.tgz" >&2
    echo "[bootstrap]     scp -O /tmp/emboviz.tgz <pod>:/root/" >&2
    echo "[bootstrap]     ssh <pod> 'mkdir -p /root/emboviz && tar xzf /root/emboviz.tgz -C /root/emboviz'" >&2
    exit 1
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
