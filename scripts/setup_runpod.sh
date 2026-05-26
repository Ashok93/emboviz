#!/usr/bin/env bash
# Bootstrap 4 per-model venvs on a fresh RunPod pod.
#
# Layout: everything in /root (fast overlay). HF cache, uv cache, venvs,
# cloned repos, outputs — all ephemeral. /mnt/persistent is reserved for
# user-triggered "save before destroy" copies.
#
# Run:    bash /root/emboviz/scripts/setup_runpod.sh 2>&1 | tee /root/logs/setup.log
#
# Idempotent: re-running skips venvs that already have their marker file.
set -euo pipefail

source /root/.bashrc.emboviz
set -a; source /root/emboviz/.env; set +a

EMBOVIZ=/root/emboviz
VENVS=/root/venvs
REPOS=/root/repos
LOGS=/root/logs
mkdir -p "$VENVS" "$REPOS" "$LOGS" /root/hf_cache /root/uv_cache /root/outputs

log() { echo "[setup $(date +%H:%M:%S)] $*"; }

mark_done() { touch "$1/.emboviz_ok"; }
is_done()   { [ -f "$1/.emboviz_ok" ]; }

# ---------------------------------------------------------------- 1. OpenVLA
install_openvla() {
    local venv="$VENVS/openvla"
    if is_done "$venv"; then log "openvla: already installed, skip"; return; fi
    log "openvla: creating venv (python 3.10)"
    uv venv --python 3.10 "$venv"
    log "openvla: installing emboviz[openvla]"
    uv pip install --python "$venv/bin/python" -e "$EMBOVIZ[openvla]" 2>&1 | tail -15
    mark_done "$venv"
    log "openvla: DONE"
}

# ---------------------------------------------------------------- 2. OFT
install_oft() {
    local venv="$REPOS/openvla-oft/.venv"
    if is_done "$venv"; then log "oft: already installed, skip"; return; fi
    if [ ! -d "$REPOS/openvla-oft" ]; then
        log "oft: cloning openvla-oft repo"
        git clone --depth 1 https://github.com/moojink/openvla-oft.git "$REPOS/openvla-oft"
    fi
    log "oft: creating venv (python 3.10)"
    uv venv --python 3.10 "$venv"
    log "oft: installing openvla-oft (its own transformers fork)"
    uv pip install --python "$venv/bin/python" -e "$REPOS/openvla-oft" 2>&1 | tail -15
    log "oft: installing emboviz on top (no-deps to keep OFT's transformers pin)"
    uv pip install --python "$venv/bin/python" --no-deps -e "$EMBOVIZ" 2>&1 | tail -5
    log "oft: installing lerobot + emboviz base deps for dataset loading"
    uv pip install --python "$venv/bin/python" \
        "lerobot>=0.3,<0.4" "huggingface_hub>=0.26,<1.0" \
        "rerun-sdk>=0.22.1" "mcap>=1.3.1" "tqdm" "Pillow" 2>&1 | tail -5
    mark_done "$venv"
    log "oft: DONE"
}

# ---------------------------------------------------------------- 3. π0 (openpi)
install_pi0() {
    local venv="$REPOS/openpi/.venv"
    if is_done "$venv"; then log "pi0: already installed, skip"; return; fi
    if [ ! -d "$REPOS/openpi" ]; then
        log "pi0: cloning openpi (with submodules)"
        git clone --recurse-submodules --depth 1 \
            https://github.com/Physical-Intelligence/openpi.git "$REPOS/openpi"
    fi
    log "pi0: uv sync (openpi pins its own torch/jax stack)"
    cd "$REPOS/openpi"
    GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -20
    log "pi0: installing emboviz on top (no-deps)"
    uv pip install --python "$venv/bin/python" --no-deps -e "$EMBOVIZ" 2>&1 | tail -5
    cd /root
    mark_done "$venv"
    log "pi0: DONE"
}

# ---------------------------------------------------------------- 4. GR00T
install_gr00t() {
    local venv="$VENVS/gr00t"
    if is_done "$venv"; then log "gr00t: already installed, skip"; return; fi
    if [ ! -d "$REPOS/Isaac-GR00T" ]; then
        log "gr00t: cloning Isaac-GR00T"
        git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git "$REPOS/Isaac-GR00T"
    fi
    log "gr00t: creating venv (python 3.10)"
    uv venv --python 3.10 "$venv"
    log "gr00t: installing emboviz[gr00t]"
    uv pip install --python "$venv/bin/python" -e "$EMBOVIZ[gr00t]" 2>&1 | tail -15
    log "gr00t: installing Isaac-GR00T (no-deps to keep transformers>=4.57)"
    uv pip install --python "$venv/bin/python" --no-deps -e "$REPOS/Isaac-GR00T" 2>&1 | tail -5
    mark_done "$venv"
    log "gr00t: DONE"
}

# ----------------------------------------------------------------- main
log "=== bootstrap start ==="
log "uv: $(uv --version)"
log "EMBOVIZ=$EMBOVIZ  VENVS=$VENVS  REPOS=$REPOS"
log "HF_HOME=$HF_HOME  UV_CACHE_DIR=$UV_CACHE_DIR"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

# Run sequentially — parallel installs would saturate uv cache / network
# and not actually be faster on a single-CPU container.
install_openvla
install_oft
install_pi0
install_gr00t

log "=== bootstrap complete ==="
log "Per-venv pythons:"
for v in "$VENVS/openvla" "$REPOS/openvla-oft/.venv" "$REPOS/openpi/.venv" "$VENVS/gr00t"; do
    if [ -f "$v/bin/python" ]; then
        echo "  $v: $($v/bin/python --version 2>&1)"
    fi
done
echo
log "disk used in /root: $(du -sh /root | cut -f1)"
log "free on overlay:   $(df -h / | tail -1 | awk '{print $4}')"
