#!/usr/bin/env bash
# Download all models + datasets we need for the 4-model audit suite.
# Everything goes to $HF_HOME=/root/hf_cache (overlay; fast).
#
# Run:    bash /root/emboviz/scripts/download_models.sh 2>&1 | tee /root/logs/downloads.log
set -euo pipefail

source /root/.bashrc.emboviz
set -a; source /root/emboviz/.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1   # turbo (parallel chunked) downloads
PY=/root/venvs/openvla/bin/python    # openvla venv has huggingface_hub

log() { echo "[dl $(date +%H:%M:%S)] $*"; }

# Helper: snapshot a HF repo into $HF_HOME via huggingface_hub.
hf_dl() {
    local repo="$1"
    local kind="${2:-model}"   # model | dataset
    log "$kind: $repo — starting"
    $PY - <<PY 2>&1 | tail -3
import os, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(
    repo_id="$repo",
    repo_type="$kind",
    token=os.environ.get("HF_TOKEN"),
)
print(f"  -> {p}  ({time.time()-t0:.1f}s)")
PY
    log "$kind: $repo DONE"
}

# Ensure hf_transfer is installed (gives 2-5x throughput on big files).
$PY -m pip install --quiet hf_transfer 2>&1 | tail -2 || true

# ------------------------------------------------ Vision models for memorization
hf_dl "IDEA-Research/grounding-dino-tiny" model
hf_dl "facebook/sam-vit-base" model

# ------------------------------------------------ Policy models
hf_dl "openvla/openvla-7b" model
hf_dl "moojink/openvla-7b-oft-finetuned-libero-spatial" model
hf_dl "nvidia/GR00T-N1.7-3B" model

# ------------------------------------------------ Datasets
# LeRobot datasets — snapshot the metadata + a single episode's parquet/mp4.
# Full splits are huge; for our audit we only need 1-3 episodes per dataset.
log "datasets: Bridge — 1 episode probe via LeRobotDataset"
$PY - <<'PY' 2>&1 | tail -10 || true
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import time
t0 = time.time()
d = LeRobotDataset("IPEC-COMMUNITY/bridge_orig_lerobot", episodes=[0])
print(f"  bridge ep0: {d.num_frames} frames ({time.time()-t0:.1f}s)")
PY

log "datasets: LIBERO-spatial (community) — 1 episode probe"
$PY - <<'PY' 2>&1 | tail -10 || true
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import time
t0 = time.time()
d = LeRobotDataset("aopolin-lv/libero_spatial_no_noops_lerobot_v21", episodes=[0])
print(f"  libero-spatial ep0: {d.num_frames} frames ({time.time()-t0:.1f}s)")
PY

log "datasets: physical-intelligence/libero — 1 episode probe (needs OFT/pi0 venv with newer lerobot)"
# OFT venv has the newer lerobot that supports physical-intelligence/libero format
/root/repos/openvla-oft/.venv/bin/python - <<'PY' 2>&1 | tail -10 || true
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import time
t0 = time.time()
d = LeRobotDataset("physical-intelligence/libero", episodes=[0])
print(f"  pi-libero ep0: {d.num_frames} frames ({time.time()-t0:.1f}s)")
PY

log "datasets: GR00T droid_sample — bundled with Isaac-GR00T repo (no download needed)"
ls /root/repos/Isaac-GR00T/demo_data/ 2>&1 | head

# ------------------------------------------------ π0 checkpoint via openpi
log "pi0: openpi checkpoint pi0_libero from gs://openpi-assets/checkpoints/"
/root/repos/openpi/.venv/bin/python - <<'PY' 2>&1 | tail -15 || true
import time
t0 = time.time()
from openpi.shared import download
p = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
print(f"  pi0_libero checkpoint: {p}  ({time.time()-t0:.1f}s)")
PY

log "=== ALL DOWNLOADS DONE ==="
log "disk used in /root: $(du -sh /root | cut -f1)"
log "hf_cache size:      $(du -sh /root/hf_cache | cut -f1)"
log "free on overlay:    $(df -h / | tail -1 | awk '{print $4}')"
