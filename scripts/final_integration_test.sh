#!/usr/bin/env bash
# Run modality_dropout probe + full itest_runner on all 4 (model, dataset)
# pairs sequentially. Captures outputs to /root/probes/FINAL/<model>/.
#
# Sequential because 24 GB GPU doesn't fit 2 models at once.
set -uo pipefail   # not -e: continue even if one model fails
trap '' PIPE

source /root/.bashrc.emboviz
set -a; source /root/emboviz/.env; set +a

OUT_ROOT=/root/probes/FINAL
LOG_ROOT=/root/logs/FINAL
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

free_gpu() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
    sleep 3
}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Per-model fixture
run_model() {
    local model=$1
    local venv=$2
    local dataset=$3
    local episode=$4
    local target_text=$5
    local model_builder=$6
    local dataset_builder=$7
    local model_kwargs=$8   # may be empty string
    log "================== $model =================="
    log "  venv: $venv"
    log "  dataset: $dataset  ep$episode  target='$target_text'"
    log "  model_builder=$model_builder kwargs=$model_kwargs"

    local mdir="$OUT_ROOT/$model"
    mkdir -p "$mdir"

    # Shared modality-pool cache: probe writes, runner reads. Skips a
    # second pool build (which used to trigger HF 429).
    local pool_cache="$mdir/pool_cache"
    mkdir -p "$pool_cache"

    # 1. Modality dropout probe
    log "  [1/2] modality dropout probe ..."
    HF_TOKEN=$HF_TOKEN "$venv/bin/python" /root/emboviz/scripts/probe_modality_dropout.py \
        --model "$model" --dataset "$dataset" --episode "$episode" --frame 0 \
        --pool-size 8 --k-samples 4 \
        --pool-cache-dir "$pool_cache" \
        --out "$mdir/dropout" \
        > "$LOG_ROOT/${model}_dropout.log" 2>&1
    log "    -> exit $? -- see $LOG_ROOT/${model}_dropout.log"
    free_gpu

    # 2. Full runner (5 diagnostics + paraphrase + imitation if available)
    log "  [2/2] full runner ..."
    HF_TOKEN=$HF_TOKEN "$venv/bin/python" /root/emboviz/scripts/itest_runner.py \
        --story-id "${model}:final:ep${episode}" \
        --model-builder "$model_builder" \
        --model-kwargs-json "$model_kwargs" \
        --dataset-builder "$dataset_builder" \
        --episode-idx "$episode" --frame-start 0 --n-frames 4 \
        --out-dir "$mdir/runner" \
        --target-text "$target_text" \
        --modality-pool-size 8 --modality-k-samples 4 \
        --modality-pool-seed 0 \
        --modality-pool-cache-dir "$pool_cache" \
        > "$LOG_ROOT/${model}_runner.log" 2>&1
    log "    -> exit $? -- see $LOG_ROOT/${model}_runner.log"
    free_gpu
}

# --- OpenVLA on Bridge ep0 ---
run_model openvla \
    /root/venvs/openvla \
    bridge 0 "the spoon" \
    "emboviz.models.registry:get_model:openvla" \
    "emboviz.datasets.lerobot_bridge:BridgeEpisodeSource" \
    ""

# --- OFT on LIBERO-spatial ep0 ---
run_model oft \
    /root/repos/openvla-oft/.venv \
    libero-spatial 0 "the black bowl" \
    "emboviz.models.registry:get_model:openvla-oft" \
    "emboviz.datasets.lerobot_libero:LiberoSpatialSource" \
    ""

# --- GR00T on droid_sample ep1 (needs camera_mapping kwarg) ---
run_model gr00t \
    /root/venvs/gr00t \
    droid-sample 1 "the blue block" \
    "emboviz.models.registry:get_model:gr00t" \
    "emboviz.datasets.lerobot_droid:GR00TDroidSampleSource" \
    '{"camera_mapping": {"primary": "exterior_image_1_left", "wrist_left": "wrist_image_left"}}'

# --- pi0 on pi-libero ep0 (needs use_pytorch=True for extract_attention) ---
run_model pi0 \
    /root/repos/openpi/.venv \
    pi-libero 0 "the white mug" \
    "emboviz.models.registry:get_model:pi0" \
    "emboviz.datasets.lerobot_libero:PhysicalIntelligenceLiberoSource" \
    '{"config_name": "pi0_libero", "use_pytorch": true}'

log "=========== FINAL ALL DONE ==========="
log "Outputs: $OUT_ROOT/<model>/{dropout,runner}/"
log "Logs:    $LOG_ROOT/<model>_{dropout,runner}.log"
