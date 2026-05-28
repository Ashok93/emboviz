# Source this from /root/.bashrc (or copy to /root/.bashrc.emboviz and
# `source /root/.bashrc.emboviz` per shell). Sets HF/uv/emboviz caches to
# the fast overlay path. /mnt/persistent is reserved for things we copy
# manually before destroying the pod (199× slower for small files).

export HF_HOME=/root/hf_cache
export HF_HUB_CACHE=/root/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/root/hf_cache/hub
export TRANSFORMERS_CACHE=/root/hf_cache/transformers
export UV_CACHE_DIR=/root/uv_cache
export EMBOVIZ_HOME=/root/emboviz
export EMBOVIZ_OUTPUTS=/root/outputs
export PATH=/root/.local/bin:$PATH
