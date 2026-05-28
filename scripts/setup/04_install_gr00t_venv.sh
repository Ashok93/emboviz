#!/usr/bin/env bash
# GR00T adapter — dev pod recipe.
#
# Same shape as the user-facing path documented in README:
#
#     uv venv .venv-gr00t --python 3.11
#     uv pip install 'emboviz[gr00t]'
#     emboviz install-gr00t      # one-shot wrapper for NVIDIA's gr00t
#                                # (NOT on PyPI; flash-attn build-isolation
#                                # bug needs --no-deps install)
#
# Per CLAUDE.md "Dev path is the user path": NO version pins here. The
# ``[gr00t]`` extra owns transformers / torch / lerobot. The
# ``emboviz install-gr00t`` CLI owns the gr00t git+ install with
# --no-deps.
#
# We also pull the Isaac-GR00T repo's git-lfs blobs into a known location
# so the dataset adapter's demo_data/droid_sample (3 sample episodes) is
# usable. A USER who installed gr00t via pip and wants droid_sample does
# the same clone manually; documented in README.
set -euo pipefail
source /root/.bashrc.emboviz

# Pull the upstream repo for demo_data/droid_sample. The gr00t Python
# package itself is installed by ``emboviz install-gr00t`` below from
# the same git URL (--no-deps).
REPO=/root/repos/Isaac-GR00T
if [ ! -d "$REPO" ]; then
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$REPO"
fi
( cd "$REPO" && git lfs install --skip-smudge --local && git lfs pull )

VENV=/root/venvs/gr00t
uv venv "$VENV" --python 3.11
uv pip install --python "$VENV/bin/python" -e "/root/emboviz[gr00t]"

echo "[gr00t] running 'emboviz install-gr00t' (installs gr00t with --no-deps)"
"$VENV/bin/emboviz" install-gr00t

echo "[gr00t] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, gr00t, emboviz
print('  torch       ', torch.__version__, '  cuda_avail=', torch.cuda.is_available())
print('  transformers', transformers.__version__)
print('  gr00t       ', gr00t.__file__)
print('  emboviz     ', emboviz.__file__)
"
echo "[gr00t] DONE — $VENV/bin/python ready"
echo "[gr00t] Note: first inference downloads nvidia/GR00T-N1.7-3B (~6 GB) + Cosmos-Reason2-2B (gated, needs HF_TOKEN)"
