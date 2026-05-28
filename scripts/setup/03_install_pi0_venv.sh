#!/usr/bin/env bash
# π0 adapter — dev pod recipe.
#
# Same shape as the user-facing path documented in README:
#
#     uv venv .venv-pi0 --python 3.11
#     uv pip install 'emboviz[pi0]'
#     emboviz install-pi0        # one-shot wrapper for openpi (NOT on PyPI;
#                                # needs GIT_LFS_SKIP_SMUDGE=1 in env)
#
# Per CLAUDE.md "Dev path is the user path": NO version pins here. The
# ``[pi0]`` extra owns transformers / torch. The ``emboviz install-pi0``
# CLI owns the openpi git+ install (the bit that can't be a transitive
# dep because env vars don't propagate into transitive builds).
set -euo pipefail
source /root/.bashrc.emboviz

VENV=/root/venvs/pi0
uv venv "$VENV" --python 3.11
uv pip install --python "$VENV/bin/python" -e "/root/emboviz[pi0]"

echo "[pi0] running 'emboviz install-pi0' (clones openpi with GIT_LFS_SKIP_SMUDGE=1)"
"$VENV/bin/emboviz" install-pi0

echo "[pi0] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, openpi, emboviz
print('  torch       ', torch.__version__, '  cuda_avail=', torch.cuda.is_available())
print('  transformers', transformers.__version__)
print('  openpi      ', openpi.__file__)
print('  emboviz     ', emboviz.__file__)
"
echo "[pi0] DONE — $VENV/bin/python ready"
echo "[pi0] Note: first inference triggers checkpoint download + Triton autotune (~5-10 min)"
