#!/usr/bin/env bash
# OpenVLA-OFT adapter — dev pod recipe.
#
# Same shape as the user-facing path documented in README:
#
#     uv venv .venv-oft --python 3.10
#     uv pip install 'emboviz[oft]'
#
# The ``[oft]`` extra resolves moojink's vendored transformers fork and
# the openvla-oft package via PEP 508 ``pkg @ git+...`` direct refs, so
# a plain ``uv pip install`` is enough — no manual ``git clone`` step.
#
# Per CLAUDE.md "Dev path is the user path": NO version pins here.
set -euo pipefail
source /root/.bashrc.emboviz

VENV=/root/venvs/oft
uv venv "$VENV" --python 3.10
uv pip install --python "$VENV/bin/python" -e "/root/emboviz[oft]"

echo "[oft] sanity import"
"$VENV/bin/python" -c "
import torch, transformers, emboviz
print('  torch       ', torch.__version__, '  cuda_avail=', torch.cuda.is_available())
print('  transformers', transformers.__version__)
print('  emboviz     ', emboviz.__file__)
try:
    import prismatic
    print('  prismatic   ', prismatic.__file__)
except Exception as e:
    print('  prismatic import failed:', type(e).__name__, e)
"
echo "[oft] DONE — $VENV/bin/python ready"
