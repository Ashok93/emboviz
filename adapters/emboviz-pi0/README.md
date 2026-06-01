# emboviz-pi0

Physical Intelligence's π0 / π0.5 adapter for [emboviz](https://github.com/Ashok93/emboviz).

π0 inference goes through `openpi`, which is **not on PyPI** and
**requires `GIT_LFS_SKIP_SMUDGE=1`** during install (its pinned
lerobot commit references unavailable LFS test fixtures). The
`emboviz install-pi0` command sets the env var automatically and
runs the openpi git install inside the isolated runtime venv.

## Install

From the [emboviz](../../README.md#installation) repo root:

```bash
uv sync --extra pi0
```

Installs this adapter alongside core, both dataset readers, and the SAM 3 /
LaMa workers. Its isolated runtime venv builds automatically on the first
`uv run emboviz analyze` — you never build it by hand.

## Use

```bash
# Start the worker (stays warm between analyze runs):
uv run emboviz-pi0 serve &

uv run emboviz analyze --config configs/pi0-libero.yaml
```

Copy the template and set `model.kwargs.checkpoint_uri` to your own fine-tune.

Note: the first inference after install triggers checkpoint download
+ Triton autotune (~5–10 minutes). Subsequent runs are fast.

For attention-extraction diagnostics, π0 needs a PyTorch backend
checkpoint — run `emboviz convert-pi0 pi0_libero` once after install.
