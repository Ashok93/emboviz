# emboviz-pi0

Physical Intelligence's π0 / π0.5 adapter for [emboviz](https://github.com/Ashok93/botsigil).

π0 inference goes through `openpi`, which is **not on PyPI** and
**requires `GIT_LFS_SKIP_SMUDGE=1`** during install (its pinned
lerobot commit references unavailable LFS test fixtures). The
`emboviz install-pi0` command sets the env var automatically and
runs the openpi git install inside the isolated runtime venv.

## Install

```bash
uv pip install emboviz emboviz-lerobot emboviz-pi0
emboviz install-pi0
```

## Use

```bash
# Start the worker (stays warm between analyze runs):
emboviz-pi0 serve &

emboviz analyze --config configs/pi0-libero.yaml
```

Copy the template and set `model.kwargs.checkpoint_uri` to your own fine-tune.

Note: the first inference after install triggers checkpoint download
+ Triton autotune (~5–10 minutes). Subsequent runs are fast.

For attention-extraction diagnostics, π0 needs a PyTorch backend
checkpoint — run `emboviz convert-pi0 pi0_libero` once after install.
