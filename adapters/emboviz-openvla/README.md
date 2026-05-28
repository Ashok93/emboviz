# emboviz-openvla

The OpenVLA-7B adapter for [emboviz](https://github.com/Ashok93/botsigil).

OpenVLA's deps (transformers 4.40-4.49, lerobot 0.3, torch, OpenVLA's
prismatic checkpoint code) conflict with several other VLA adapters'
deps. This package solves that by running the model in an **isolated
runtime venv** spawned via Ray's per-actor `runtime_env`.

## Install

```bash
# Core (small, no torch):
uv pip install emboviz

# This adapter (still small — a thin shim):
uv pip install emboviz-openvla

# Materialise the isolated runtime venv (one-time, downloads torch +
# transformers + lerobot + OpenVLA's prismatic into ~/.emboviz/venvs/openvla):
emboviz install-openvla
```

## Use

```bash
emboviz analyze \
    --model openvla \
    --dataset bridge \
    --episodes 537 \
    --mask-query "the cloth" \
    --diagnostics all \
    --output ./report
```

The diagnostic suite calls into the Ray actor that lives inside the
isolated venv; the model never imports into your main environment.
