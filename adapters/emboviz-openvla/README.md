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
emboviz analyze --config configs/openvla-bridge.yaml
```

The run is described entirely by the config (model + checkpoint, dataset
mapping, diagnostics). Copy the template and set `model.kwargs.hf_repo` to
your own fine-tune. The diagnostic suite runs the model in the isolated ZMQ
worker venv; it never imports into your main environment.
