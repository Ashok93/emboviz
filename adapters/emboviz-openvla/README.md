# emboviz-openvla

The OpenVLA-7B adapter for [emboviz](https://github.com/Ashok93/emboviz).

OpenVLA's deps (transformers >=4.40,<4.50, lerobot 0.3, torch, OpenVLA's
prismatic checkpoint code) conflict with several other VLA adapters'
deps. This package solves that by running the model in an isolated venv
(`~/.emboviz/venvs/openvla`) spawned as a subprocess and reached over the
ZMQ wire.

## Install

From the [emboviz](../../README.md#installation) repo root:

```bash
uv sync --extra openvla
```

Installs this adapter alongside core, both dataset readers, and the SAM 3 /
LaMa workers. Its isolated runtime venv builds automatically on the first
`uv run emboviz analyze` — you never build it by hand.

## Use

```bash
uv run emboviz analyze --config configs/openvla-bridge.yaml
```

The run is described entirely by the config (model + checkpoint, dataset
mapping, diagnostics). Copy the template and set `model.kwargs.hf_repo` to
your own fine-tune. The diagnostics run the model in the isolated ZMQ
worker venv; it never imports into your main environment.
