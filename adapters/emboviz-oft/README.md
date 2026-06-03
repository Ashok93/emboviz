# emboviz-oft

OpenVLA-OFT (Optimized Fine-Tuning) adapter for [emboviz](https://github.com/Ashok93/emboviz).

OFT pins a fork of `transformers` (moojink/transformers-openvla-oft)
and `openvla-oft` is research code (git-only). Both deps are
incompatible with mainline transformers and would conflict with the
OpenVLA / π0 / GR00T adapters' pins. This package isolates them in an
isolated venv (`~/.emboviz/venvs/oft`) spawned as a subprocess and
reached over the ZMQ wire.

## Install

From the [emboviz](../../README.md#installation) repo root:

```bash
uv sync --extra oft
```

Installs this adapter alongside core, both dataset readers, and the SAM 3 /
LaMa workers. Its isolated runtime venv builds automatically on the first
`uv run emboviz analyze` — you never build it by hand.

## Use

```bash
uv run emboviz analyze --config configs/oft.yaml
```

Copy the template and set `model.kwargs.checkpoint` to your own fine-tune.
