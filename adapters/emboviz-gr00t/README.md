# emboviz-gr00t

NVIDIA's [GR00T-N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B)
adapter for [emboviz](https://github.com/Ashok93/emboviz).

`emboviz install-gr00t` installs NVIDIA's `gr00t` package **with its own
declared dependencies** — gr00t's pyproject is the source of truth, we
mirror nothing. The one dependency we drop is `flash-attn`: it can't
build under pip's isolation (its `setup.py` imports torch before pip
installs it) and gr00t only ships prebuilt wheels for cp310/cp312 (this
venv is cp311). The adapter runs on SDPA / eager attention, so flash-attn
is never invoked — it's excluded via a uv `--override`, and every other
gr00t dependency installs normally.

## Install

From the [emboviz](../../README.md#installation) repo root:

```bash
uv sync --extra gr00t
```

Installs this adapter alongside core, both dataset readers, and the SAM 3 /
LaMa workers. Its isolated runtime venv builds automatically on the first
`uv run emboviz analyze` — you never build it by hand.

## Use

```bash
uv run emboviz-gr00t serve &

uv run emboviz analyze --config configs/gr00t-libero.yaml
```

Copy the template and set `model.kwargs.model_path` to your own fine-tune.
