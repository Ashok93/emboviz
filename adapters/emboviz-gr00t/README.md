# emboviz-gr00t

NVIDIA's [GR00T-N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B)
adapter for [emboviz](https://github.com/Ashok93/botsigil).

`emboviz install-gr00t` installs NVIDIA's `gr00t` package **with its own
declared dependencies** — gr00t's pyproject is the source of truth, we
mirror nothing. The one dependency we drop is `flash-attn`: it can't
build under pip's isolation (its `setup.py` imports torch before pip
installs it) and gr00t only ships prebuilt wheels for cp310/cp312 (this
venv is cp311). The adapter runs on SDPA / eager attention, so flash-attn
is never invoked — it's excluded via a uv `--override`, and every other
gr00t dependency installs normally.

## Install

```bash
uv pip install emboviz emboviz-gr00t emboviz-reader-gr00t
emboviz install-gr00t
emboviz install-reader-gr00t     # GR00T-format dataset reader (format: gr00t)
```

## Use

```bash
emboviz-gr00t serve &

emboviz analyze --config configs/gr00t-libero.yaml
```

Copy the template and set `model.kwargs.model_path` to your own fine-tune.
