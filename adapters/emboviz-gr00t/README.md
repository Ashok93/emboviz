# emboviz-gr00t

NVIDIA's [GR00T-N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B)
adapter for [emboviz](https://github.com/Ashok93/botsigil).

GR00T's upstream package on PyPI depends on `flash-attn`, which can't
build under pip's isolation (its `setup.py` imports torch before pip
installs it). `emboviz install-gr00t` handles this with a two-pass
install — standard deps first, then `gr00t` itself with `--no-deps`.
The adapter falls back to SDPA at runtime so flash-attn is never
actually invoked.

## Install

```bash
uv pip install emboviz emboviz-gr00t
emboviz install-gr00t
```

## Use

```bash
emboviz-gr00t serve &

emboviz analyze --config configs/gr00t-droid-sample.yaml
```

Copy the template and set `model.kwargs.model_path` to your own fine-tune.
