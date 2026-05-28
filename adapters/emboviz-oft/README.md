# emboviz-oft

OpenVLA-OFT (Optimized Fine-Tuning) adapter for [emboviz](https://github.com/Ashok93/botsigil).

OFT pins a fork of `transformers` (moojink/transformers-openvla-oft)
and `openvla-oft` is research code (git-only). Both deps are
incompatible with mainline transformers and would conflict with the
OpenVLA / π0 / GR00T adapters' pins. This package isolates them via
Ray's per-actor `runtime_env`.

## Install

```bash
uv pip install emboviz emboviz-oft
emboviz install-oft
```

## Use

```bash
emboviz analyze --config configs/oft-libero-spatial.yaml
```

Copy the template and set `model.kwargs.checkpoint` to your own fine-tune.
