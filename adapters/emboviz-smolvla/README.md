# emboviz-smolvla

SmolVLA adapter for [emboviz](../README.md).

SmolVLA ([Shukor et al. 2025](https://huggingface.co/blog/smolvla)) is a
compact vision-language-action model: a SmolVLM2 backbone (SigLIP vision
encoder + SmolLM2 decoder) processes images, a language instruction, and
the robot state, and a flow-matching action expert produces an action
chunk. Inference is **stochastic** (the action expert samples noise and
denoises), so per-frame predictions are averaged over samples by the
calibration layer.

It runs in its own isolated venv (`lerobot[smolvla]` + torch) as a ZeroMQ
worker, the same pattern as the other model adapters.

## Install

```bash
uv pip install emboviz emboviz-lerobot emboviz-smolvla
emboviz install-smolvla       # builds the isolated lerobot[smolvla] runtime venv
```

## Run

```bash
emboviz-smolvla serve --kwargs '{"checkpoint": "lerobot/smolvla_base", "camera_mapping": {"primary": "observation.images.top"}}'
```

The analyze runner auto-spawns the worker from the run config.

## Config

```yaml
model:
  adapter: smolvla
  kwargs:
    checkpoint: lerobot/smolvla_base           # HF repo id or local dir (your finetune)
    camera_mapping:                            # logical role -> the checkpoint's image-feature key
      primary: observation.images.top
      wrist:   observation.images.wrist
```

`camera_mapping` must cover exactly the checkpoint's `image_features`.
The instruction is tokenized by the checkpoint's pre-processor. Normalization
stats come from the checkpoint itself, across both lerobot layouts: a saved
processor pipeline (`policy_preprocessor.json`, lerobot ≥ 0.5) is loaded
directly; an older checkpoint that bakes the stats into `model.safetensors`
has them read back into `dataset_stats` and the pipeline rebuilt from those.
If a feature needs normalization but the checkpoint carries neither, the
adapter raises rather than running it un-normalized.

## Diagnostics

| Diagnostic | Supported |
|---|---|
| memorization, scene sensitivity, chunk consistency, modality dropout (incl. instruction) | ✅ |
| attention | ✅ SmolVLM2 prefix self-attention (instruction token → image patches) |

**Attention.** The map is the last instruction token's attention over the
image patches, read from the SmolVLM2 prefix forward that fills the KV
cache before the action expert denoises — the same visual-grounding
signal the OpenVLA and π0 maps use. The action expert's suffix→prefix
attention (the action pathway) is not used for this map. See
[`LITERATURE.md` §4](../../LITERATURE.md).
