# emboviz-act

ACT (Action Chunking Transformer) adapter for [emboviz](../README.md).

ACT ([Zhao et al. 2023](https://arxiv.org/abs/2304.13705)) is a DETR-style
CVAE policy: per-camera ResNet features and a proprioceptive-state token
feed a transformer encoder, and learned action queries cross-attend to
that memory to produce an action chunk. It consumes **vision + robot
state only — no language instruction**, and inference is deterministic
(the CVAE latent is the zero prior at inference time).

It runs in its own isolated venv (lerobot + torch) as a ZeroMQ worker,
the same pattern as the other model adapters.

## Install

```bash
uv pip install emboviz emboviz-lerobot emboviz-act
emboviz install-act          # builds the isolated lerobot runtime venv
```

## Run

```bash
emboviz-act serve --kwargs '{"checkpoint": "<repo_or_dir>", "camera_mapping": {"primary": "observation.images.top"}}'
```

The analyze runner auto-spawns the worker from the run config; the manual
command above is for reference.

## Config

```yaml
model:
  adapter: act
  kwargs:
    checkpoint: your-org/your-act-checkpoint   # HF repo id or local dir
    camera_mapping:                            # logical role -> the checkpoint's image-feature key
      primary: observation.images.top
      wrist:   observation.images.wrist
```

`camera_mapping` must cover exactly the checkpoint's `image_features`.
Normalization stats come from the checkpoint's own pre/post-processor
pipeline (`make_pre_post_processors(..., pretrained_path=checkpoint)`);
none are reconstructed here.

## Diagnostics

| Diagnostic | Supported |
|---|---|
| memorization, scene sensitivity, chunk consistency | ✅ |
| modality dropout | ✅ (instruction auto-skips — ACT has no language) |
| attention | ✅ decoder cross-attention (action query → encoder image tokens) |

**Attention.** ACT's map is the DETR-style decoder cross-attention from
the first action query to the encoder's image tokens. The image tokens
are a flattened ResNet feature grid (`H/stride × W/stride`, generally
non-square), reported with an explicit `(h, w)` grid shape. This is the
action pathway's attention, not a language-anchored object localizer —
read it as "where the action prediction attends." See
[`LITERATURE.md` §4](../../LITERATURE.md).
