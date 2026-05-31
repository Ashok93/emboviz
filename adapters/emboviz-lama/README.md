# emboviz-lama

LaMa (big-lama) inpainting sidecar for [emboviz](../README.md) — the
**on-manifold third mask-fill** of the memorization diagnostic.

## Why a sidecar

The memorization diagnostic masks the manipulated target and measures how
much the policy's action changes. To avoid measuring the model's reaction
to the *masking artifact* instead of the *absence of the object*,
[`LITERATURE.md` §1](../../LITERATURE.md) prescribes a fill **ensemble**
spanning the OOD/on-manifold axis:

| Fill | On-manifold? | Ships in |
|---|---|---|
| `channel_mean` | no (aggressive, OOD) | core |
| `gaussian_blur` | no (OOD-leaning) | core |
| `lama_inpaint` | **yes** (plausible background) | **this adapter** |

LaMa needs `torch`, which can't share a venv with the VLA adapters, so it
runs as its own ZeroMQ worker — the same pattern as `emboviz-sam3`. It is
**deterministic** and **feed-forward**, unlike the 2025-era diffusion
object-removers; a calibrated diagnostic needs a reproducible,
conservative fill that does not hallucinate new content into the hole.

## Install

```bash
uv pip install emboviz emboviz-lama
emboviz install-lama          # builds the isolated runtime venv
```

`emboviz install-lama` materialises `~/.emboviz/venvs/lama` from this
adapter's `AdapterSpec` — the same path a PyPI user follows.

## Run

```bash
# preload the model + run a one-forward self-test BEFORE accepting
# requests (the analyze runner auto-spawns this for you when a config
# requests the lama_inpaint fill)
emboviz-lama serve
```

`emboviz-lama serve` starts a **ZMQ worker** over a Unix-domain socket
`ipc://~/.emboviz/sockets/lama.sock`. Override the endpoint with the
`EMBOVIZ_LAMA_ENDPOINT` env var (e.g. `tcp://...` for a remote host).

## Weights / provenance

The default checkpoint is the TorchScript `big-lama.pt` at
[`okaris/big-lama`](https://huggingface.co/okaris/big-lama) on the
HuggingFace Hub, **pinned to commit `a77c4957…`** so the bytes are
reproducible. okaris authored the original simple-LaMa wrapper; this is
the canonical TorchScript export of the
[`advimman/lama`](https://github.com/advimman/lama) big-lama weights —
Apache-2.0 by derivation. Override via:

```bash
EMBOVIZ_LAMA_MODEL=/path/to/big-lama.pt emboviz-lama serve   # local .pt
EMBOVIZ_LAMA_REPO=other/repo EMBOVIZ_LAMA_REVISION=<sha> emboviz-lama serve
```

The preprocessing (`normalize /255`, `mask > 0`, symmetric pad-to-mod-8)
is vendored from
[`simple-lama-inpainting`](https://github.com/enesmsahin/simple-lama-inpainting)
(Apache-2.0), with two fixes the worker adds for an honest intervention:
it **crops** the mod-8 padding back to the original size, and it
**composites** LaMa's output onto the original *only within the mask* so
every non-target pixel stays byte-identical.

## API

The typed client is `emboviz_lama.client.LamaClient` (extends the wire's
`RpcClient` over the ZMQ DEALER/UDS transport). The emboviz-side caller
`emboviz.perturb.image._inpaint.LamaInpainter` wraps it.

```python
from emboviz_lama.client import LamaClient

client = LamaClient()                       # endpoint from EMBOVIZ_LAMA_ENDPOINT
filled = client.inpaint(image_png_bytes, mask)   # mask: HxW uint8/bool
# filled is an (H, W, 3) uint8 ndarray — original image, masked region
# replaced by LaMa's inpainting.
```
