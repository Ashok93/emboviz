# emboviz-sam3

SAM 3 sidecar service for [emboviz](../README.md).

## Why a sidecar

The official [`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)
repo requires Python 3.12+ and torch 2.7+. The HuggingFace `Sam3Model`
integration lives in `transformers >= 4.56`. None of the four VLA adapter
venvs that emboviz supports (OpenVLA, OpenVLA-OFT, π0, GR00T) can host
both of those constraints alongside their pinned model deps — OpenVLA
on transformers 4.49, OFT on a vendored fork, π0 on 4.53, GR00T on 4.57
(with its own Python 3.11 pin).

The fix is the same one every multi-runtime production setup uses:
**isolate SAM 3 in its own Python 3.12 venv and reach it over the ZeroMQ
wire** — the same DEALER/UDS transport the VLA model workers use. No
adapter venv needs torch/transformers to talk to it; the wire carries
bytes + msgpack and is Python-version-agnostic.

## Install

Ships with [emboviz](../../README.md#installation) core — `uv sync` installs
it; you do not install it separately. The isolated worker venv builds
automatically on first use.

`emboviz install-sam3` materialises `~/.emboviz/venvs/sam3` from this
adapter's `AdapterSpec` — the same path a PyPI user follows.

## Run

```bash
# preload the model BEFORE accepting requests (so the first detect()
# doesn't pay the ~30 s warmup)
uv run emboviz-sam3 serve
```

`emboviz-sam3 serve` starts a **ZMQ worker** over a Unix-domain socket
`ipc://~/.emboviz/sockets/sam3.sock`. Override the endpoint with the
`EMBOVIZ_SAM3_ENDPOINT` env var (e.g. `tcp://...` for a remote host).
There is no HTTP server, no port, no `EMBOVIZ_SAM3_URL`.

## API

The typed client is `emboviz_sam3.client.Sam3Client` (extends the wire's
`RpcClient` over the ZMQ DEALER/UDS transport). The emboviz-side caller
`emboviz.perturb._target_detection.SAM3Detector` wraps it.

```python
from emboviz_sam3.client import Sam3Client

client = Sam3Client()                       # endpoint from EMBOVIZ_SAM3_ENDPOINT
result = client.detect(
    image_bytes,                            # PNG/JPEG bytes
    target_text="the mug",                  # the concept phrase
    score_threshold=0.30,                   # optional
    mask_threshold=0.50,                    # optional
)
```

`detect(...)` returns:

```python
{
    "instances": [
        {
            "bbox": (x0, y0, x1, y1),
            "score": 0.91,
            "mask": <uint8 ndarray, shape (H, W)>,
        },
        ...
    ],
    "image_size": [H, W],
    "label": "the mug",
}
```

Each instance's `mask` is a raw `uint8` ndarray of shape `(H, W)` sent
over the ZMQ wire (msgpack-numpy) — not COCO RLE, not pycocotools, not
an HTTP payload.
