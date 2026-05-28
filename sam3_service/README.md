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
**isolate SAM 3 in its own Python 3.12 venv and expose it over HTTP**.
Every adapter venv only needs `httpx` (pure-Python, no torch/transformers)
to talk to it.

## Install

```bash
uv venv /root/venvs/sam3 --python 3.12
uv pip install --python /root/venvs/sam3/bin/python -e .
```

Or use the bundled installer (called by emboviz's `install_all.sh`):

```bash
bash ../scripts/setup/05_install_sam3_venv.sh
```

## Run

```bash
# preload the model BEFORE accepting requests (so first /detect doesn't
# pay the ~30 s warmup)
/root/venvs/sam3/bin/emboviz-sam3 serve --preload
```

Defaults to `127.0.0.1:8311`. Override the model with
`EMBOVIZ_SAM3_MODEL_ID=<hf_repo>` env var if you have a fine-tuned SAM 3
checkpoint.

The emboviz client (`emboviz.perturb._target_detection.SAM3Detector`)
talks to this server. Override the URL with
`EMBOVIZ_SAM3_URL=http://...` if the server runs on a non-default port
or another host.

## API

### `GET /health`

```json
{
  "alive": true,
  "model_loaded": true,
  "model_id": "facebook/sam3",
  "device": "cuda:0",
  "version": "0.1.0"
}
```

### `POST /detect` (multipart)

Form fields:
- `image`: PNG/JPEG bytes
- `target_text`: the concept phrase (e.g. `"the mug"`)
- `score_threshold`: optional, default 0.30
- `mask_threshold`: optional, default 0.50

Response (JSON):
```json
{
  "instances": [
    {
      "bbox": [x0, y0, x1, y1],
      "score": 0.91,
      "mask": {"size": [H, W], "counts": "<COCO RLE>"}
    },
    ...
  ],
  "image_size": [H, W],
  "label": "the mug"
}
```

Masks are COCO RLE-encoded so the JSON payload stays small (a 480×640
boolean mask is ~5 KB compressed vs ~300 KB raw). Decode client-side
with `pycocotools.mask.decode`.
