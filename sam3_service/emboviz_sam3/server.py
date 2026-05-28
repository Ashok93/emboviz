"""FastAPI server that hosts SAM 3 and answers text→mask queries over HTTP.

Endpoints
---------

``GET  /health``            — liveness + model-loaded readiness probe
``POST /detect``            — multipart: ``image`` (bytes) + ``req`` (JSON
                              :class:`DetectRequest`) → :class:`DetectResponse`

Lifecycle
---------

The SAM 3 model is lazy-loaded on the FIRST ``/detect`` call (so
``--host 0.0.0.0`` can come up fast and ``/health`` is honest about
``model_loaded=False`` until then). Subsequent calls reuse the loaded
model. A second lock guards the load so concurrent first-callers don't
race.

We use the canonical HuggingFace ``transformers`` API per the SAM 3
model card:

    inputs  = processor(images=pil, text=phrase, return_tensors="pt")
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=..., mask_threshold=...,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

``results`` carries per-instance ``masks`` (binarized, resized to the
original image HxW), ``boxes`` (xyxy pixel coords), and ``scores``.
"""
from __future__ import annotations

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from emboviz_sam3 import __version__
from emboviz_sam3.schema import (
    CocoRLE,
    DetectRequest,
    DetectResponse,
    Health,
    Instance,
)

log = logging.getLogger("emboviz_sam3")

# Model identifier — overridable via env var so a user can point at a
# fine-tuned SAM 3 checkpoint without code changes.
SAM3_MODEL_ID = os.environ.get("EMBOVIZ_SAM3_MODEL_ID", "facebook/sam3")


# ─── lazy state ────────────────────────────────────────────────────────────

class _State:
    """Holds the loaded model + processor and a lock so the first request
    triggers a single load, not N concurrent loads."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            self._load_sync()

    def _load_sync(self) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        log.info("loading SAM 3: %s (this can take ~30 s on first run)",
                 SAM3_MODEL_ID)
        # ``device_map="auto"`` lets accelerate place the model on whatever
        # GPU is available; on a single-GPU pod that is GPU 0.
        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
        self.model = Sam3Model.from_pretrained(
            SAM3_MODEL_ID,
            device_map="auto",
        )
        self.model.eval()
        # Resolve the device from the loaded model's first parameter.
        self.device = str(next(self.model.parameters()).device)
        log.info("SAM 3 ready on device=%s", self.device)


STATE = _State()


# ─── FastAPI app ───────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    log.info("emboviz-sam3 %s starting (model not yet loaded)", __version__)
    yield
    log.info("emboviz-sam3 shutting down")


app = FastAPI(
    title="emboviz-sam3",
    version=__version__,
    summary="SAM 3 text→mask sidecar for emboviz",
    lifespan=_lifespan,
)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(
        alive=True,
        model_loaded=STATE.loaded,
        model_id=SAM3_MODEL_ID if STATE.loaded else None,
        device=STATE.device,
        version=__version__,
    )


@app.post("/load", response_model=Health)
def force_load() -> Health:
    """Optional: trigger the model load explicitly (instead of on first
    /detect). Useful when a client wants to gate on ``model_loaded=True``
    before starting a long episode run."""
    STATE.ensure_loaded()
    return health()


@app.post("/detect", response_model=DetectResponse)
def detect(
    image: UploadFile = File(..., description="The PNG/JPEG image bytes."),
    target_text: str = Form(...),
    score_threshold: float = Form(0.30),
    mask_threshold: float = Form(0.50),
) -> DetectResponse:
    """Run SAM 3 concept segmentation: image + text → instance masks.

    The image arrives as raw bytes (multipart upload) so the client does
    not have to base64-encode it. We decode with PIL, run SAM 3, and
    encode each instance's binary mask as COCO RLE before returning so
    the JSON payload stays small.
    """
    # Validate the JSON-ish body fields via DetectRequest's validators
    # (Form() bypasses pydantic so we re-validate here).
    req = DetectRequest(
        target_text=target_text,
        score_threshold=float(score_threshold),
        mask_threshold=float(mask_threshold),
    )

    raw = image.file.read()
    if not raw:
        raise HTTPException(400, "uploaded image was empty")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: "
                                 f"{type(e).__name__}: {e}")

    STATE.ensure_loaded()
    assert STATE.model is not None and STATE.processor is not None

    return _run_detect(pil, req)


def _run_detect(pil: Image.Image, req: DetectRequest) -> DetectResponse:
    import torch
    from pycocotools import mask as coco_mask

    assert STATE.model is not None and STATE.processor is not None
    model, processor = STATE.model, STATE.processor

    inputs = processor(
        images=pil, text=req.target_text, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    # The processor's post-process returns one results dict per image in
    # the batch; we always pass a single image so we take [0].
    target_sizes = inputs.get("original_sizes")
    if target_sizes is None:
        target_sizes = [list(pil.size[::-1])]   # (H, W)
    else:
        target_sizes = target_sizes.tolist()
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=req.score_threshold,
        mask_threshold=req.mask_threshold,
        target_sizes=target_sizes,
    )[0]

    masks = results.get("masks")
    scores = results.get("scores")
    boxes = results.get("boxes")
    if masks is None or scores is None:
        # Empty detection → honest empty response.
        return DetectResponse(
            instances=[], image_size=[pil.height, pil.width],
            label=req.target_text,
        )

    masks_np: np.ndarray = (
        masks.cpu().numpy().astype(bool)
        if hasattr(masks, "cpu")
        else np.asarray(masks).astype(bool)
    )
    scores_np: np.ndarray = (
        scores.cpu().numpy().astype(float)
        if hasattr(scores, "cpu")
        else np.asarray(scores).astype(float)
    )
    boxes_np: Optional[np.ndarray] = None
    if boxes is not None:
        boxes_np = (
            boxes.cpu().numpy()
            if hasattr(boxes, "cpu")
            else np.asarray(boxes)
        )

    if masks_np.ndim == 2:
        masks_np = masks_np[None]

    instances: list[Instance] = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if not m.any():
            continue
        ys, xs = np.where(m)
        if boxes_np is not None and i < boxes_np.shape[0]:
            bx = boxes_np[i]
            x0, y0, x1, y1 = (int(round(float(v))) for v in bx)
        else:
            x0, y0, x1, y1 = (
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            )
        # COCO RLE expects Fortran-order uint8 (H, W).
        rle = coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))
        # ``counts`` is bytes; ship as latin-1 string so it round-trips
        # through JSON. The client decodes with ``encode("latin-1")``.
        counts_str = rle["counts"].decode("latin-1")
        instances.append(Instance(
            bbox=[x0, y0, x1, y1],
            score=float(scores_np[i]) if i < len(scores_np) else 1.0,
            mask=CocoRLE(size=list(rle["size"]), counts=counts_str),
        ))

    # Highest-scoring first so consumers that take ``[0]`` get the best.
    instances.sort(key=lambda inst: -inst.score)
    return DetectResponse(
        instances=instances,
        image_size=[pil.height, pil.width],
        label=req.target_text,
    )
