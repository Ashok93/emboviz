"""SAM 3 text→mask detector.

Loads ``Sam3Model`` + ``Sam3Processor`` from HuggingFace transformers
on construction; subsequent :meth:`detect` calls reuse the loaded
state. Single-image, single-text-prompt inference per call.

Output shape (msgpack-friendly so :class:`~emboviz.adapters.wire` can
just pack it):

    {
        "instances": [
            {
                "bbox":  [x0, y0, x1, y1],
                "score": float,
                "mask":  np.ndarray (H, W) uint8 in {0, 1},
            },
            ...   # sorted highest-score-first
        ],
        "image_size": [H, W],
        "label": <target_text>,
    }

The mask is shipped as a raw uint8 ndarray rather than COCO-RLE
because msgpack-numpy's binary numpy codec is already efficient on
ZMQ (~0.3 ms per 480×640 mask over a local UDS, vs the ~5 KB
compressed payload + decode overhead of COCO-RLE). The client side
:mod:`emboviz_sam3.client` casts to bool at the boundary.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from typing import Any, Optional

import numpy as np
from PIL import Image


log = logging.getLogger("emboviz_sam3")

# Default checkpoint: an ungated, community-distilled SAM 3 LiteText
# variant (500M params, MobileCLIP text encoder) that uses the same
# SAM 3 processor + post-processing API. Picking it as the default
# means the adapter works out of the box without an HF access request.
#
# We use the ``yonigozlan/`` mirror specifically because it ships
# a fully-formed ``preprocessor_config.json`` (with
# ``image_processor_type`` set) so :func:`AutoProcessor.from_pretrained`
# loads cleanly — the alternate ``vil-uob/sam3-litetext-s0`` mirror
# is missing that field and crashes the auto-loader.
#
# Users with approved access to Meta's full SAM 3 checkpoint can opt
# in via:
#
#     EMBOVIZ_SAM3_MODEL_ID=facebook/sam3 emboviz-sam3 serve
#
# or by constructing the detector with model_id="facebook/sam3".
DEFAULT_MODEL_ID = "yonigozlan/sam3-litetext-s0"


class Sam3Detector:
    """Wraps SAM 3 inference behind a clean ``detect`` method.

    The HuggingFace model classes are imported lazily so importing this
    module is cheap in the user's main venv during entry-point
    discovery; the heavy load happens on first construction inside the
    SAM 3 runtime venv.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device_map: str = "auto",
        preload: bool = True,
    ):
        # Resolution order: explicit kwarg → env var → default. The
        # default is an ungated SAM 3-compatible distillation so a
        # fresh install needs no HF access request.
        self.model_id = model_id or os.environ.get(
            "EMBOVIZ_SAM3_MODEL_ID", DEFAULT_MODEL_ID,
        )
        self._device_map = device_map
        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()
        if preload:
            self._load()

    # ----- model lifecycle ------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def device(self) -> Optional[str]:
        return self._device

    def _load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            import torch
            from transformers import Sam3Model, Sam3Processor

            log.info("loading SAM 3 (%s) — this can take ~30 s on first run",
                     self.model_id)
            # Load the concrete IMAGE-level classes ``Sam3Model`` +
            # ``Sam3Processor`` explicitly — NOT ``AutoModel`` /
            # ``AutoProcessor``.
            #
            # The published ``facebook/sam3`` repo is the *video*
            # checkpoint: its config is ``Sam3VideoConfig`` and its
            # declared ``processor_class`` is ``Sam3VideoProcessor``. So
            # ``AutoModel`` / ``AutoProcessor`` resolve to the VIDEO
            # model + processor, whose ``__call__`` takes no ``text=``
            # argument — a text prompt then falls through to the image
            # processor kwargs and raises ``Sam3ImageProcessorKwargs.
            # __init__() got an unexpected keyword argument 'text'``.
            #
            # The documented API for image text→mask segmentation
            # (transformers SAM3 model docs) is the explicit pair:
            #     model     = Sam3Model.from_pretrained("facebook/sam3")
            #     processor = Sam3Processor.from_pretrained("facebook/sam3")
            #     inputs    = processor(images=img, text="...", ...)
            #     outputs   = model(**inputs)
            #     processor.post_process_instance_segmentation(outputs, ...)
            # ``Sam3Model`` loads the DETR detector held in the video
            # config's ``detector_config`` (a ``Sam3Config``) — the
            # open-vocabulary image detector SAM 3 exposes.
            self._processor = Sam3Processor.from_pretrained(self.model_id)
            self._model = Sam3Model.from_pretrained(self.model_id)
            # Place the model explicitly with ``.to(device)`` — the
            # documented SAM 3 pattern
            # (``Sam3Model.from_pretrained(...).to(device)``). We do NOT
            # use accelerate's ``device_map="auto"``: for this single
            # ~840M detector accelerate was leaving the whole model on
            # CPU even with 30+ GB of free GPU, turning each detection
            # into a ~140 s CPU forward instead of a sub-second GPU one.
            # SAM 3 fits on one GPU; there is nothing to shard.
            if self._device_map in (None, "auto"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device_map
            self._model = self._model.to(device)
            self._model.eval()
            self._device = str(next(self._model.parameters()).device)
            log.info("SAM 3 ready on device=%s", self._device)

    def close(self) -> None:
        """Release the model from GPU memory."""
        try:
            del self._model
            del self._processor
        finally:
            self._model = None
            self._processor = None
            self._device = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ----- core detection -------------------------------------------------

    def detect(
        self,
        image_bytes: bytes,
        target_text: str,
        score_threshold: float = 0.15,
        mask_threshold: float = 0.40,
    ) -> dict[str, Any]:
        """Run one (image, text) concept segmentation.

        Parameters
        ----------
        image_bytes
            PNG / JPEG bytes of the image to segment. Decoded with PIL.
        target_text
            The concept phrase to localize (e.g. ``"the mug"``). Must
            be non-empty — we never guess targets.
        score_threshold
            Instances with score below this are dropped at the source
            (mirrors what the perturber would otherwise filter). Default
            0.15 — below SAM 3's "high-precision" 0.30 so faint / small /
            partially-occluded targets survive on secondary views.
        mask_threshold
            Per-pixel mask-logit cutoff for binarization. Default 0.40,
            slightly below SAM 3's published 0.50, for a fuller mask.

        Returns
        -------
        ``{"instances": [...], "image_size": [H, W], "label": str}`` —
        ``instances`` sorted highest-score-first.
        """
        import torch
        if not image_bytes:
            raise ValueError("Sam3Detector.detect: empty image bytes")
        target_text = (target_text or "").strip()
        if not target_text:
            raise ValueError(
                "Sam3Detector.detect: ``target_text`` must be non-empty"
            )

        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        H, W = pil.height, pil.width

        self._load()
        assert self._model is not None and self._processor is not None
        model, processor = self._model, self._processor

        inputs = processor(
            images=pil, text=target_text, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = inputs.get("original_sizes")
        if target_sizes is None:
            target_sizes = [[H, W]]
        else:
            target_sizes = target_sizes.tolist()

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(score_threshold),
            mask_threshold=float(mask_threshold),
            target_sizes=target_sizes,
        )[0]

        masks = results.get("masks")
        scores = results.get("scores")
        boxes = results.get("boxes")
        if masks is None or scores is None:
            return {"instances": [], "image_size": [H, W], "label": target_text}

        masks_np = (
            masks.cpu().numpy().astype(np.uint8)
            if hasattr(masks, "cpu") else np.asarray(masks).astype(np.uint8)
        )
        scores_np = (
            scores.cpu().numpy().astype(float)
            if hasattr(scores, "cpu") else np.asarray(scores).astype(float)
        )
        boxes_np: Optional[np.ndarray] = None
        if boxes is not None:
            boxes_np = (
                boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
            )

        if masks_np.ndim == 2:
            masks_np = masks_np[None]

        instances: list[dict[str, Any]] = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            if not m.any():
                continue
            if boxes_np is not None and i < boxes_np.shape[0]:
                bx = boxes_np[i]
                x0, y0, x1, y1 = (int(round(float(v))) for v in bx)
            else:
                ys, xs = np.where(m)
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
            instances.append({
                "bbox":  [x0, y0, x1, y1],
                "score": float(scores_np[i]) if i < len(scores_np) else 1.0,
                "mask":  m,                     # uint8 (H, W)
            })

        instances.sort(key=lambda inst: -inst["score"])
        return {
            "instances": instances,
            "image_size": [H, W],
            "label": target_text,
        }

    # ----- introspection --------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return cheap introspection used by the ``health`` wire method."""
        return {
            "model_id": self.model_id,
            "model_loaded": self.loaded,
            "device": self._device,
        }
