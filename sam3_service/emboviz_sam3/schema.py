"""Wire schema for the emboviz ↔ emboviz-sam3 HTTP boundary.

Both the server and the adapter-side client import these types so the
field names match. The server lives in this package's Python 3.12 venv;
the client uses an identical schema implemented with the standard
library in :mod:`emboviz.perturb._target_detection` (so adapter venvs
don't have to add pydantic as a dependency).

Mask encoding: we use **COCO RLE** (compressed run-length encoding).
A 480×640 boolean mask is ~300 KB raw; COCO RLE typically compresses
it to under 5 KB, which keeps HTTP round-trips fast enough that the
sidecar is not the bottleneck.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    """One detection request: image + a text concept to find.

    The image is sent as raw bytes via multipart/form-data on the
    HTTP layer (FastAPI's ``File``); this model is only the JSON
    metadata payload, so it does NOT carry the bytes.
    """

    target_text: str = Field(
        ..., min_length=1, max_length=512,
        description="The concept phrase to localize (e.g. 'the mug').",
    )
    score_threshold: float = Field(
        0.30, ge=0.0, le=1.0,
        description="Drop instances with score below this.",
    )
    mask_threshold: float = Field(
        0.50, ge=0.0, le=1.0,
        description="Per-pixel mask logit cutoff for binarization.",
    )


class CocoRLE(BaseModel):
    """A COCO-format run-length encoded mask.

    The ``counts`` field is the compressed RLE string (a bytes object
    encoded as latin-1 so it survives JSON). ``size`` is ``[H, W]``.
    Decoded via ``pycocotools.mask.decode``.
    """

    size: list[int] = Field(..., min_length=2, max_length=2)
    counts: str


class Instance(BaseModel):
    """One detected instance of the requested concept."""

    bbox: list[int] = Field(
        ..., min_length=4, max_length=4,
        description="(x0, y0, x1, y1) pixel coords, exclusive end.",
    )
    score: float = Field(..., ge=0.0, le=1.0)
    mask: CocoRLE


class DetectResponse(BaseModel):
    """The full detection result for one (image, text) query.

    Empty ``instances`` = no detection above threshold; the client must
    treat this as "couldn't test on this frame" rather than fabricate.
    """

    instances: list[Instance] = Field(default_factory=list)
    image_size: list[int] = Field(
        ..., min_length=2, max_length=2,
        description="(H, W) of the input image, for client-side mask reshape.",
    )
    label: str = Field(
        ..., description="The target_text echoed back for logging clarity.",
    )


class Health(BaseModel):
    """GET /health response — used by clients to wait for the model load."""

    alive: bool
    model_loaded: bool
    model_id: Optional[str] = None
    device: Optional[str] = None
    version: str
