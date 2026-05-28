"""Adapter-side client for the SAM 3 worker.

Lives in the ``emboviz-sam3`` package so anyone calling SAM 3 imports
from ``emboviz_sam3.client`` — the wire schema (method names + args
shape) and the typed Python API stay together. The class extends
:class:`emboviz.adapters.RpcClient` so transport plumbing (ZMQ DEALER,
msgpack framing, request IDs, timeouts, error frames) is shared with
the VLA clients.

Usage::

    from emboviz_sam3.client import Sam3Client

    client = Sam3Client()
    result = client.detect(image_bytes, target_text="the mug")
    for inst in result["instances"]:
        bbox, score, mask = inst["bbox"], inst["score"], inst["mask"]
        ...
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from emboviz.adapters import RpcClient


class Sam3Client(RpcClient):
    """Typed RPC client for the SAM 3 ZMQ worker.

    Inherits :meth:`request`, :meth:`ping`, :meth:`close` etc. from
    :class:`RpcClient` and adds the SAM 3-specific typed methods.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        timeout_ms: int = 120_000,
    ):
        super().__init__("sam3", endpoint=endpoint, timeout_ms=timeout_ms)

    def detect(
        self,
        image_bytes: bytes,
        target_text: str,
        *,
        score_threshold: float = 0.30,
        mask_threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Run one (image, text) concept segmentation.

        Returns ``{"instances": [...], "image_size": [H, W], "label": ...}``
        with each instance dict carrying ``"bbox"``, ``"score"``, and
        ``"mask"`` (a uint8 ndarray of shape ``(H, W)``).
        """
        result = self.request("detect", {
            "image_bytes":     bytes(image_bytes),
            "target_text":     str(target_text),
            "score_threshold": float(score_threshold),
            "mask_threshold":  float(mask_threshold),
        })
        # Defensive: ensure each mask comes back as a uint8 ndarray
        # regardless of how msgpack-numpy reconstructed it.
        for inst in result.get("instances") or []:
            m = inst.get("mask")
            if m is not None:
                inst["mask"] = np.asarray(m, dtype=np.uint8)
        return result

    def health(self) -> dict[str, Any]:
        """Lightweight liveness + model-loaded probe."""
        return self.request("health")
