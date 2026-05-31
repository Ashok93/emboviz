"""Adapter-side client for the LaMa worker.

Lives in the ``emboviz-lama`` package so anyone calling LaMa imports from
``emboviz_lama.client`` — the wire schema (method names + args shape) and
the typed Python API stay together. The class extends
:class:`emboviz.adapters.RpcClient` so transport plumbing (ZMQ DEALER,
msgpack framing, request IDs, timeouts, error frames) is shared with the
VLA and SAM 3 clients.

Usage::

    from emboviz_lama.client import LamaClient

    client = LamaClient()
    filled = client.inpaint(image_png_bytes, mask)   # mask: HxW uint8/bool
    # filled is an (H, W, 3) uint8 ndarray
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from emboviz_wire import RpcClient


class LamaClient(RpcClient):
    """Typed RPC client for the LaMa ZMQ worker.

    Inherits :meth:`request`, :meth:`ping`, :meth:`close` etc. from
    :class:`RpcClient` and adds the LaMa-specific typed methods.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        timeout_ms: int = 120_000,
    ):
        super().__init__("lama", endpoint=endpoint, timeout_ms=timeout_ms)

    def inpaint(self, image_bytes: bytes, mask: np.ndarray) -> np.ndarray:
        """Inpaint one image over a binary mask.

        Parameters
        ----------
        image_bytes
            PNG / JPEG bytes of the RGB image.
        mask
            ``H×W`` array; nonzero pixels are the region to fill.

        Returns
        -------
        ``(H, W, 3)`` uint8 ndarray — the original image with ONLY the
        masked region replaced by LaMa's inpainting.
        """
        result = self.request("inpaint", {
            "image_bytes": bytes(image_bytes),
            "mask": np.asarray(mask, dtype=np.uint8),
        })
        return np.asarray(result["image"], dtype=np.uint8)

    def health(self) -> dict[str, Any]:
        """Lightweight liveness + model-loaded probe."""
        return self.request("health")
