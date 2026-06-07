"""Adapter-side client for the SD inpainting worker.

Lives in the ``emboviz-sd-inpaint`` package so anyone calling it imports from
``emboviz_sd_inpaint.client`` — the wire schema (method names + args shape) and
the typed Python API stay together. The class extends
:class:`emboviz.adapters.RpcClient` so transport plumbing (ZMQ DEALER, msgpack
framing, request IDs, timeouts, error frames) is shared with the VLA, SAM 3, and
LaMa clients.

Usage::

    from emboviz_sd_inpaint.client import SDInpaintClient

    client = SDInpaintClient()
    filled = client.fill(image_png_bytes, mask, "a spoon")   # mask: HxW uint8/bool
    # filled is an (H, W, 3) uint8 ndarray
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from emboviz_wire import RpcClient


class SDInpaintClient(RpcClient):
    """Typed RPC client for the SD inpainting ZMQ worker."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        timeout_ms: int = 300_000,
    ):
        super().__init__("sd-inpaint", endpoint=endpoint, timeout_ms=timeout_ms)

    def fill(
        self,
        image_bytes: bytes,
        mask: np.ndarray,
        prompt: str,
        *,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 0,
        negative_prompt: str = "",
    ) -> np.ndarray:
        """Insert ``prompt`` into the masked region of one image.

        Parameters
        ----------
        image_bytes
            PNG / JPEG bytes of the RGB image.
        mask
            ``H×W`` array; nonzero pixels are the region to regenerate.
        prompt
            Text description of the object to paint into the masked region.

        Returns
        -------
        ``(H, W, 3)`` uint8 ndarray — the original image with ONLY the masked
        region regenerated.
        """
        args: dict[str, Any] = {
            "image_bytes": bytes(image_bytes),
            "mask": np.asarray(mask, dtype=np.uint8),
            "prompt": str(prompt),
            "seed": int(seed),
            "negative_prompt": str(negative_prompt),
        }
        if num_inference_steps is not None:
            args["num_inference_steps"] = int(num_inference_steps)
        if guidance_scale is not None:
            args["guidance_scale"] = float(guidance_scale)
        result = self.request("fill", args)
        return np.asarray(result["image"], dtype=np.uint8)

    def health(self) -> dict[str, Any]:
        """Lightweight liveness + model-loaded probe."""
        return self.request("health")
