"""Inpainting — the shared "fill a masked region with plausible content"
interface used by the memorization diagnostic's on-manifold fill.

The memorization diagnostic masks the manipulated target and measures how
much the policy's action changes. Two of its fills (``channel_mean``,
``gaussian_blur``) are pure-numpy and OOD-leaning; the third,
``lama_inpaint``, fills the hole with plausible background so the
agreement gate spans the on-manifold/OOD axis (LITERATURE.md §1).

LaMa needs torch, so it runs in an isolated ZeroMQ worker
(``emboviz-lama``) exactly like SAM 3. This module is the host-side
facade — it never imports torch.

Two layers, mirroring ``_target_detection``:

  • :class:`LamaInpainter` — a ZMQ client to the worker. One inpaint per
    call; raises a clear, actionable error if the worker isn't reachable.
  • :class:`CachingInpainter` — memoizes by an explicit key so the
    diagnostic and the Rerun-overlay reconstruction share one forward
    pass per (frame, camera), the same way ``CachingTargetDetector``
    memoizes detections.
"""

from __future__ import annotations

import io
from typing import Optional, Protocol

import numpy as np


# Hashable key callers attach to a result so the caching layer can dedupe
# the diagnostic's fill and the Rerun overlay's reconstruction. The
# memorization paths key on ``(scene_id, camera)`` — the inpaint of a
# given camera frame is deterministic in (image, mask), so that identity
# is a sound cache key.
InpaintKey = tuple


class Inpainter(Protocol):
    """Fills a masked region of an image with plausible content.

    Contract: ``image`` is an ``H×W×3`` uint8 RGB array; ``mask`` is an
    ``H×W`` boolean/uint8 array (nonzero = the region to fill). Returns an
    ``H×W×3`` uint8 array identical to ``image`` everywhere EXCEPT the
    masked region, which is replaced by the fill. ``key`` is an optional
    cache identity (ignored by stateless implementations).
    """

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray, *, key: Optional[InpaintKey] = None,
    ) -> np.ndarray: ...


class LamaInpainter:
    """On-manifold inpainting via LaMa (big-lama), over the ZMQ wire.

    Thin wrapper around :class:`emboviz_lama.client.LamaClient`. The actual
    LaMa model runs in a SEPARATE venv (the ``emboviz-lama`` worker) and
    answers ZMQ ``inpaint`` requests; this side never imports torch.

    The worker returns the original image with ONLY the masked region
    replaced (it composites internally), so the per-mask-only semantics
    match the ``channel_mean`` / ``gaussian_blur`` fills.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """Args:
            endpoint: ZMQ endpoint of the running ``emboviz-lama`` worker.
                Default: read from ``EMBOVIZ_LAMA_ENDPOINT`` env var, else
                ``ipc://~/.emboviz/sockets/lama.sock``.
            timeout: per-request RPC timeout in seconds. LaMa inference is
                ~1 s on CPU, sub-second on GPU; the first request to a
                freshly started worker pays the load + self-test cost
                unless it was started with ``--preload`` (the default).
        """
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self._client = None
        self._health_checked = False

    # -- low-level ZMQ helpers -----------------------------------------

    def _zmq(self):
        if self._client is not None:
            return self._client
        try:
            from emboviz_lama.client import LamaClient
        except ImportError as e:
            raise ImportError(
                "LamaInpainter requires the ``emboviz-lama`` adapter package "
                "(it ships the typed RPC client alongside the worker code). "
                "It ships with emboviz core, so if it's missing your install "
                "is incomplete — reinstall from the repo root with:\n"
                "    uv sync"
            ) from e
        self._client = LamaClient(
            endpoint=self.endpoint,
            timeout_ms=int(self.timeout * 1000),
        )
        return self._client

    def _check_health(self) -> None:
        """First-call probe: confirm the worker is reachable and emit a
        clear, actionable error if it isn't. We do not auto-spawn the
        worker here — the analyze runner brings it up the same way it
        brings up SAM 3 (connect → auto-install → auto-spawn)."""
        if self._health_checked:
            return
        client = self._zmq()
        if not client.ping(timeout_ms=2000):
            raise RuntimeError(
                f"LamaInpainter cannot reach the LaMa worker at "
                f"{client._endpoint}.\n\n"
                "Start the worker (in its own venv):\n"
                "    ~/.emboviz/venvs/lama/bin/emboviz-lama serve\n\n"
                "Or override the endpoint via "
                "EMBOVIZ_LAMA_ENDPOINT=ipc://... or tcp://...\n\n"
                "`emboviz analyze` builds and spawns this worker automatically; "
                "to pre-build its isolated venv yourself, run:\n"
                "    uv run emboviz install-lama\n\n"
                "If you don't want the on-manifold inpaint fill, drop "
                "'lama_inpaint' from analysis.fills (the channel_mean + "
                "gaussian_blur fills need no worker)."
            )
        self._health_checked = True

    # -- public Inpainter contract -------------------------------------

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray, *, key: Optional[InpaintKey] = None,
    ) -> np.ndarray:
        self._check_health()
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"LamaInpainter.inpaint expects an HxWx3 RGB uint8 image; "
                f"got shape {arr.shape}."
            )
        # PNG (lossless) so the fill isn't computed against JPEG artifacts.
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        return self._zmq().inpaint(buf.getvalue(), np.asarray(mask))


class CachingInpainter:
    """Wraps any :class:`Inpainter` and memoizes by explicit key.

    The memorization diagnostic computes the fill, and the runner's
    Rerun-overlay collection reconstructs the same masked image; without
    caching we'd pay a second LaMa forward per (frame, camera). The cache
    key is supplied by the caller (``(scene_id, camera)``) — identity, not
    image content — so re-requesting the same frame returns the cached
    fill. Mirrors :class:`~emboviz.perturb._target_detection.CachingTargetDetector`.
    """

    def __init__(self, base: Inpainter):
        self._base = base
        self._cache: dict[InpaintKey, np.ndarray] = {}

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray, *, key: Optional[InpaintKey] = None,
    ) -> np.ndarray:
        if key is None:
            # No identity to cache against — compute without storing.
            return self._base.inpaint(image, mask)
        if key in self._cache:
            return self._cache[key]
        out = self._base.inpaint(image, mask)
        self._cache[key] = out
        return out

    def lookup(self, key: InpaintKey) -> Optional[np.ndarray]:
        """Read-only access to a cached fill (``None`` if not computed)."""
        return self._cache.get(key)

    def clear(self) -> None:
        """Drop all cached fills (e.g. between episodes)."""
        self._cache.clear()


class ObjectInserter(Protocol):
    """Regenerates a masked region of an image to contain a described object.

    The text-guided counterpart to :class:`Inpainter` (which only removes/fills
    with background). Contract: ``image`` is an ``H×W×3`` uint8 RGB array;
    ``mask`` is ``H×W`` boolean/uint8 (nonzero = the region to regenerate);
    ``prompt`` names the object to paint in. Returns an ``H×W×3`` uint8 array
    identical to ``image`` everywhere EXCEPT the masked region.
    """

    def insert(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray: ...


class SDInpaintInserter:
    """Text-guided object insertion via Stable Diffusion inpainting, over ZMQ.

    Thin host-side facade over :class:`emboviz_sd_inpaint.client.SDInpaintClient`.
    The diffusers pipeline runs in the SEPARATE ``emboviz-sd-inpaint`` worker
    venv (torch + diffusers); this side never imports either. Mirrors
    :class:`LamaInpainter`.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 300.0,
        *,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 0,
        negative_prompt: str = "",
    ):
        """Args:
            endpoint: ZMQ endpoint of the running ``emboviz-sd-inpaint`` worker.
                Default: ``EMBOVIZ_SD_INPAINT_ENDPOINT`` env var, else
                ``ipc://~/.emboviz/sockets/sd-inpaint.sock``.
            timeout: per-request RPC timeout in seconds. A full diffusion pass
                is several seconds on GPU; the first request to a freshly
                started worker also pays the model load unless preloaded.
            num_inference_steps, guidance_scale, seed, negative_prompt:
                generation settings forwarded per call (None -> the worker's
                defaults).
        """
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = int(seed)
        self.negative_prompt = negative_prompt
        self._client = None
        self._health_checked = False

    def _zmq(self):
        if self._client is not None:
            return self._client
        try:
            from emboviz_sd_inpaint.client import SDInpaintClient
        except ImportError as e:
            raise ImportError(
                "SDInpaintInserter requires the ``emboviz-sd-inpaint`` adapter "
                "package (it ships the typed RPC client alongside the worker "
                "code). It ships with emboviz core, so if it's missing your "
                "install is incomplete — reinstall from the repo root with:\n"
                "    uv sync"
            ) from e
        self._client = SDInpaintClient(
            endpoint=self.endpoint, timeout_ms=int(self.timeout * 1000),
        )
        return self._client

    def _check_health(self) -> None:
        if self._health_checked:
            return
        client = self._zmq()
        if not client.ping(timeout_ms=2000):
            raise RuntimeError(
                f"SDInpaintInserter cannot reach the SD inpaint worker at "
                f"{client._endpoint}.\n\n"
                "Start the worker (in its own venv):\n"
                "    ~/.emboviz/venvs/sd-inpaint/bin/emboviz-sd-inpaint serve\n\n"
                "Or override the endpoint via EMBOVIZ_SD_INPAINT_ENDPOINT=ipc://... "
                "or tcp://...\n\n"
                "The dream driver builds and spawns this worker automatically; to "
                "pre-build its isolated venv yourself, run:\n"
                "    emboviz install-sd-inpaint"
            )
        self._health_checked = True

    def insert(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        self._check_health()
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"SDInpaintInserter.insert expects an HxWx3 RGB uint8 image; "
                f"got shape {arr.shape}."
            )
        import io as _io
        from PIL import Image
        buf = _io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        return self._zmq().fill(
            buf.getvalue(), np.asarray(mask), prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed, negative_prompt=self.negative_prompt,
        )


__all__ = [
    "Inpainter", "InpaintKey", "LamaInpainter", "CachingInpainter",
    "ObjectInserter", "SDInpaintInserter",
]
