"""Stable Diffusion text-guided inpainting (worker side).

Loads a diffusers inpainting pipeline (default
``stabilityai/stable-diffusion-2-inpainting``) and regenerates a masked region of
an image to contain a described object. This is the object-INSERTION backend for
the dream scene swap (the counterpart to LaMa's object REMOVAL).

torch / diffusers are imported lazily so importing this module stays cheap in the
user's main venv during entry-point discovery; the heavy load happens on first
construction inside the isolated ``sd-inpaint`` runtime venv.

Mask convention (diffusers inpainting): WHITE mask pixels are repainted, BLACK
preserved. The caller passes a boolean ``H×W`` mask where ``True`` = the region
to fill; it is written as a white-on-black ``L`` image.

Honest-counterfactual contract: diffusers inpainting nominally preserves the
unmasked region, but the VAE round-trip perturbs it slightly. We therefore
composite the generated frame onto the original ONLY within the mask, so every
non-target pixel is byte-identical to the input — the same per-mask-only semantics
as the LaMa removal worker.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any, Optional

import numpy as np
from PIL import Image


log = logging.getLogger("emboviz_sd_inpaint")

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"


def _round_to_multiple(x: int, mod: int = 8) -> int:
    """Round ``x`` to the nearest positive multiple of ``mod`` (>= mod)."""
    return max(mod, int(round(x / mod)) * mod)


class SDInpaintModel:
    """Wraps a diffusers inpainting pipeline behind a clean ``fill``."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        revision: Optional[str] = None,
        device: str = "auto",
        preload: bool = True,
        inference_resolution: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        import os

        self.model_id = model_id or os.environ.get("EMBOVIZ_SD_INPAINT_MODEL", DEFAULT_MODEL_ID)
        self.revision = revision or os.environ.get("EMBOVIZ_SD_INPAINT_REVISION")
        self._device_pref = device
        self.inference_resolution = int(inference_resolution)
        self.default_steps = int(num_inference_steps)
        self.default_guidance = float(guidance_scale)
        self._pipe = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()
        if preload:
            self._load()

    # ----- model lifecycle ------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

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
            from diffusers import AutoPipelineForInpainting

            if self._device_pref in (None, "auto"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device_pref
            dtype = torch.float16 if device == "cuda" else torch.float32

            log.info("loading SD inpainting pipeline %s on %s (%s)", self.model_id, device, dtype)
            kwargs: dict[str, Any] = {"torch_dtype": dtype}
            if self.revision:
                kwargs["revision"] = self.revision
            pipe = AutoPipelineForInpainting.from_pretrained(self.model_id, **kwargs)
            # The diagnostic controls the prompt; the NSFW checker would blank
            # benign robot frames as false positives, so disable it explicitly.
            if getattr(pipe, "safety_checker", None) is not None:
                pipe.safety_checker = None
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)

            self._pipe = pipe
            self._device = device
            self._self_test()
            log.info("SD inpainting ready on device=%s", self._device)

    def _self_test(self) -> None:
        """One tiny forward to validate the pipeline takes (prompt, image,
        mask_image) and returns an image. Catches a bad checkpoint / API
        mismatch loudly at worker startup, not mid-analysis."""
        probe = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 16:48] = True
        try:
            out = self._fill_array(probe, mask, "a red cube", steps=2, guidance=1.0, seed=0)
        except Exception as e:  # noqa: BLE001 — re-raise with guidance
            raise RuntimeError(
                f"SD inpaint self-test FAILED for model {self.model_id!r}: "
                f"{type(e).__name__}: {e}. The checkpoint may not be a diffusers "
                "inpainting pipeline. Set model_id / EMBOVIZ_SD_INPAINT_MODEL to a "
                "valid inpainting checkpoint (e.g. stabilityai/stable-diffusion-2-"
                "inpainting)."
            ) from e
        if out.shape != (64, 64, 3):
            raise RuntimeError(
                f"SD inpaint self-test returned shape {out.shape}, expected (64, 64, 3)."
            )

    def close(self) -> None:
        try:
            del self._pipe
        finally:
            self._pipe = None
            self._device = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ----- core inpainting ------------------------------------------------

    def _fill_array(
        self, image: np.ndarray, mask: np.ndarray, prompt: str,
        *, steps: int, guidance: float, seed: int, negative_prompt: str = "",
    ) -> np.ndarray:
        """Insert ``prompt`` into ``image`` (H×W×3 uint8) over ``mask`` (H×W
        bool, True = fill), returning H×W×3 uint8 with ONLY the masked pixels
        changed.

        Generation runs at a model-friendly resolution (longest side =
        ``inference_resolution``, both dims multiples of 8, aspect preserved);
        the result is resized back and composited into the original-resolution
        mask so geometry and every non-target pixel are preserved.
        """
        import torch

        assert self._pipe is not None
        H, W = image.shape[:2]
        mask_bool = np.asarray(mask) > 0

        scale = self.inference_resolution / max(H, W)
        gh, gw = _round_to_multiple(int(H * scale)), _round_to_multiple(int(W * scale))

        img_small = Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8), mode="RGB").resize(
            (gw, gh), Image.BILINEAR
        )
        mask_small = Image.fromarray(
            np.where(mask_bool, np.uint8(255), np.uint8(0)), mode="L"
        ).resize((gw, gh), Image.NEAREST)

        generator = torch.Generator(device=self._device).manual_seed(int(seed))
        with torch.inference_mode():
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=img_small,
                mask_image=mask_small,
                height=gh,
                width=gw,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
            ).images[0]

        gen_full = np.asarray(result.convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.uint8)
        composite = image.copy()
        composite[mask_bool] = gen_full[mask_bool]
        return composite

    def fill(
        self, image_bytes: bytes, mask: np.ndarray, prompt: str,
        *, num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None, seed: int = 0,
        negative_prompt: str = "",
    ) -> dict[str, Any]:
        """Insert ``prompt`` into a PNG/JPEG-encoded image over a binary mask.

        Returns ``{"image": np.ndarray (H, W, 3) uint8, "image_size": [H, W]}`` —
        the original image with ONLY the masked region regenerated.
        """
        if not image_bytes:
            raise ValueError("SDInpaintModel.fill: empty image bytes")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("SDInpaintModel.fill: a non-empty prompt is required.")
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.asarray(pil, dtype=np.uint8)
        H, W = image.shape[:2]

        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
            mask_arr = mask_arr[..., 0]
        if mask_arr.shape[:2] != (H, W):
            raise ValueError(
                f"SDInpaintModel.fill: mask shape {mask_arr.shape} does not match "
                f"image {(H, W)}. The caller must mask at the image's pixel grid."
            )
        if not (mask_arr > 0).any():
            raise ValueError(
                "SDInpaintModel.fill: mask is empty (no pixels to fill). The caller "
                "must only request a fill for a non-empty detected mask."
            )

        self._load()
        out = self._fill_array(
            image, mask_arr, prompt,
            steps=int(num_inference_steps or self.default_steps),
            guidance=float(guidance_scale if guidance_scale is not None else self.default_guidance),
            seed=int(seed), negative_prompt=negative_prompt,
        )
        return {"image": out, "image_size": [H, W]}

    # ----- introspection --------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "model_loaded": self.loaded,
            "device": self._device,
            "inference_resolution": self.inference_resolution,
        }
