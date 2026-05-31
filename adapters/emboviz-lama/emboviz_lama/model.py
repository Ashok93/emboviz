"""big-lama TorchScript inpainting (worker side).

Loads a TorchScript export of LaMa (Suvorov et al., *Resolution-robust
Large Mask Inpainting with Fourier Convolutions*, WACV 2022 —
arXiv:2109.07161) and fills a masked region with plausible background.
LaMa is the on-manifold third fill of emboviz's memorization diagnostic
(LITERATURE.md §1).

Weights / provenance
--------------------
The default checkpoint is the TorchScript ``big-lama.pt`` hosted at
``okaris/big-lama`` on the HuggingFace Hub, pinned to an immutable commit.
okaris authored the original simple-LaMa wrapper; this is the canonical
TorchScript export of the ``advimman/lama`` big-lama weights, Apache-2.0
by derivation. ``hf_hub_download`` caches it under ``~/.cache/huggingface``.

Override the source, in order of precedence:
  • ``model_path=`` kwarg / ``EMBOVIZ_LAMA_MODEL`` env → a local ``.pt``.
  • ``repo_id=`` / ``revision=`` kwargs (or ``EMBOVIZ_LAMA_REPO`` /
    ``EMBOVIZ_LAMA_REVISION`` env) → a different HF TorchScript export.

Preprocessing
-------------
``_get_image`` / ``_ceil_modulo`` / ``_pad_img_to_modulo`` /
``_prepare_img_and_mask`` below are vendored VERBATIM from
``simple-lama-inpainting`` (Apache-2.0,
https://github.com/enesmsahin/simple-lama-inpainting/blob/main/simple_lama_inpainting/utils/util.py),
itself derived from ``advimman/lama``. We vendor rather than depend on the
package because it pins ``Pillow<10``, which conflicts with
``emboviz-wire``'s ``Pillow>=10``. Vendoring also lets us fix two things
the upstream ``__call__`` does not do, both required for an honest
diagnostic intervention:

  1. **Crop the mod-8 symmetric padding back to the original H×W.** The
     upstream wrapper returns the padded image for non-mod-8 inputs.
  2. **Composite the model output onto the original ONLY within the
     mask.** LaMa reconstructs the whole frame; we keep every non-target
     pixel byte-identical to the input and paste LaMa's content into the
     hole alone. The intervention then changes the target region and
     nothing else — the "remove only the target" contract the
     memorization diagnostic requires (and the same per-mask-only
     semantics as the channel-mean and Gaussian-blur fills).
"""

from __future__ import annotations

import io
import logging
import os
import threading
from typing import Any, Optional

import numpy as np
from PIL import Image


log = logging.getLogger("emboviz_lama")

# Default checkpoint: the canonical TorchScript big-lama export, pinned to
# an immutable commit so the bytes are reproducible (never a floating
# branch / release URL). Apache-2.0 by derivation from advimman/lama.
DEFAULT_LAMA_REPO = "okaris/big-lama"
DEFAULT_LAMA_FILE = "big-lama.pt"
DEFAULT_LAMA_REVISION = "a77c4957376bb29a47a3339283477d4b31748b68"


# ──────────────────────────────────────────────────────────────────────
# Vendored preprocessing — simple-lama-inpainting (Apache-2.0), derived
# from advimman/lama. Verbatim except for omitting the cv2-backed
# ``scale_image`` path (we never scale, so the cv2 dependency is dropped).
# ──────────────────────────────────────────────────────────────────────

def _get_image(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def _ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def _pad_img_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    channels, height, width = img.shape
    out_height = _ceil_modulo(height, mod)
    out_width = _ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def _prepare_img_and_mask(image, mask, device, pad_out_to_modulo: int = 8):
    import torch

    out_image = _get_image(image)
    out_mask = _get_image(mask)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = _pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = _pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


# ──────────────────────────────────────────────────────────────────────
# The worker model.
# ──────────────────────────────────────────────────────────────────────

class LamaInpaintModel:
    """Wraps the big-lama TorchScript model behind a clean ``inpaint``.

    torch is imported lazily so importing this module stays cheap in the
    user's main venv during entry-point discovery; the heavy load happens
    on first construction inside the isolated LaMa runtime venv.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        device: str = "auto",
        preload: bool = True,
    ):
        # Resolution: explicit local path → env local path → HF download.
        self.model_path = model_path or os.environ.get("EMBOVIZ_LAMA_MODEL")
        self.repo_id = repo_id or os.environ.get("EMBOVIZ_LAMA_REPO", DEFAULT_LAMA_REPO)
        self.filename = filename or os.environ.get("EMBOVIZ_LAMA_FILE", DEFAULT_LAMA_FILE)
        self.revision = revision or os.environ.get(
            "EMBOVIZ_LAMA_REVISION", DEFAULT_LAMA_REVISION,
        )
        self._device_pref = device
        self._model = None
        self._device: Optional[str] = None
        self._resolved_path: Optional[str] = None
        self._lock = threading.Lock()
        if preload:
            self._load()

    # ----- model lifecycle ------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        return self._device

    def _resolve_checkpoint(self) -> str:
        """Return a local path to the TorchScript ``.pt``, downloading the
        pinned HF checkpoint if no explicit local path was given."""
        if self.model_path:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"EMBOVIZ_LAMA_MODEL / model_path points at "
                    f"{self.model_path!r}, which does not exist."
                )
            return self.model_path
        from huggingface_hub import hf_hub_download

        log.info(
            "fetching LaMa checkpoint %s/%s@%s from the HuggingFace Hub",
            self.repo_id, self.filename, self.revision[:12],
        )
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            revision=self.revision,
        )

    def _load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            import torch

            if self._device_pref in (None, "auto"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device_pref

            path = self._resolve_checkpoint()
            log.info("loading LaMa TorchScript model from %s on %s", path, device)
            model = torch.jit.load(path, map_location=device)
            model.eval()
            model = model.to(device)

            self._model = model
            self._device = device
            self._resolved_path = path
            self._self_test()
            log.info("LaMa ready on device=%s", self._device)

    def _self_test(self) -> None:
        """Run one tiny forward to validate the TorchScript export takes
        ``(image, mask)`` and returns an image of the expected shape.

        This is the one thing we cannot verify offline — that a given
        export's ``forward`` signature matches the ``(image, mask)`` ->
        ``(1, 3, H, W)`` contract. We catch a mismatch HERE, loudly, at
        worker startup, rather than mid-analysis as a confusing failure.
        """
        probe = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 1
        try:
            out = self._inpaint_array(probe, mask)
        except Exception as e:  # noqa: BLE001 — re-raise with guidance
            raise RuntimeError(
                f"LaMa self-test forward FAILED for checkpoint "
                f"{self._resolved_path!r}: {type(e).__name__}: {e}. The "
                "TorchScript export may have a different forward signature "
                "than the (image, mask) contract emboviz expects. Point "
                "EMBOVIZ_LAMA_MODEL at a compatible big-lama.pt (e.g. the "
                "simple-lama-inpainting release export) or set "
                "EMBOVIZ_LAMA_REPO / EMBOVIZ_LAMA_REVISION."
            ) from e
        if out.shape != (16, 16, 3):
            raise RuntimeError(
                f"LaMa self-test returned shape {out.shape}, expected "
                f"(16, 16, 3). Checkpoint {self._resolved_path!r} is not "
                "the expected big-lama image export."
            )

    def close(self) -> None:
        try:
            del self._model
        finally:
            self._model = None
            self._device = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ----- core inpainting ------------------------------------------------

    def _inpaint_array(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint ``image`` (H×W×3 uint8) over ``mask`` (H×W, nonzero =
        fill), returning H×W×3 uint8 with ONLY the masked pixels changed."""
        import torch

        assert self._model is not None
        H, W = image.shape[:2]
        mask_bool = np.asarray(mask) > 0
        # 255-valued single-channel uint8 mask is what the vendored
        # ``_get_image`` -> ``(>0)`` path expects. Build it explicitly as
        # uint8 (np.where keeps the dtype unambiguous for Image.fromarray,
        # which rejects anything but uint8 in mode "L").
        mask_img = Image.fromarray(
            np.where(mask_bool, np.uint8(255), np.uint8(0)), mode="L",
        )
        pil_img = Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8), mode="RGB")

        img_t, mask_t = _prepare_img_and_mask(pil_img, mask_img, self._device)
        with torch.inference_mode():
            out = self._model(img_t, mask_t)

        # (1, 3, Hp, Wp) in [0, 1] -> Hp×Wp×3 uint8 (matches LaMa's
        # predict.py / simple-lama denorm), then crop the mod-8 padding.
        arr = out[0].permute(1, 2, 0).detach().cpu().numpy()
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if arr.shape[0] < H or arr.shape[1] < W:
            raise RuntimeError(
                f"LaMa output {arr.shape[:2]} is smaller than the input "
                f"{(H, W)}; cannot crop to the original size."
            )
        arr = arr[:H, :W, :]

        # Composite: keep every non-target pixel byte-identical to the
        # input; paste LaMa's reconstruction into the masked hole only.
        composite = image.copy()
        composite[mask_bool] = arr[mask_bool]
        return composite

    def inpaint(self, image_bytes: bytes, mask: np.ndarray) -> dict[str, Any]:
        """Inpaint a PNG/JPEG-encoded image over a binary mask.

        Parameters
        ----------
        image_bytes
            PNG / JPEG bytes of the RGB image. Decoded with PIL.
        mask
            ``H×W`` array; nonzero pixels are the region to fill. Must
            match the image's height and width.

        Returns
        -------
        ``{"image": np.ndarray (H, W, 3) uint8, "image_size": [H, W]}`` —
        the original image with ONLY the masked region replaced by LaMa's
        inpainting.
        """
        if not image_bytes:
            raise ValueError("LamaInpaintModel.inpaint: empty image bytes")
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.asarray(pil, dtype=np.uint8)
        H, W = image.shape[:2]

        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
            mask_arr = mask_arr[..., 0]
        if mask_arr.shape[:2] != (H, W):
            raise ValueError(
                f"LamaInpaintModel.inpaint: mask shape {mask_arr.shape} does "
                f"not match image {(H, W)}. The caller must mask at the "
                "image's pixel grid."
            )
        if not (mask_arr > 0).any():
            # Nothing to fill — returning the original would be a silent
            # no-op intervention. The host gates on mask presence before
            # calling, so reaching here is a contract violation.
            raise ValueError(
                "LamaInpaintModel.inpaint: mask is empty (no pixels to "
                "fill). The memorization diagnostic must only request a "
                "fill for a non-empty detected mask."
            )

        self._load()
        out = self._inpaint_array(image, mask_arr)
        return {"image": out, "image_size": [H, W]}

    # ----- introspection --------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "checkpoint": self._resolved_path,
            "model_loaded": self.loaded,
            "device": self._device,
        }
