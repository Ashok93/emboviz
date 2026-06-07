"""Wire-method dispatcher for the SD inpainting worker.

Implements the :class:`emboviz.adapters.ServiceHandler` protocol: the ``methods``
property enumerates exactly which wire method names are exposed. Anything not
listed is rejected by the server — no introspection, no implicit exposure.
"""

from __future__ import annotations

from typing import Any, Callable

from emboviz_sd_inpaint.model import SDInpaintModel


class SDInpaintHandler:
    """Wraps :class:`~emboviz_sd_inpaint.model.SDInpaintModel` for ZMQ dispatch."""

    def __init__(self, model: SDInpaintModel):
        self._m = model

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "SDInpaintHandler":
        return cls(SDInpaintModel(**kwargs))

    # ----- explicit dispatch table ----------------------------------------

    @property
    def methods(self) -> dict[str, Callable[[dict], Any]]:
        return {
            "fill": self._fill,
            "health": self._health,
        }

    # ----- handlers -------------------------------------------------------

    def _fill(self, args: dict) -> dict:
        return self._m.fill(
            image_bytes=args["image_bytes"],
            mask=args["mask"],
            prompt=args["prompt"],
            num_inference_steps=args.get("num_inference_steps"),
            guidance_scale=args.get("guidance_scale"),
            seed=args.get("seed", 0),
            negative_prompt=args.get("negative_prompt", ""),
        )

    def _health(self, _: dict) -> dict:
        return self._m.health()

    # ----- teardown -------------------------------------------------------

    def close(self) -> None:
        self._m.close()
