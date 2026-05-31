"""Wire-method dispatcher for the LaMa worker.

Implements the :class:`emboviz.adapters.ServiceHandler` protocol: the
``methods`` property enumerates exactly which wire method names are
exposed. Anything not listed here is rejected by the server with a
``KeyError`` — no introspection, no implicit exposure.
"""

from __future__ import annotations

from typing import Any, Callable

from emboviz_lama.model import LamaInpaintModel


class LamaInpaintHandler:
    """Wraps :class:`~emboviz_lama.model.LamaInpaintModel` for ZMQ dispatch."""

    def __init__(self, model: LamaInpaintModel):
        self._m = model

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "LamaInpaintHandler":
        return cls(LamaInpaintModel(**kwargs))

    # ----- explicit dispatch table ----------------------------------------

    @property
    def methods(self) -> dict[str, Callable[[dict], Any]]:
        return {
            "inpaint": self._inpaint,
            "health": self._health,
        }

    # ----- handlers -------------------------------------------------------

    def _inpaint(self, args: dict) -> dict:
        return self._m.inpaint(
            image_bytes=args["image_bytes"],
            mask=args["mask"],
        )

    def _health(self, _: dict) -> dict:
        return self._m.health()

    # ----- teardown -------------------------------------------------------

    def close(self) -> None:
        self._m.close()
