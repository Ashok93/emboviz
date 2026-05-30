"""Wire-method dispatcher for the SAM 3 worker.

Implements the :class:`emboviz.adapters.ServiceHandler` protocol: the
``methods`` property enumerates exactly which wire method names are
exposed. Anything not listed here is rejected by the server with a
``KeyError`` — no introspection, no implicit exposure.
"""

from __future__ import annotations

from typing import Any, Callable

from emboviz_sam3.model import Sam3Detector


class Sam3DetectorHandler:
    """Wraps :class:`~emboviz_sam3.model.Sam3Detector` for ZMQ dispatch."""

    def __init__(self, detector: Sam3Detector):
        self._d = detector

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "Sam3DetectorHandler":
        return cls(Sam3Detector(**kwargs))

    # ----- explicit dispatch table ----------------------------------------

    @property
    def methods(self) -> dict[str, Callable[[dict], Any]]:
        return {
            "detect": self._detect,
            "health": self._health,
        }

    # ----- handlers -------------------------------------------------------

    def _detect(self, args: dict) -> dict:
        return self._d.detect(
            image_bytes=args["image_bytes"],
            target_text=args["target_text"],
            score_threshold=float(args.get("score_threshold", 0.30)),
            mask_threshold=float(args.get("mask_threshold", 0.50)),
        )

    def _health(self, _: dict) -> dict:
        return self._d.health()

    # ----- teardown -------------------------------------------------------

    def close(self) -> None:
        self._d.close()
