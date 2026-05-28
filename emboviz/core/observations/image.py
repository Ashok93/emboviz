"""RGB image observation, keyed by camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RGBImage:
    """One RGB frame from a named camera.

    `data` is a PIL.Image.Image at runtime; we avoid importing PIL here so
    this module imports cleanly without it. Adapters and exporters do the
    actual handling.
    """

    data: Any
    camera_id: str = "primary"
