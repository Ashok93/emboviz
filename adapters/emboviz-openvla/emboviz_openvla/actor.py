"""Ray actor wrapping the OpenVLA-7B model.

Subclasses :class:`emboviz.adapters.BaseAdapterActor` and delegates
:meth:`_build_model` to :mod:`emboviz_openvla.model`.

This module stays import-light: the only top-level imports are
:class:`BaseAdapterActor` (numpy-only) and stdlib. The heavy
``import torch / transformers`` happens inside
:class:`emboviz_openvla.model.OpenVLA.__init__` — which only runs
inside the runtime venv, never in core's main env.
"""

from __future__ import annotations

from typing import Any

from emboviz.adapters import BaseAdapterActor
from emboviz.models.protocol import VLAModel


class OpenVLAActor(BaseAdapterActor):
    """Ray actor that loads OpenVLA-7B inside its isolated venv."""

    def _build_model(self, **kwargs: Any) -> VLAModel:
        # Lazy import — emboviz_openvla.model pulls in torch +
        # transformers, which only exist in the runtime venv.
        from emboviz_openvla.model import OpenVLAAdapter
        return OpenVLAAdapter(**kwargs)
