"""Metric protocol — stateless, model-agnostic scalar producers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """Base class for all metrics.

    A Metric *may* take ActionResults, AttentionMaps, HiddenStates, or any
    other domain object; concrete Metric classes declare what they accept
    via their `compute` signature. Diagnostics know what Metrics they pair
    with — there is no Metric-side type discovery.
    """

    name: str

    @abstractmethod
    def compute(self, *args, **kwargs) -> float: ...
