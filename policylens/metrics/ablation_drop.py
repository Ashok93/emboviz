"""Ablation-drop metric — action change caused by neuron ablation.

Used by neuron-ablation diagnostics and the BYOVLA-style sensitivity map
when the underlying intervention is `predict_with_neuron_ablation`.
"""

from __future__ import annotations

from policylens.core.types import ActionResult
from policylens.metrics.action_divergence import ActionDivergenceMetric
from policylens.metrics.base import Metric


class AblationDropMetric(Metric):
    """Returns ‖action_clean − action_ablated‖. Alias for ActionDivergenceMetric
    with semantic naming for ablation use cases."""

    name = "ablation_drop"

    def __init__(self, model=None):
        self.divergence = ActionDivergenceMetric(model=model)

    def compute(self, clean: ActionResult, ablated: ActionResult) -> float:
        return self.divergence.compute(clean, ablated)
