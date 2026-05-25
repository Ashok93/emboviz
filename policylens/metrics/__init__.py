"""Metrics — compare two outputs and produce a scalar.

Metrics are stateless. They consume ActionResults, AttentionMaps, or
HiddenStates (depending on metric kind) and produce a float.
"""

from policylens.metrics.action_divergence import ActionDivergenceMetric
from policylens.metrics.ablation_drop import AblationDropMetric
from policylens.metrics.attention_js import AttentionJSMetric
from policylens.metrics.base import Metric
from policylens.metrics.instruction_sensitivity import InstructionSensitivityMetric
from policylens.metrics.pointing_game import PointingGameMetric
from policylens.metrics.probe_confidence import ProbeConfidenceMetric

__all__ = [
    "Metric",
    "ActionDivergenceMetric",
    "AblationDropMetric",
    "AttentionJSMetric",
    "InstructionSensitivityMetric",
    "PointingGameMetric",
    "ProbeConfidenceMetric",
]
