"""Metrics — compare two outputs and produce a scalar.

Metrics are stateless. They consume ActionResults, AttentionMaps, or
HiddenStates (depending on metric kind) and produce a float.
"""

from emboviz.metrics.action_divergence import ActionDivergenceMetric
from emboviz.metrics.ablation_drop import AblationDropMetric
from emboviz.metrics.attention_js import AttentionJSMetric
from emboviz.metrics.base import Metric
from emboviz.metrics.instruction_sensitivity import InstructionSensitivityMetric
from emboviz.metrics.pointing_game import PointingGameMetric
from emboviz.metrics.probe_confidence import ProbeConfidenceMetric

__all__ = [
    "Metric",
    "ActionDivergenceMetric",
    "AblationDropMetric",
    "AttentionJSMetric",
    "InstructionSensitivityMetric",
    "PointingGameMetric",
    "ProbeConfidenceMetric",
]
