"""Preset probe specs + training entry points.

Each preset describes a target variable, the layer indices to probe, and
the data-collection function for assembling training samples. Train via
the CLI; runtime loading via `probes.store.load_probe`.
"""

from policylens.probes.presets.failure_predictor import (
    FAILURE_PROBE_NAME,
    label_frames_from_deviation,
)

__all__ = ["FAILURE_PROBE_NAME", "label_frames_from_deviation"]
