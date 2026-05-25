"""Linear probes over VLA hidden states.

A probe is a small classifier (logistic regression by default) trained to
decode some target variable — object color, target presence, action phase,
success/failure — from the model's internal hidden states. Probe accuracy
× action-invariance gives the 'information present but unused' diagnostic.

Probes are *trained offline* once per (model, target) and stored to disk.
At inference, `ProbeDiagnostic` loads a stored probe and runs it on the
scene's hidden states.
"""

from emboviz.probes.base import LinearProbe, ProbeSpec
from emboviz.probes.store import load_probe, save_probe
from emboviz.probes.trainer import train_linear_probe

__all__ = [
    "LinearProbe",
    "ProbeSpec",
    "load_probe",
    "save_probe",
    "train_linear_probe",
]
