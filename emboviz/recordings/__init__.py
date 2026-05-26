"""Deployment-recording adapters — for replay of real-robot rollouts.

A *recording* is fundamentally different from a *dataset*:

  Training/eval datasets (emboviz.datasets)
    • Recorded BEFORE the model existed (human teleoperation)
    • Each step has an ``expert_action`` recorded by the demonstrator
    • Diagnostics that compare to a human (imitation L2) are meaningful
    • Examples: Bridge, LIBERO, DROID

  Deployment recordings (emboviz.recordings)
    • Recorded by RUNNING the user's trained model on a real robot
    • Each step has the MODEL'S predicted action (no human to compare to)
    • imitation_accuracy is N/A — there is no expert
    • Diagnostics ask "did the policy use vision / instruction / state
      / wrist cam to produce that action?" — answerable from re-running
      the model on the recorded observations under intervention
    • Examples: MCAP from ROS 2 logs, Rerun .rrd from rr.log() calls

Adapters here always set ``metadata["has_recorded_expert_action"] =
False`` so the runner's --show-imitation gate correctly suppresses the
BC validation metric on deployment data.

Available adapters (lazy):
  • MCAPRecording — ROS 2 / Isaac SIM default (extra: mcap)
  • RerunRecording — Rerun .rrd code-first logs (extra: rerun) — planned
"""

from __future__ import annotations

__all__ = [
    "MCAPRecording",
    "RerunRecording",
]


_LAZY: dict[str, tuple[str, str, str]] = {
    "MCAPRecording":   ("emboviz.recordings.mcap",  "MCAPRecording",  "mcap"),
    "RerunRecording":  ("emboviz.recordings.rerun", "RerunRecording", "rerun"),
}


def __getattr__(name: str):
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name, extra = entry
    try:
        import importlib
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"emboviz.recordings.{name} requires the '{extra}' extra. "
            f"Install with: pip install 'emboviz[{extra}]'. "
            f"Underlying error: {e}"
        ) from e
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
