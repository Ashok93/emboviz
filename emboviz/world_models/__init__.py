"""World-model-side analysis.

Where the worker contract :class:`emboviz_wire.world_model_protocol.WorldModel`
defines *how to obtain* a predicted rollout, this package is *what emboviz does
with it*. The first capability is **trust calibration**: comparing a world
model's predicted rollout against the recorded episode it was conditioned on,
to measure how many frames the prediction can be trusted before it drifts.

This is deliberately separate from :mod:`emboviz.diagnostics` (which operate on
a :class:`VLAModel` policy via a fixed ``run(model, scene)`` contract). A
world-model trust analysis operates on two :class:`Trajectory` objects — a
predicted rollout and the real episode — so it has its own small API.
"""

from emboviz.world_models.trust import (
    FrameMetric,
    TrustResult,
    action_dependence,
    compute_trust_curve,
    frame_divergence,
)
from emboviz.world_models.rollout import (
    episode_actions,
    rollout_episode,
    summarize,
    trust_report,
)

__all__ = [
    # trust
    "FrameMetric",
    "TrustResult",
    "action_dependence",
    "compute_trust_curve",
    "frame_divergence",
    # rollout orchestration
    "episode_actions",
    "rollout_episode",
    "summarize",
    "trust_report",
]
