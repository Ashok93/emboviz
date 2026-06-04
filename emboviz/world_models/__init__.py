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
    TrustAnalysis,
    analyze_trust,
    reanchored_rollout,
    rollout_episode,
    summarize,
    trust_report,
)
from emboviz.world_models.keyframes import (
    CriticalWindow,
    Keyframe,
    critical_windows,
    detect_keyframes,
)
from emboviz.world_models.simulate import (
    DreamRollout,
    closed_loop_rollout,
)
from emboviz.world_models.stress import (
    StressClip,
    recorded_action_source,
    stress_test,
)
from emboviz.world_models.viz import (
    frames_to_arrays,
    save_frame_comparison,
    save_trust_curve,
    save_video,
)

__all__ = [
    # trust
    "FrameMetric",
    "TrustResult",
    "action_dependence",
    "compute_trust_curve",
    "frame_divergence",
    # rollout orchestration
    "TrustAnalysis",
    "analyze_trust",
    "reanchored_rollout",
    "rollout_episode",
    "summarize",
    "trust_report",
    # critical-moment keyframes + stress test
    "CriticalWindow",
    "Keyframe",
    "StressClip",
    "critical_windows",
    "detect_keyframes",
    "recorded_action_source",
    "stress_test",
    # closed-loop simulator
    "DreamRollout",
    "closed_loop_rollout",
    # rendering
    "frames_to_arrays",
    "save_frame_comparison",
    "save_trust_curve",
    "save_video",
]
