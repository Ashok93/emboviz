"""World-model-side analysis.

Where the worker contract :class:`emboviz_wire.world_model_protocol.WorldModel`
defines *how to obtain* a predicted rollout, this package is *what emboviz does
with it*: the **closed-loop dream** — fly a policy inside the world model from a
recorded episode's decisive moments, optionally with the seed scene edited
(object swapped or removed), and render reality next to the counterfactual.

This is deliberately separate from :mod:`emboviz.diagnostics` (which operate on
a :class:`VLAModel` policy via a fixed ``run(model, scene)`` contract). The dream
drives a :class:`emboviz_wire.world_model_protocol.WorldModel` from policy
actions, so it has its own small API.
"""

from emboviz.world_models.keyframes import (
    Keyframe,
    detect_keyframes,
)
from emboviz.world_models.simulate import (
    DreamRollout,
    closed_loop_rollout,
)
from emboviz.world_models.viz import (
    frames_to_arrays,
    save_video,
)

__all__ = [
    # critical-moment keyframes
    "Keyframe",
    "detect_keyframes",
    # closed-loop simulator
    "DreamRollout",
    "closed_loop_rollout",
    # rendering
    "frames_to_arrays",
    "save_video",
]
