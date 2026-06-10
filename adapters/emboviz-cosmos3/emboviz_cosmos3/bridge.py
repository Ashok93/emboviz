"""Cosmos action bridge — encode a policy's chunk as Cosmos conditioning.

The model-agnostic half of the bridge — integrating a policy's action chunk
into a Cartesian state sequence under a declared convention, and the stateful
closed-loop trackers — lives in :mod:`emboviz_wire.policy_bridge` (shared by
every world-model adapter). This module owns only the Cosmos-specific step:
encoding an integrated state sequence into the ``droid_lerobot`` normalized
action representation via :func:`emboviz_cosmos3.domains.encode_droid_states`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from emboviz_wire.policy_bridge import (
    ActionConvention,
    integrate_policy_chunk,
)
from emboviz_wire.types import Scene, Trajectory

from emboviz_cosmos3.domains import encode_droid_states


def policy_chunk_to_cosmos(
    seed_state_xyz_euler: np.ndarray,
    chunk: np.ndarray,
    convention: ActionConvention,
) -> np.ndarray:
    """Convert a policy action chunk into Cosmos ``droid_lerobot`` conditioning.

    Returns ``(T, 10)`` normalized actions ready for forward dynamics.
    """
    states, grippers = integrate_policy_chunk(seed_state_xyz_euler, chunk, convention)
    return encode_droid_states(states, grippers)


def policy_action_source(
    predict_fn: Callable[[Scene], "object"],
    *,
    convention: ActionConvention,
) -> Callable[[Trajectory, int, int], np.ndarray]:
    """Build a stress-test action source driven by the user's policy.

    ``predict_fn`` maps a :class:`Scene` to an ``ActionResult`` (e.g. a connected
    ``VLAModel`` client's ``predict``). The source runs the policy on the seed
    frame, takes the first ``n_actions`` rows of its action chunk, and bridges
    them into Cosmos conditioning via ``convention``. The seed frame must carry
    end-effector state (the integration anchor); a missing chunk or state raises.
    """

    def source(traj: Trajectory, seed_index: int, n_actions: int) -> np.ndarray:
        seed = traj.frames[seed_index]
        if seed.observations.state is None:
            raise ValueError(
                f"policy bridge needs end-effector state at seed frame {seed_index} "
                "to integrate the policy's actions, but observations.state is None."
            )
        result = predict_fn(seed)
        chunk = getattr(result, "action_chunk", None)
        if chunk is None:
            raise ValueError(
                "policy bridge needs a multi-step action_chunk to roll a critical "
                "moment, but the policy returned a single action (action_chunk is "
                "None). Use a chunk-predicting policy, or render a single step."
            )
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[0] < n_actions:
            raise ValueError(
                f"policy predicted a chunk of {chunk.shape[0]} steps but the stress "
                f"rollout needs {n_actions}; lower --n-actions or use a longer chunk."
            )
        seed_state = np.asarray(seed.observations.state.values, dtype=np.float32)
        return policy_chunk_to_cosmos(seed_state, chunk[:n_actions], convention)

    return source


__all__ = [
    "policy_action_source",
    "policy_chunk_to_cosmos",
]
