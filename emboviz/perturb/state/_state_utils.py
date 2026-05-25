"""Shared helpers for state-side perturbers.

Each builder rebuilds the Scene with one Observations field replaced.
We centralise this so the perturbers stay tiny and readable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from emboviz.core.observations import ActionHistory, GripperState, Proprioception
from emboviz.core.types import PerturbedScene, Scene


def make_perturbed_state_scene(
    scene: Scene,
    perturber_name: str,
    axis: str,
    variant_id: str,
    new_state: Proprioception,
    description: str = "",
    parameters: Optional[dict] = None,
) -> PerturbedScene:
    """Build a PerturbedScene with the Proprioception replaced."""
    new_obs = replace(scene.observations, state=new_state)
    new_scene = replace(scene, observations=new_obs)
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or f"{perturber_name}:{variant_id}",
    )


def make_perturbed_gripper_scene(
    scene: Scene,
    perturber_name: str,
    axis: str,
    variant_id: str,
    new_gripper: GripperState,
    description: str = "",
    parameters: Optional[dict] = None,
) -> PerturbedScene:
    """Build a PerturbedScene with the GripperState replaced."""
    new_obs = replace(scene.observations, gripper=new_gripper)
    new_scene = replace(scene, observations=new_obs)
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or f"{perturber_name}:{variant_id}",
    )


def make_perturbed_history_scene(
    scene: Scene,
    perturber_name: str,
    axis: str,
    variant_id: str,
    new_history: ActionHistory,
    description: str = "",
    parameters: Optional[dict] = None,
) -> PerturbedScene:
    """Build a PerturbedScene with the ActionHistory replaced."""
    new_obs = replace(scene.observations, action_history=new_history)
    new_scene = replace(scene, observations=new_obs)
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or f"{perturber_name}:{variant_id}",
    )
