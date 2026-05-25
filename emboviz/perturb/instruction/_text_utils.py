"""Small shared text helpers for instruction perturbers."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional

from emboviz.core.types import PerturbedScene, Scene


def replace_word(text: str, old: str, new: str) -> str:
    """Whole-word replace, case-insensitive, escapes regex specials in `old`."""
    return re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)


def make_perturbed_scene(
    scene: Scene,
    perturber_name: str,
    axis: str,
    variant_id: str,
    new_instruction: str,
    description: str = "",
    parameters: Optional[dict] = None,
) -> PerturbedScene:
    """Build a PerturbedScene with the instruction replaced.

    Observations are preserved (image, state, gripper, etc.); only the
    text changes. Scene is frozen so we use `dataclasses.replace`.
    """
    new_scene = replace(scene, instruction=new_instruction)
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or new_instruction,
    )
