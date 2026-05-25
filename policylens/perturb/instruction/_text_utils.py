"""Small shared text helpers for instruction perturbers."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Optional

from policylens.core.types import PerturbedScene, Scene


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
    """Convenience: build a PerturbedScene with the new instruction."""
    new_scene = Scene(
        image=scene.image,
        instruction=new_instruction,
        metadata=scene.metadata,
        scene_id=scene.scene_id,
    )
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or new_instruction,
    )
