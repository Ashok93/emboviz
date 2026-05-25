"""Canonical VLA failure-mode catalog.

Derived from the cross-paper consensus:
  • LIBERO-Plus (arXiv 2510.13626)
  • LIBERO-Pro (arXiv 2510.03827)
  • COLOSSEUM (arXiv 2402.08191)
  • Eva-VLA (arXiv 2509.18953)
  • "Robust Skills, Brittle Grounding" (arXiv 2602.24143)
  • "When Vision Overrides Language" (arXiv 2602.17659)
  • IGAR (arXiv 2603.06001)
  • BYOVLA (arXiv 2410.01971)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    LANGUAGE = "language"
    VISION = "vision"
    INTERNAL = "internal"   # mechanistic / hidden-state failures


@dataclass(frozen=True)
class FailureMode:
    code: str                  # canonical short code, e.g. "language.noun_swap"
    name: str
    modality: Modality
    description: str
    references: tuple[str, ...]


FAILURE_MODES: dict[str, FailureMode] = {
    # ---- LANGUAGE ----
    "language.noun_swap": FailureMode(
        code="language.noun_swap",
        name="Noun-swap blindness",
        modality=Modality.LANGUAGE,
        description=(
            "Action stays nearly identical when the instruction noun is replaced "
            "by another within-category noun (spoon → fork). Model is reading "
            "vision, not language."
        ),
        references=("LIBERO-CF 2602.17659", "LIBERO-Plus 2510.13626"),
    ),
    "language.preposition_swap": FailureMode(
        code="language.preposition_swap",
        name="Spatial preposition blindness",
        modality=Modality.LANGUAGE,
        description=(
            "Model doesn't follow direction/relation changes "
            "(under/on/behind/left/right)."
        ),
        references=("What's Up? 2310.19785", "GSR-Bench 2406.13246"),
    ),
    "language.color_swap": FailureMode(
        code="language.color_swap",
        name="Color-attribute binding failure",
        modality=Modality.LANGUAGE,
        description=(
            "Model picks the wrong-color object when scene has multiple colored "
            "candidates."
        ),
        references=("LIBERO-CF 2602.17659", "Confusion Benchmark"),
    ),
    "language.count_swap": FailureMode(
        code="language.count_swap",
        name="Count / ordinal blindness",
        modality=Modality.LANGUAGE,
        description=(
            "Model ignores count words (one / two / the second / the smaller)."
        ),
        references=("LIBERO-CF 2602.17659",),
    ),
    "language.negation": FailureMode(
        code="language.negation",
        name="Negation blindness",
        modality=Modality.LANGUAGE,
        description=(
            "Model executes 'don't pick X' the same as 'pick X'."
        ),
        references=("VLA hallucination studies",),
    ),
    "language.refusal_absent": FailureMode(
        code="language.refusal_absent",
        name="Failure to refuse on absent object",
        modality=Modality.LANGUAGE,
        description=(
            "Instruction names an object that is not in the scene; model "
            "executes a grasp on whatever is available."
        ),
        references=("IVA / Do What? 2508.16292",),
    ),

    # ---- VISION ----
    "vision.occlusion": FailureMode(
        code="vision.occlusion",
        name="Occlusion sensitivity",
        modality=Modality.VISION,
        description=(
            "Action degrades sharply when the target object is partially hidden."
        ),
        references=("Eva-VLA 2509.18953", "COLOSSEUM 2402.08191"),
    ),
    "vision.viewpoint": FailureMode(
        code="vision.viewpoint",
        name="Viewpoint brittleness",
        modality=Modality.VISION,
        description=(
            "Small camera-pose perturbations collapse performance. The single "
            "largest documented VLA failure axis."
        ),
        references=("LIBERO-Plus 2510.13626", "AnyCamVLA"),
    ),
    "vision.lighting": FailureMode(
        code="vision.lighting",
        name="Lighting / colour-shift sensitivity",
        modality=Modality.VISION,
        description=(
            "Gamma / colour-temperature changes degrade recognition and grasp."
        ),
        references=("COLOSSEUM 2402.08191", "Eva-VLA 2509.18953"),
    ),
    "vision.distractor": FailureMode(
        code="vision.distractor",
        name="Distractor sensitivity",
        modality=Modality.VISION,
        description=(
            "Adding visually-similar distractor objects to the scene breaks "
            "target selection."
        ),
        references=("COLOSSEUM 2402.08191", "NICE 2511.22777"),
    ),
    "vision.memorization": FailureMode(
        code="vision.memorization",
        name="Trajectory memorization",
        modality=Modality.VISION,
        description=(
            "Removing the target object from the scene; the model still "
            "executes a coherent trajectory toward where the object usually is. "
            "Indicates the policy memorized motion, not vision-conditioned action."
        ),
        references=("LIBERO-Pro 2510.03827",),
    ),
    "vision.binding_grounding": FailureMode(
        code="vision.binding_grounding",
        name="Cross-modal binding failure",
        modality=Modality.VISION,
        description=(
            "Attention from the named noun lands outside the actual object in "
            "the scene."
        ),
        references=("Q-GroundCAM 2404.19128", "Few-Heads Grounding 2503.06287"),
    ),
    "vision.scene_sensitivity": FailureMode(
        code="vision.scene_sensitivity",
        name="Per-region scene sensitivity",
        modality=Modality.VISION,
        description=(
            "BYOVLA-style: masking irrelevant regions of the scene changes the "
            "action — model is reading background or distractor cues."
        ),
        references=("BYOVLA 2410.01971",),
    ),

    # ---- INTERNAL ----
    "internal.probe_decodable_but_ignored": FailureMode(
        code="internal.probe_decodable_but_ignored",
        name="Information present but unused",
        modality=Modality.INTERNAL,
        description=(
            "A linear probe on the model's hidden states recovers e.g. object "
            "color with >0.9 accuracy, yet the predicted action is invariant to "
            "color. The model 'sees' but doesn't 'act' on the information."
        ),
        references=("Probing OpenVLA 2502.04558", "Seeing but Not Believing 2510.17771"),
    ),
}
