"""Instruction perturbers — change the text, leave the image alone.

Every perturber tests a specific linguistic axis of grounding.
"""

from emboviz.perturb.instruction.color_swap import ColorSwapPerturber
from emboviz.perturb.instruction.count_swap import CountSwapPerturber
from emboviz.perturb.instruction.empty import EmptyInstructionPerturber
from emboviz.perturb.instruction.negation import NegationPerturber
from emboviz.perturb.instruction.noun_swap import NounSwapPerturber
from emboviz.perturb.instruction.ood_task import OODTaskPerturber
from emboviz.perturb.instruction.preposition_swap import PrepositionSwapPerturber
from emboviz.perturb.instruction.refusal import RefusalPerturber

__all__ = [
    "ColorSwapPerturber",
    "CountSwapPerturber",
    "EmptyInstructionPerturber",
    "NegationPerturber",
    "NounSwapPerturber",
    "OODTaskPerturber",
    "PrepositionSwapPerturber",
    "RefusalPerturber",
]
