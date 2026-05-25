"""Prompt-engineering sensitivity — paraphrase the instruction without
changing meaning. Tests whether the model is genuinely grounded on the
semantic content or just memorized specific surface phrasing.

If the action is invariant across paraphrases — fine, the model truly
understands the task. If the action drifts wildly between equivalent
prompts — the model is over-fit to surface form, deployment will be
brittle to operator phrasing.

Generates variants by rewriting articles, adding qualifiers, or
reordering clauses. Stays MEANING-PRESERVING (unlike NounSwap or
ColorSwap which change semantics).
"""

from __future__ import annotations

from typing import Iterable

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.instruction._text_utils import make_perturbed_scene


def _paraphrase(instruction: str) -> list[tuple[str, str]]:
    """Generate (variant_id, new_instruction) pairs.

    Conservative — only apply transformations that preserve meaning:
      • "the X" ↔ "a X"
      • "X" → "the X"
      • Add specifier "this" / "that"
      • Trailing politeness ("please")
      • Reorder clauses joined by "and"/"then"
    """
    s = instruction.strip().rstrip(".!?")
    variants: list[tuple[str, str]] = []
    lower = s.lower()
    words = s.split()

    # Toggle definite/indefinite article on the first article we find.
    for i, w in enumerate(words):
        wl = w.lower()
        if wl == "the":
            new_words = list(words)
            new_words[i] = "a"
            variants.append(("the_to_a", " ".join(new_words)))
            break
        if wl == "a":
            new_words = list(words)
            new_words[i] = "the"
            variants.append(("a_to_the", " ".join(new_words)))
            break

    # Add an explicit "this" / "that" if the instruction has a noun
    # following an article — adds specificity without changing meaning.
    for i, w in enumerate(words[:-1]):
        if w.lower() in {"the", "a"}:
            new_words = list(words)
            new_words[i] = "this"
            variants.append(("the_to_this", " ".join(new_words)))
            break

    # Add "please" prefix — pure politeness, semantic no-op.
    variants.append(("with_please", f"please {s.lower()}"))

    # Replace "pick up" with "grab" / "lift" (synonyms in most contexts).
    if "pick up" in lower:
        variants.append(("pickup_to_grab", s.lower().replace("pick up", "grab")))
        variants.append(("pickup_to_lift", s.lower().replace("pick up", "lift")))

    # Replace "place" with "put".
    if "place " in lower:
        variants.append(("place_to_put", s.lower().replace("place ", "put ")))

    # Reorder "X and Y" → "Y and X" (semantic-preserving when X and Y
    # are independent objects, which is the common case).
    if " and " in lower:
        parts = s.split(" and ", 1)
        if len(parts) == 2:
            variants.append(("reorder_and", f"{parts[1]} and {parts[0]}"))

    return variants


class PromptParaphrasePerturber(Perturber):
    """Meaning-preserving paraphrases of the instruction."""

    name = "prompt_paraphrase"
    axis = "language.prompt_paraphrase"
    affects = frozenset({"instruction"})

    def __init__(self, max_variants: int = 6):
        self.max_variants = max_variants

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        if not scene.instruction:
            return
        candidates = _paraphrase(scene.instruction)
        # Deduplicate by new_instruction (the paraphrase set can produce duplicates)
        seen = set()
        unique = []
        for vid, new_instr in candidates:
            key = new_instr.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append((vid, new_instr))
        for vid, new_instr in unique[: self.max_variants]:
            yield make_perturbed_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=vid,
                new_instruction=new_instr,
                description=f"\"{scene.instruction}\" → \"{new_instr}\"",
                parameters={"original": scene.instruction, "paraphrase_id": vid},
            )
