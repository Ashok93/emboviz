"""Auto-generate counterfactual instruction variants per Bridge episode.

The sweep needs to construct *typed* perturbations per episode rather than a
hand-written list. We classify each object word in the instruction by
category (utensil, container, food, toy, ...) and swap within-category to
form a noun-swap variant. Then we add structural variants (verb swap,
direction reverse, empty).

This lets the diagnostic ask the *axis* question across many scenes:
  • Is noun-swap consistently the lowest-ISS variant?  ⇒ noun blindness
    is systematic.
  • Is direction-swap consistently followed?  ⇒ verb/spatial grounding
    works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Coarse object categories culled from Bridge / BridgeV2 task descriptions.
# Not exhaustive but covers the dominant nouns in the dataset.
OBJECT_CATEGORIES: dict[str, list[str]] = {
    "utensil": [
        "spoon", "fork", "knife", "spatula", "ladle", "chopsticks",
        "chopstick", "tongs",
    ],
    "container": [
        "bowl", "plate", "cup", "mug", "basket", "tray", "dish", "pot",
        "pan", "container", "box", "jar", "glass", "bucket",
    ],
    "food": [
        "apple", "banana", "orange", "carrot", "broccoli", "corn",
        "lemon", "potato", "tomato", "pear", "grape", "fruit", "vegetable",
        "bread", "egg",
    ],
    "toy": [
        "block", "cube", "ball", "sphere", "lego", "toy", "duck",
        "teddy", "doll",
    ],
    "cloth": [
        "cloth", "towel", "rag", "napkin", "sponge", "shirt", "sock",
    ],
    # Colors / modifiers — kept for reference but excluded from noun-binding
    # coverage analysis because they're attributes, not objects.
    # "color": [...]
}

# Categories that *count* as nouns for binding analysis (excludes attribute words).
NOUN_CATEGORIES = ["utensil", "container", "food", "toy", "cloth"]

# Inverse lookup: word → category.
_WORD_TO_CATEGORY: dict[str, str] = {
    word: cat for cat, words in OBJECT_CATEGORIES.items() for word in words
}


@dataclass
class Perturbation:
    """One counterfactual variant of an instruction."""

    axis: str             # "noun_swap", "verb_swap", "direction_swap", "empty", "ood_task"
    text: str             # the perturbed instruction
    swap_from: str = ""   # the original noun/verb being replaced (for noun_swap)
    swap_to: str = ""     # the replacement (for noun_swap)


@dataclass
class PerturbationSet:
    base_instruction: str
    target_noun: str | None              # the primary noun we're testing
    target_category: str | None
    perturbations: list[Perturbation] = field(default_factory=list)


def parse_instruction_nouns(instruction: str) -> list[tuple[str, str]]:
    """Return list of (word, category) for every recognized noun in the prompt."""
    matches: list[tuple[str, str]] = []
    for token in re.findall(r"\b[a-zA-Z]+\b", instruction.lower()):
        cat = _WORD_TO_CATEGORY.get(token)
        if cat is not None:
            matches.append((token, cat))
    return matches


def pick_target_noun(instruction: str) -> tuple[str | None, str | None]:
    """Pick the most likely 'thing-being-manipulated' noun from the instruction.

    Heuristic: prefer utensils > food > toy > container, since the
    manipulated object is usually one of the first three; containers are
    typically source/destination not target.
    """
    matches = parse_instruction_nouns(instruction)
    priority = ["utensil", "food", "toy", "cloth", "container"]
    for cat in priority:
        for word, c in matches:
            if c == cat:
                return word, c
    return None, None


def build_perturbations(
    instruction: str,
    n_noun_swaps: int = 1,
    include_structural: bool = True,
) -> PerturbationSet:
    """Build a typed set of counterfactual variants for one instruction."""
    target_noun, target_cat = pick_target_noun(instruction)
    perturbations: list[Perturbation] = []

    # 1. Noun swap: replace target noun with another word in same category.
    if target_noun is not None and target_cat is not None:
        candidates = [w for w in OBJECT_CATEGORIES[target_cat] if w != target_noun]
        for i in range(min(n_noun_swaps, len(candidates))):
            swap_to = candidates[i]
            perturbations.append(Perturbation(
                axis="noun_swap",
                text=_swap_word(instruction, target_noun, swap_to),
                swap_from=target_noun,
                swap_to=swap_to,
            ))

    if include_structural:
        # 2. Direction swap: rewrite "from X to Y" → "from Y to X" if pattern present.
        rev = _reverse_direction(instruction)
        if rev != instruction:
            perturbations.append(Perturbation(axis="direction_swap", text=rev))

        # 3. Verb swap to a different action.
        perturbations.append(Perturbation(axis="verb_swap", text="lift the basket"))

        # 4. Empty instruction — pure vision baseline.
        perturbations.append(Perturbation(axis="empty", text=""))

        # 5. OOD task — completely unrelated.
        perturbations.append(Perturbation(axis="ood_task", text="press the red button"))

    return PerturbationSet(
        base_instruction=instruction,
        target_noun=target_noun,
        target_category=target_cat,
        perturbations=perturbations,
    )


def _swap_word(text: str, a: str, b: str) -> str:
    return re.sub(rf"\b{re.escape(a)}\b", b, text, flags=re.IGNORECASE)


def _reverse_direction(text: str) -> str:
    """If instruction matches 'from <X> to <Y>', return 'from <Y> to <X>'."""
    m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s|$|[.,!?])", text, flags=re.IGNORECASE)
    if not m:
        return text
    x, y = m.group(1), m.group(2)
    return text.replace(f"from {x} to {y}", f"from {y} to {x}")
