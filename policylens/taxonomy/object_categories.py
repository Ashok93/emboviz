"""Object-category vocabulary for noun-swap perturbations and coverage analysis.

Each category is a closed set of words that share a referential role —
they can all be the manipulated object of a "pick the X" instruction. We
swap WITHIN a category to test binding ("spoon" → "fork", both utensils)
because cross-category swaps test grammar more than grounding.

Adding new objects: append to the category list. The perturber picks them
up automatically.
"""

from __future__ import annotations


OBJECT_CATEGORIES: dict[str, list[str]] = {
    "utensil": [
        "spoon", "fork", "knife", "spatula", "ladle",
        "chopsticks", "chopstick", "tongs", "whisk",
    ],
    "container": [
        "bowl", "plate", "cup", "mug", "basket", "tray",
        "dish", "pot", "pan", "container", "box", "jar",
        "glass", "bucket", "kettle",
    ],
    "food": [
        "apple", "banana", "orange", "carrot", "broccoli", "corn",
        "lemon", "potato", "tomato", "pear", "grape", "bread",
        "egg", "cucumber", "pepper", "mushroom", "onion", "lettuce",
    ],
    "toy": [
        "block", "cube", "ball", "sphere", "lego", "toy",
        "duck", "teddy", "doll", "ring",
    ],
    "cloth": [
        "cloth", "towel", "rag", "napkin", "sponge", "shirt", "sock",
    ],
    "tool": [
        "hammer", "screwdriver", "wrench", "pliers", "scissors",
    ],
}

# Categories that count as nouns for binding analysis (not attributes).
NOUN_CATEGORIES = list(OBJECT_CATEGORIES.keys())


# Inverse lookup
_WORD_TO_CAT = {w: c for c, words in OBJECT_CATEGORIES.items() for w in words}


def category_for_word(word: str) -> str | None:
    return _WORD_TO_CAT.get(word.lower().strip())


# Colour vocabulary — attribute, NOT a noun category. Used by ColorSwapPerturber.
COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange", "purple", "white",
    "black", "pink", "brown", "gray", "grey",
]


# Count / ordinal vocabulary — used by CountSwapPerturber.
COUNT_WORDS = ["one", "two", "three", "four", "five"]
ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "last"]
