"""Spatial preposition vocabulary for the PrepositionSwapPerturber."""

from __future__ import annotations


SPATIAL_PREPOSITIONS = [
    "on", "in", "under", "above", "below", "behind", "in front of",
    "next to", "beside", "between", "on top of", "underneath",
    "to the left of", "to the right of", "near", "far from",
]


# Opposing pairs for clean A↔B swaps (the most damning grounding test).
PREPOSITION_PAIRS: list[tuple[str, str]] = [
    ("on", "under"),
    ("on top of", "underneath"),
    ("above", "below"),
    ("in front of", "behind"),
    ("to the left of", "to the right of"),
    ("near", "far from"),
    ("inside", "outside"),
]
