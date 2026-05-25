"""Canonical taxonomies — failure modes, object categories, spatial relations.

These are PURE DATA. No model knowledge, no perturbation logic — just lists
and enums that the perturbers and diagnostics consult.
"""

from emboviz.taxonomy.failure_modes import FAILURE_MODES, FailureMode
from emboviz.taxonomy.object_categories import (
    NOUN_CATEGORIES,
    OBJECT_CATEGORIES,
    category_for_word,
)
from emboviz.taxonomy.spatial_prepositions import (
    PREPOSITION_PAIRS,
    SPATIAL_PREPOSITIONS,
)

__all__ = [
    "FAILURE_MODES",
    "FailureMode",
    "NOUN_CATEGORIES",
    "OBJECT_CATEGORIES",
    "category_for_word",
    "PREPOSITION_PAIRS",
    "SPATIAL_PREPOSITIONS",
]
