"""Text-based coverage analysis on a dataset's task descriptions."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from policylens.taxonomy.object_categories import OBJECT_CATEGORIES, category_for_word


@dataclass
class CategoryStats:
    category: str
    object_counts: Counter = field(default_factory=Counter)
    pair_counts: Counter = field(default_factory=Counter)

    @property
    def total_episodes_with_category(self) -> int:
        return sum(self.object_counts.values())


@dataclass
class CoverageSnapshot:
    """Per-dataset structured snapshot. Consumed by gap_detector."""

    dataset_name: str
    total_episodes: int
    category_stats: dict[str, CategoryStats]


def analyze_dataset_coverage(
    instructions: list[str],
    dataset_name: str = "fine-tune set",
) -> CoverageSnapshot:
    """Compute per-category object counts + within-category co-occurrence counts."""
    unique = list(dict.fromkeys(
        i.strip().lower() for i in instructions if i and i.strip()
    ))

    cats: dict[str, CategoryStats] = {
        c: CategoryStats(category=c) for c in OBJECT_CATEGORIES
    }

    for instr in unique:
        matched = _parse_instruction_nouns(instr)
        by_cat: dict[str, list[str]] = defaultdict(list)
        for word, c in matched:
            by_cat[c].append(word)
        for c, words in by_cat.items():
            for w in set(words):
                cats[c].object_counts[w] += 1
            uniq = sorted(set(words))
            for i, a in enumerate(uniq):
                for b in uniq[i + 1:]:
                    cats[c].pair_counts[(a, b)] += 1

    return CoverageSnapshot(
        dataset_name=dataset_name,
        total_episodes=len(unique),
        category_stats=cats,
    )


def _parse_instruction_nouns(instruction: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for token in re.findall(r"\b[a-zA-Z]+\b", instruction.lower()):
        cat = category_for_word(token)
        if cat is not None:
            matches.append((token, cat))
    return matches
