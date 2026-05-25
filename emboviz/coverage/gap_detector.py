"""Map detected failure axes to actionable data-collection recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from emboviz.coverage.text_analyzer import CoverageSnapshot
from emboviz.taxonomy.object_categories import OBJECT_CATEGORIES


# Empirical thresholds tied to IGAR/LIBERO-PRO findings — below ~5 within-
# category pairs, models reliably exhibit noun-blindness.
CRITICAL_THRESHOLD = 5
MODERATE_THRESHOLD = 30


@dataclass
class CoverageGap:
    failure_axis: str           # e.g., "language.noun_swap on utensil"
    target_pattern: str
    observed_count: int
    total_episodes: int
    severity: str               # "critical" / "moderate" / "ok"
    recommendation: str
    details: dict = field(default_factory=dict)


@dataclass
class CoverageReport:
    snapshot: CoverageSnapshot
    gaps: list[CoverageGap]


def detect_gaps(
    snapshot: CoverageSnapshot,
    failing_axes: list[dict],
    critical_threshold: int = CRITICAL_THRESHOLD,
    moderate_threshold: int = MODERATE_THRESHOLD,
) -> CoverageReport:
    """`failing_axes` is a list of {'axis': 'noun_swap', 'category': 'utensil'} dicts."""
    gaps: list[CoverageGap] = []

    for axis in failing_axes:
        category = axis.get("category")
        if category is None or category not in snapshot.category_stats:
            continue
        cs = snapshot.category_stats[category]
        total_pairs = sum(cs.pair_counts.values())
        severity = (
            "critical" if total_pairs < critical_threshold else
            "moderate" if total_pairs < moderate_threshold else
            "ok"
        )
        all_objs = OBJECT_CATEGORIES[category]
        possible = {(a, b) for a in all_objs for b in all_objs if a < b}
        observed = set(cs.pair_counts.keys())
        missing = sorted(possible - observed)[:10]
        recommendation = _make_recommendation(category, severity, total_pairs, missing)
        gaps.append(CoverageGap(
            failure_axis=f"{axis.get('axis', 'noun_swap')} on {category}",
            target_pattern=f"episodes with ≥2 {category}s named in instruction",
            observed_count=total_pairs,
            total_episodes=snapshot.total_episodes,
            severity=severity,
            recommendation=recommendation,
            details={
                "distinct_objects_seen": len(cs.object_counts),
                "object_counts": dict(cs.object_counts.most_common()),
                "pairs_seen": [(list(p), c) for p, c in cs.pair_counts.most_common(5)],
                "pairs_missing_examples": [list(p) for p in missing],
            },
        ))

    return CoverageReport(snapshot=snapshot, gaps=gaps)


def _make_recommendation(category: str, severity: str, observed: int, missing_pairs) -> str:
    if severity == "critical":
        ex = " · ".join(f"{a}+{b}" for a, b in missing_pairs[:6]) or "(none observed)"
        return (
            f"CRITICAL gap. Your dataset has only {observed} demonstrations where two "
            f"{category}s appear in the same scene — below the empirical threshold ({CRITICAL_THRESHOLD}) "
            f"at which noun-grounding emerges (IGAR, arXiv 2603.06001).\n"
            f"  • Record 20–40 contrastive demos with the missing within-category pairs:\n"
            f"    {ex}\n"
            f"  • For each pair, vary which is the named target so the instruction is the\n"
            f"    only disambiguator.\n"
            f"  • Include 5–10 refusal demos where the named {category} is ABSENT and the\n"
            f"    correct action is search/stop, not grasp-anything."
        )
    if severity == "moderate":
        return (
            f"Moderate gap. Your dataset has {observed} {category}-pair scenes — some "
            f"grounding signal exists but the model isn't reliably exploiting it. "
            f"Recommend doubling coverage with additional contrastive demos, especially "
            f"pairs you currently see <3 times."
        )
    return (
        f"Coverage adequate ({observed} {category}-pair scenes in fine-tune set). "
        f"If you're still seeing noun-blindness on this category, the bottleneck is "
        f"likely model capacity / training schedule, not data coverage."
    )
