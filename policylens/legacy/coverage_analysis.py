"""Module C — Coverage analysis on the user's fine-tune dataset.

Given a detected failure axis (e.g., noun-swap blindness on utensils), look
at the training set's task descriptions and answer the actionable question:

  *"How many of your demos contain ≥2 utensils in the same scene?
    Zero → that's why your model can't disambiguate."*

This is the closed-loop bit — the bit that turns "your model is broken" into
"record THESE demos."

We do it text-only on the dataset's tasks table because:
  • It's instant (no image embedding needed).
  • Bridge / Open X-Embodiment task strings already encode object identities.
  • The user's recommendation is at the dataset-design level (record more
    contrastive scenarios), not pixel-level.

For a fully visual coverage analysis we'd embed scene images and cluster —
that's a v2 feature.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from policylens.instruction_perturb import OBJECT_CATEGORIES, parse_instruction_nouns


@dataclass
class CategoryStats:
    """Per-category dataset distribution."""

    category: str
    object_counts: Counter             # word → number of episodes mentioning it
    pair_counts: Counter               # (word_a, word_b) → episodes with BOTH

    @property
    def total_episodes_with_category(self) -> int:
        return sum(self.object_counts.values())


@dataclass
class CoverageGap:
    """One actionable gap finding."""

    failure_axis: str                 # "noun_swap on utensil", etc.
    target_pattern: str               # what we counted
    observed_count: int               # how many demos match
    total_episodes: int               # denominator
    severity: str                     # "critical" / "moderate" / "ok"
    recommendation: str               # what to record
    details: dict = field(default_factory=dict)


@dataclass
class CoverageReport:
    dataset_name: str
    total_episodes: int
    category_stats: dict[str, CategoryStats]
    gaps: list[CoverageGap]


def analyze_dataset_coverage(
    instructions: list[str],
    dataset_name: str = "fine-tune set",
) -> CoverageReport:
    """Build per-category coverage stats from a flat list of instruction strings.

    For Bridge, instructions come from `meta/tasks.jsonl` via the LeRobot
    dataset metadata. We deduplicate identical task descriptions (Bridge
    has many duplicates).
    """
    unique_instructions = list(dict.fromkeys(i.strip().lower() for i in instructions if i.strip()))

    # Per-category accumulation.
    cats: dict[str, CategoryStats] = {
        c: CategoryStats(category=c, object_counts=Counter(), pair_counts=Counter())
        for c in OBJECT_CATEGORIES
    }

    for instr in unique_instructions:
        matched = parse_instruction_nouns(instr)
        by_cat: dict[str, list[str]] = defaultdict(list)
        for word, c in matched:
            by_cat[c].append(word)

        for c, words in by_cat.items():
            for w in set(words):
                cats[c].object_counts[w] += 1
            # Co-occurrence within category — the bit that matters for binding.
            uniq = sorted(set(words))
            for i, a in enumerate(uniq):
                for b in uniq[i + 1:]:
                    cats[c].pair_counts[(a, b)] += 1

    return CoverageReport(
        dataset_name=dataset_name,
        total_episodes=len(unique_instructions),
        category_stats=cats,
        gaps=[],   # filled by detect_gaps()
    )


def detect_gaps(
    report: CoverageReport,
    failing_axes: list[dict],
    critical_threshold: int = 5,
    moderate_threshold: int = 30,
) -> list[CoverageGap]:
    """Produce gap findings tied to specific failure axes.

    `failing_axes` is a list of dicts the diagnostic produces, e.g.:
        [{"axis": "noun_swap", "category": "utensil", "noun": "spoon"}]

    For each: count co-occurrence demos of that category in the training set,
    flag severity, write a templated recommendation. Severity thresholds are
    rules-of-thumb tied to IGAR / LIBERO-PRO findings — under ~5 contrastive
    demos per pair, models reliably exhibit noun blindness.
    """
    gaps: list[CoverageGap] = []

    for axis in failing_axes:
        category = axis.get("category")
        if category is None:
            continue
        cat_stats = report.category_stats.get(category)
        if cat_stats is None:
            continue

        # Count co-occurrences for this category.
        total_pairs = sum(cat_stats.pair_counts.values())
        n_distinct_objects = len(cat_stats.object_counts)
        n_pairs_observed = len(cat_stats.pair_counts)

        severity = (
            "critical" if total_pairs < critical_threshold else
            "moderate" if total_pairs < moderate_threshold else
            "ok"
        )

        # Top pairs the dataset DOES have.
        top_pairs = cat_stats.pair_counts.most_common(5)
        # Compute pairs the dataset is missing (vs all possible within category).
        all_objects = OBJECT_CATEGORIES[category]
        possible = {(a, b) for a in all_objects for b in all_objects if a < b}
        observed = set(cat_stats.pair_counts.keys())
        missing = sorted(possible - observed)[:10]

        recommendation = _make_recommendation(category, severity, total_pairs, missing)

        gaps.append(CoverageGap(
            failure_axis=f"noun_swap on {category}",
            target_pattern=f"episodes with ≥2 {category}s named in instruction",
            observed_count=total_pairs,
            total_episodes=report.total_episodes,
            severity=severity,
            recommendation=recommendation,
            details={
                "distinct_objects_seen": n_distinct_objects,
                "object_counts": dict(cat_stats.object_counts.most_common()),
                "pairs_seen": [(list(p), c) for p, c in top_pairs],
                "pairs_missing_examples": [list(p) for p in missing],
            },
        ))

    report.gaps = gaps
    return gaps


def _make_recommendation(category: str, severity: str, observed: int, missing_pairs) -> str:
    if severity == "critical":
        examples = " · ".join(f"{a}+{b}" for a, b in missing_pairs[:6]) or "(no within-category pairs in dataset)"
        return (
            f"CRITICAL gap. Your fine-tune set contains only {observed} demonstrations "
            f"where two {category}s appear in the same scene. This is below the "
            f"empirical threshold (5) at which noun-grounding emerges (IGAR, "
            f"arXiv 2603.06001). Concrete fix:\n"
            f"  • Record 20–40 contrastive demos with the missing within-category pairs:\n"
            f"    {examples}\n"
            f"  • For each pair, vary the named target between the two co-present objects\n"
            f"    so the instruction is the only disambiguator.\n"
            f"  • Include 5–10 'refusal' demos where the named {category} is ABSENT\n"
            f"    and the correct behavior is search/stop, not grasp-anything."
        )
    if severity == "moderate":
        return (
            f"Moderate gap. Your dataset has {observed} {category}-pair scenes — "
            f"some grounding signal exists but the model isn't reliably exploiting it. "
            f"Recommend doubling coverage with additional contrastive demos, "
            f"especially pairs you currently see <3 times."
        )
    return (
        f"Coverage adequate ({observed} {category}-pair scenes in fine-tune set). "
        f"If you're still seeing noun-blindness on this category, the bottleneck is "
        f"likely model capacity / training schedule, not data coverage."
    )


def render_coverage_report(report: CoverageReport, out_path: Path) -> None:
    """Text dump — readable Markdown summary of the coverage report."""
    lines = [
        f"# Coverage Report — {report.dataset_name}",
        "",
        f"**Total unique instructions analysed**: {report.total_episodes}",
        "",
        "## Per-category coverage",
        "",
    ]
    for cat, stats in report.category_stats.items():
        if stats.total_episodes_with_category == 0:
            continue
        lines.append(f"### {cat}")
        lines.append(
            f"- {len(stats.object_counts)} distinct objects, "
            f"{sum(stats.object_counts.values())} mentions across episodes."
        )
        lines.append(f"- Top objects: {dict(stats.object_counts.most_common(5))}")
        co = sum(stats.pair_counts.values())
        lines.append(f"- **Within-category co-occurrences (≥2 in one scene): {co} episodes**")
        if stats.pair_counts:
            lines.append("  - Top pairs:")
            for pair, c in stats.pair_counts.most_common(5):
                lines.append(f"    - `{pair[0]} + {pair[1]}` : {c} episodes")
        lines.append("")

    if report.gaps:
        lines.append("## Data-gap findings")
        lines.append("")
        for g in report.gaps:
            badge = {"critical": "🟥", "moderate": "🟧", "ok": "🟩"}[g.severity]
            lines.append(f"### {badge} {g.failure_axis}  —  severity: **{g.severity.upper()}**")
            lines.append(f"- Target pattern: {g.target_pattern}")
            lines.append(f"- Observed: **{g.observed_count}** demonstrations")
            lines.append(f"- Recommendation:")
            for ln in g.recommendation.splitlines():
                lines.append(f"  {ln}")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def collect_bridge_instructions(dataset) -> list[str]:
    """Best-effort: pull all unique instruction strings from a loaded LeRobotDataset."""
    meta_tasks = getattr(dataset.meta, "tasks", None)
    if meta_tasks is None:
        return []
    if isinstance(meta_tasks, dict):
        items = list(meta_tasks.values())
    else:
        items = list(meta_tasks)
    # Each item may be a string or a {'task': ...} dict.
    out = []
    for it in items:
        if isinstance(it, dict) and "task" in it:
            out.append(str(it["task"]))
        elif isinstance(it, str):
            out.append(it)
    return out
