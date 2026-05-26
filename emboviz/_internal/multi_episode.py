"""Multi-episode aggregation.

Real users do not analyze ONE episode — they upload 5, 20, 100 episodes
(usually the failing ones) and want a *cross-episode pattern* alongside
the per-episode drill-downs.

This module:

  • Parses the user's ``--episodes`` argument ("0", "0,3,7", "0-5",
    "all") into a concrete list of episode indices.
  • Walks the existing per-episode runner (``run_story``) over each
    episode, capturing per-episode summary.json output paths.
  • Aggregates per-axis severity distributions across episodes into a
    single ``MultiEpisodeReport`` Finding the user can read at the top
    of the report.

Per-episode artifacts (summary.json, rollout.rrd) are kept on disk
unchanged. The aggregate sits next to them as ``aggregate.json`` +
``aggregate.md`` (Phase 9 will add HTML).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ────────────────────────────────────────────────────────────────────
# Episode range parsing
# ────────────────────────────────────────────────────────────────────

def parse_episode_spec(spec: str, n_episodes_available: Optional[int] = None) -> list[int]:
    """Parse ``--episodes`` argument into a sorted, deduplicated list of ints.

    Accepted forms:
      • ``"7"``           → [7]
      • ``"0,3,7"``       → [0, 3, 7]
      • ``"0-5"``         → [0, 1, 2, 3, 4, 5]   (inclusive)
      • ``"0,3-5,9"``     → [0, 3, 4, 5, 9]
      • ``"all"``         → range(0, n_episodes_available)  (requires arg)

    ``all`` requires ``n_episodes_available`` because we don't probe the
    dataset just to answer "how many." The caller passes the size in.

    Raises ValueError on malformed specs — we never silently coerce.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty episode spec")

    if spec == "all":
        if n_episodes_available is None:
            raise ValueError(
                "--episodes all requires the dataset's episode count; "
                "pass --episodes as a comma list or range, or use the "
                "dataset's list_episodes() before invocation."
            )
        return list(range(int(n_episodes_available)))

    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"episode range '{chunk}' has hi < lo")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


# ────────────────────────────────────────────────────────────────────
# Episode + multi-episode bundling
# ────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeReport:
    """One episode's worth of analysis — path-only, no in-memory blobs.

    The actual diagnostic numbers live in summary.json (per_axis findings,
    trajectory-level findings, calibration, failure moments). The report
    points at the on-disk artifacts so multi-episode aggregation doesn't
    have to hold every per-frame number for 100 episodes in RAM.
    """

    episode_idx: int
    out_dir: Path
    summary_path: Path
    rollout_rrd_path: Optional[Path] = None

    def load_summary(self) -> dict:
        return json.loads(self.summary_path.read_text())


@dataclass
class MultiEpisodeReport:
    """N EpisodeReports + the cross-episode aggregate."""

    model_id: str
    dataset_id: str
    episodes: list[EpisodeReport] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    aggregate_path: Optional[Path] = None


# ────────────────────────────────────────────────────────────────────
# Cross-episode aggregation
# ────────────────────────────────────────────────────────────────────

def aggregate_axis_across_episodes(
    summaries: list[dict],
    axis: str,
) -> Optional[dict]:
    """For one axis, roll up the per-episode trajectory findings.

    Returns a dict with:
      • n_episodes: how many had this axis
      • n_pass / n_moderate / n_critical / n_unknown:
        episode-level dominant-severity counts
      • mean_score_across_episodes
      • observed_summary: plain-English headline of the cross-episode pattern
    """
    rows: list[dict] = []
    for s in summaries:
        per_axis = s.get("per_axis", {})
        if axis not in per_axis:
            continue
        rows.append(per_axis[axis])

    if not rows:
        return None

    # Determine episode-level "dominant" severity. We use the trajectory
    # finding's raw_numbers if available (it has n_pass/n_critical/etc.);
    # otherwise we fall back to the per-axis mean severity.
    def _dominant(row: dict) -> str:
        finding = row.get("finding") or {}
        rn = finding.get("raw_numbers") or {}
        if rn:
            order = ["critical", "moderate", "info", "pass", "unknown"]
            counts = {k: rn.get(f"n_{k}", 0) for k in order}
            return max(order, key=lambda k: (counts[k], -order.index(k)))
        # Legacy fallback: peek at per-frame severities list
        sevs = row.get("severities") or []
        if not sevs:
            return "unknown"
        from collections import Counter
        return Counter(sevs).most_common(1)[0][0]

    dominants = [_dominant(r) for r in rows]
    from collections import Counter
    sev_counter = Counter(dominants)
    n_eps = len(rows)
    mean_score = (
        sum(r.get("mean_score", float("nan")) for r in rows if r.get("mean_score") is not None) / n_eps
        if n_eps else float("nan")
    )

    # Pull a representative per-episode finding to ground the meaning
    rep_axis_finding = None
    for sev in ("critical", "moderate", "info", "pass", "unknown"):
        for r in rows:
            if _dominant(r) == sev:
                rep_axis_finding = r.get("finding")
                break
        if rep_axis_finding is not None:
            break

    parts = []
    for sev in ("critical", "moderate", "info", "pass", "unknown"):
        if sev_counter[sev]:
            parts.append(f"{sev_counter[sev]}/{n_eps} episode(s) {sev}")
    observed = (
        f"On `{axis}` across {n_eps} episode(s): "
        + ", ".join(parts) + "."
    )
    if rep_axis_finding:
        observed += " Representative: " + (rep_axis_finding.get("observed") or "")[:200]

    return {
        "axis":               axis,
        "n_episodes":         n_eps,
        "episode_dominant_severity_counts": dict(sev_counter),
        "mean_score_across_episodes": mean_score,
        "observed":           observed,
        "meaning":            rep_axis_finding.get("meaning") if rep_axis_finding else None,
        "next_step":          rep_axis_finding.get("next_step") if rep_axis_finding else None,
    }


def aggregate_episodes(reports: Iterable[EpisodeReport]) -> dict:
    """Build the cross-episode aggregate dict from a list of EpisodeReports."""
    reports = list(reports)
    if not reports:
        return {"n_episodes": 0, "axes": {}, "messages": []}

    summaries = [r.load_summary() for r in reports]
    all_axes: set[str] = set()
    for s in summaries:
        all_axes.update((s.get("per_axis") or {}).keys())

    axes_agg: dict[str, dict] = {}
    for axis in sorted(all_axes):
        rolled = aggregate_axis_across_episodes(summaries, axis)
        if rolled is not None:
            axes_agg[axis] = rolled

    # Episode-level model + dataset metadata sanity check.
    model_ids = {s.get("model") for s in summaries if s.get("model")}
    sources   = {s.get("trajectory_source", "").split(":")[0] for s in summaries}
    messages: list[str] = []
    if len(model_ids) > 1:
        messages.append(
            f"⚠ Multiple model_ids across episodes: {sorted(model_ids)}. "
            "Aggregate may mix policies."
        )
    if len(sources) > 1:
        messages.append(
            f"⚠ Multiple dataset sources across episodes: {sorted(sources)}."
        )

    return {
        "n_episodes": len(reports),
        "episode_indices": [r.episode_idx for r in reports],
        "model_ids":  sorted(model_ids),
        "dataset_sources": sorted(sources),
        "axes":       axes_agg,
        "messages":   messages,
    }


def write_aggregate_markdown(aggregate: dict, model_id: str, out_path: Path) -> Path:
    """Render the cross-episode aggregate as a markdown report."""
    lines: list[str] = [
        f"# Emboviz aggregate report — {model_id}",
        "",
        f"- **Episodes analyzed**: {aggregate['n_episodes']}",
        f"- **Episode indices**: {aggregate.get('episode_indices', [])}",
        "",
    ]
    for msg in aggregate.get("messages", []):
        lines.append(f"> {msg}")
    if aggregate.get("messages"):
        lines.append("")

    lines += ["## Cross-episode findings (per axis)", ""]
    axes = aggregate.get("axes") or {}
    if not axes:
        lines.append("_(no axes produced verdicts across these episodes)_")
    else:
        for axis, agg in axes.items():
            lines += [
                f"### `{axis}`",
                "",
                f"- **Observed**: {agg.get('observed','')}",
            ]
            if agg.get("meaning"):
                lines.append(f"- **Meaning**: {agg['meaning']}")
            if agg.get("next_step"):
                lines.append(f"- **Next step**: {agg['next_step']}")
            counts = agg.get("episode_dominant_severity_counts") or {}
            lines.append(f"- **Severity counts** (per episode): `{counts}`")
            ms = agg.get("mean_score_across_episodes")
            if ms is not None:
                lines.append(f"- **Mean score across episodes**: `{ms}`")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path
