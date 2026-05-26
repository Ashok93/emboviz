"""Per-episode + aggregate report generation (Markdown + HTML).

The runner persists ``summary.json`` per episode. This module turns those
JSON dicts into the human-readable artifacts users actually open:

  • Per-episode markdown — copy-paste-into-ticket version of the
    findings, plus the Rerun command.
  • Per-episode HTML — same content, browser-friendly, dark theme.
  • Aggregate HTML — cross-episode patterns + per-episode links.

The markdown variants are core (no extra deps). The HTML variants
require Jinja2 from the ``viz`` extra; calling them without it raises a
clean ImportError pointing at ``pip install 'emboviz[viz]'``.

We deliberately don't synthesize prose — every sentence comes from the
Finding objects the diagnostics emitted. The reporter is a layout +
formatting layer, not an interpretive layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# Mapping severity → CSS badge class fragment used in the HTML templates.
_BADGE_CLASS = {
    "critical": "crit",
    "moderate": "mod",
    "info":     "info",
    "pass":     "pass",
    "unknown":  "unk",
}


# ────────────────────────────────────────────────────────────────────
# Helpers — read findings out of a per-episode summary.json
# ────────────────────────────────────────────────────────────────────

def _severity_order(sev: str) -> int:
    """Worst → best ordering for sort. Higher = worse."""
    return {"critical": 4, "moderate": 3, "unknown": 2, "info": 1, "pass": 0}.get(sev, -1)


def _axis_finding_view(axis: str, row: dict) -> dict:
    """Translate a per-axis summary dict into the template-ready view."""
    finding = row.get("finding") or {}
    sev = row.get("severity", "unknown")
    return {
        "axis":       axis,
        "severity":   sev,
        "badge":      _BADGE_CLASS.get(sev, "unk"),
        "observed":   finding.get("observed") or row.get("explanation") or "",
        "meaning":    finding.get("meaning")  or "",
        "next_step":  finding.get("next_step") or "",
        "raw_numbers": finding.get("raw_numbers") or {},
        "scalar":     row.get("mean_score"),
    }


def _sorted_findings(per_axis: dict) -> list[dict]:
    findings = [_axis_finding_view(ax, row) for ax, row in per_axis.items()]
    findings.sort(key=lambda f: -_severity_order(f["severity"]))
    return findings


# ────────────────────────────────────────────────────────────────────
# Markdown — per episode (core; no extras needed)
# ────────────────────────────────────────────────────────────────────

def render_episode_markdown(summary: dict, *, rrd_path: Optional[str] = None) -> str:
    """Plain-English per-episode report, copy-paste-into-ticket style."""
    findings        = _sorted_findings(summary.get("per_axis") or {})
    not_applicable  = summary.get("not_applicable") or {}
    calibration     = summary.get("calibration") or {}
    failure_moments = summary.get("failure_moments") or []
    model_id        = summary.get("model", "?")
    traj_source     = summary.get("trajectory_source", "?")
    n_frames        = summary.get("n_frames", 0)
    instruction     = summary.get("instruction", "")

    lines: list[str] = [
        f"# Emboviz — episode report",
        "",
        f"- **Model**: `{model_id}`",
        f"- **Trajectory**: `{traj_source}` ({n_frames} frame(s))",
        f"- **Instruction**: `{instruction}`",
    ]
    if rrd_path:
        lines += [
            "",
            f"Scrub frame-by-frame in Rerun: `rerun {rrd_path}`",
        ]

    lines += ["", "## Findings (worst first)", ""]
    if not findings:
        lines.append("_(no diagnostics produced a verdict on this trajectory)_")
    for f in findings:
        lines += [
            f"### `{f['axis']}`",
            "",
            f"- **Observed**: {f['observed']}",
        ]
        if f["meaning"]:
            lines.append(f"- **Meaning**: {f['meaning']}")
        if f["next_step"]:
            lines.append(f"- **Next step**: {f['next_step']}")
        if f["raw_numbers"]:
            lines.append("- **Raw numbers**:")
            for k, v in f["raw_numbers"].items():
                lines.append(f"  - `{k}`: `{v}`")
        lines.append("")

    if not_applicable:
        lines += ["## Diagnostics that could not run", ""]
        for axis, why in not_applicable.items():
            lines += [f"- `{axis}` — {why}"]
        lines.append("")

    if failure_moments:
        lines += ["## Failure moments (≥2 critical axes per frame)", ""]
        for fm in failure_moments:
            lines.append(
                f"- frame **{fm.get('frame_idx')}**: "
                f"{fm.get('n_critical_axes', 0)} axis(es) — "
                f"{', '.join(fm.get('critical_axes', []))}"
            )
        lines.append("")

    lines += [
        "## Calibration",
        "",
        f"- noise floor = `{calibration.get('noise_floor', 0):.4f}`",
        f"- typical action magnitude = `{calibration.get('typical_action_magnitude', 0):.4f}`",
        f"- averaging samples per call = `{calibration.get('n_samples', 1)}`",
        "",
        "All Δaction values above are normalized: "
        "`(raw − noise_floor) / typical_action_magnitude`.",
    ]
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# HTML — per episode + aggregate (requires Jinja2 from viz extra)
# ────────────────────────────────────────────────────────────────────

def _jinja_env():
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as e:
        raise ImportError(
            "HTML report rendering needs the `viz` extra. Install with: "
            "pip install 'emboviz[viz]'. Underlying error: " + str(e)
        ) from e
    templates_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_episode_html(
    summary: dict, *, rrd_path: Optional[str] = None,
) -> str:
    env = _jinja_env()
    tpl = env.get_template("episode.html.j2")
    findings = _sorted_findings(summary.get("per_axis") or {})
    return tpl.render(
        episode_idx       = summary.get("frame_indices", [0])[0] if summary.get("frame_indices") else "?",
        model_id          = summary.get("model", "?"),
        trajectory_source = summary.get("trajectory_source", "?"),
        n_frames          = summary.get("n_frames", 0),
        instruction       = summary.get("instruction", ""),
        findings          = findings,
        not_applicable    = summary.get("not_applicable") or {},
        failure_moments   = summary.get("failure_moments") or [],
        calibration       = summary.get("calibration") or {},
        rrd_path          = rrd_path,
    )


def render_aggregate_html(
    aggregate: dict,
    *,
    model_id: str,
    episode_links: list[dict],
) -> str:
    """Render the cross-episode aggregate as an HTML page.

    ``episode_links`` is a list of dicts: ``{"idx": int, "html_rel": str,
    "summary_rel": str, "rrd_rel": str|None}`` describing per-episode
    artifacts relative to the aggregate HTML's location.
    """
    env = _jinja_env()
    tpl = env.get_template("aggregate.html.j2")
    return tpl.render(
        model_id        = model_id,
        n_episodes      = aggregate.get("n_episodes", 0),
        dataset_sources = aggregate.get("dataset_sources", []),
        messages        = aggregate.get("messages", []),
        axes            = aggregate.get("axes") or {},
        episode_links   = episode_links,
    )


# ────────────────────────────────────────────────────────────────────
# File-writing convenience wrappers
# ────────────────────────────────────────────────────────────────────

def write_episode_reports(
    summary: dict, out_dir: Path, *, rrd_path: Optional[str] = None,
) -> dict[str, Path]:
    """Write episode_summary.md (always) + episode_summary.html (if viz).

    Returns ``{"md": Path, "html": Path | None}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    md_path.write_text(render_episode_markdown(summary, rrd_path=rrd_path))
    html_path: Optional[Path] = None
    try:
        html = render_episode_html(summary, rrd_path=rrd_path)
        html_path = out_dir / "report.html"
        html_path.write_text(html)
    except ImportError:
        # viz extra not installed — markdown is still written, the user
        # can pip install 'emboviz[viz]' for HTML later.
        pass
    return {"md": md_path, "html": html_path}
