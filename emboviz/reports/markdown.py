"""Markdown-format suite report — engineer-readable, copy-paste-into-ticket.

User-facing output:
  • A short headline per diagnostic axis.
  • The Finding (observed / meaning / next-step) in plain English.
  • A single-character emoji badge that hints at priority WITHOUT
    using severity words (no "CRITICAL"/"PASS"/"MODERATE" anywhere
    in the rendered text).

The severity enum is used only for sort order (worst-first within each
section). Inconclusive diagnostics ("UNKNOWN" — couldn't run, or
response below noise floor) go in their own section so they don't
distract from actual findings.
"""

from __future__ import annotations

from pathlib import Path

from emboviz.core.results import Severity
from emboviz.suites.base import SuiteResult


# Badges intentionally do NOT spell out severity words. They are a visual
# hint for skim-reading; the Finding text below carries the substance.
_BADGE = {
    Severity.CRITICAL: "🔴",
    Severity.MODERATE: "🟠",
    Severity.INFO:     "🔵",
    Severity.PASS:     "🟢",
    Severity.UNKNOWN:  "⚪",
}


def _finding_or_legacy(r) -> tuple[str, str, str]:
    """Return (observed, meaning, next_step) — from Finding if present,
    else fall back to legacy explanation text in `observed`."""
    if r.finding is not None:
        return r.finding.observed, r.finding.meaning, r.finding.next_step
    return (r.explanation or "", "", "")


def render_markdown_report(suite_result: SuiteResult, out_path: Path) -> Path:
    runnable = [r for r in suite_result.results.values() if r.severity != Severity.UNKNOWN]
    skipped  = [r for r in suite_result.results.values() if r.severity == Severity.UNKNOWN]

    # Sort worst → best by internal severity priority, then by score.
    runnable.sort(key=lambda r: (-r.severity.sort_key, r.scalar_score))

    lines: list[str] = [
        f"# Emboviz report — {suite_result.suite_name}",
        "",
        f"- **Model**: `{suite_result.model_id}`",
        f"- **Scene**: `{suite_result.scene_id}`",
        "",
        "## Findings",
        "",
    ]
    for r in runnable:
        observed, meaning, next_step = _finding_or_legacy(r)
        lines += [
            f"### {_BADGE[r.severity]} {r.axis}",
            "",
            f"- **Observed**: {observed}",
        ]
        if meaning:
            lines.append(f"- **Meaning**: {meaning}")
        if next_step:
            lines.append(f"- **Next step**: {next_step}")
        if r.finding and r.finding.raw_numbers:
            lines.append("- **Raw numbers** (for the curious):")
            for k, v in r.finding.raw_numbers.items():
                lines.append(f"  - `{k}`: {v}")
        lines.append("")

    if skipped:
        lines += ["## Diagnostics that could not run", ""]
        for r in skipped:
            observed, _, next_step = _finding_or_legacy(r)
            lines += [
                f"### {_BADGE[r.severity]} {r.axis}",
                "",
                f"- **Why**: {observed or r.explanation}",
            ]
            if next_step:
                lines.append(f"- **To get a verdict**: {next_step}")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path
