"""Markdown-format suite report — engineer-readable, copy-paste-into-ticket."""

from __future__ import annotations

from pathlib import Path

from policylens.core.results import Severity
from policylens.suites.base import SuiteResult


_SEV_BADGE = {
    Severity.CRITICAL: "🟥",
    Severity.MODERATE: "🟧",
    Severity.PASS: "🟩",
    Severity.INFO: "🟦",
    Severity.UNKNOWN: "⬜",
}


def render_markdown_report(suite_result: SuiteResult, out_path: Path) -> Path:
    runnable = [r for r in suite_result.results.values() if r.severity != Severity.UNKNOWN]
    skipped = [r for r in suite_result.results.values() if r.severity == Severity.UNKNOWN]

    runnable.sort(key=lambda r: r.scalar_score)
    lines = [
        f"# PolicyLens Report — {suite_result.suite_name}",
        "",
        f"**Model**: `{suite_result.model_id}`",
        f"**Scene**: `{suite_result.scene_id}`",
        "",
        "## Failure profile (sorted, lowest score first)",
        "",
        "| Axis | Score | Severity | Verdict |",
        "| --- | ---: | --- | --- |",
    ]
    for r in runnable:
        badge = _SEV_BADGE[r.severity]
        explain = (r.explanation or "").replace("\n", " ")
        lines.append(
            f"| {r.axis} | {r.scalar_score:.3f} | {badge} {r.severity.value} | {explain} |"
        )

    if skipped:
        lines += ["", "## Skipped diagnostics", ""]
        for r in skipped:
            lines.append(f"- {r.diagnostic_name} — {r.explanation}")

    if runnable:
        lines += ["", "## Detail per diagnostic", ""]
        for r in runnable:
            lines.append(f"### {_SEV_BADGE[r.severity]} {r.diagnostic_name}  ({r.axis})")
            lines.append(f"- score: **{r.scalar_score:.3f}**  ({r.direction})")
            lines.append(f"- severity: **{r.severity.value}**")
            lines.append(f"- verdict: {r.explanation}")
            if r.recommendation:
                lines.append(f"- recommendation:")
                for ln in r.recommendation.splitlines():
                    lines.append(f"  {ln}")
            if r.per_variant:
                lines.append(f"- per-variant scores:")
                for vid, val in r.per_variant.items():
                    lines.append(f"  - `{vid}`: {val:.3f}")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path
