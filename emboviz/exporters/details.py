"""Per-diagnostic detail pages — Markdown emission, no prose synthesis.

Each diagnostic gets its own page with:
  • axis + diagnostic name
  • severity + scalar score
  • the diagnostic's own factual explanation (already in DiagnosticResult)
  • per-variant scores in a table
  • compact summary of `raw` payload (depends on diagnostic)

The pages are deliberately *data-dense and prose-light*. No "here's what
to do about it" — that's a Cloud feature, context-aware and interactive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.suites.base import SuiteResult


def render_detail_pages(suite_result: SuiteResult, out_dir: Path) -> list[Path]:
    """Emit one Markdown file per diagnostic into `out_dir`. Returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for diag_name, r in suite_result.results.items():
        path = out_dir / f"{r.axis.replace('.', '__')}.md"
        path.write_text(_format_markdown(r, diag_name))
        paths.append(path)
    return paths


def _format_markdown(r: DiagnosticResult, diag_name: str) -> str:
    sev_badge = {
        Severity.CRITICAL: "🟥 CRITICAL",
        Severity.MODERATE: "🟧 MODERATE",
        Severity.INFO:     "🟦 INFO",
        Severity.PASS:     "🟩 PASS",
        Severity.UNKNOWN:  "⬜ N/A",
    }[r.severity]

    lines: list[str] = [
        f"# {r.axis}",
        "",
        f"**diagnostic**: `{diag_name}`  ",
        f"**severity**: {sev_badge}  ",
        f"**scalar score**: `{r.scalar_score:.4f}` ({r.direction.replace('_', ' ')})  ",
        f"**model**: `{r.model_id}`  ",
        f"**scene**: `{r.scene_id}`",
        "",
        "## Finding",
        "",
        r.explanation,
        "",
    ]

    if r.per_variant:
        lines += ["## Per-variant scores", "", "| variant | score |", "|---|---|"]
        for v, s in r.per_variant.items():
            try:
                s_str = f"{float(s):.4f}"
            except (TypeError, ValueError):
                s_str = str(s)
            lines.append(f"| `{v}` | {s_str} |")
        lines.append("")

    if r.raw:
        lines += ["## Raw data (debugging)", "", "<details><summary>show</summary>", ""]
        for k, v in r.raw.items():
            lines.append(f"- **{k}**: {_short_repr(v)}")
        lines += ["", "</details>", ""]

    return "\n".join(lines)


def _short_repr(v) -> str:
    """One-line readable repr, truncated to keep pages scannable."""
    if isinstance(v, (list, tuple)) and len(v) > 8:
        head = ", ".join(str(x) for x in v[:6])
        return f"[{head}, ... ({len(v)} items)]"
    if isinstance(v, dict) and len(v) > 8:
        keys = list(v.keys())[:6]
        return "{" + ", ".join(f"{k!r}: ..." for k in keys) + f", ... ({len(v)} keys)}}"
    s = repr(v)
    return s if len(s) <= 200 else s[:197] + "..."
