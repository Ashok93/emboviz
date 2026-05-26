"""Per-diagnostic detail pages — Markdown emission, no severity-word leak.

Each diagnostic gets its own page with:
  • axis + diagnostic name
  • the Finding (observed / meaning / next-step) in plain English
  • per-variant scores in a table
  • compact summary of `raw` payload (depends on diagnostic)

Severity is internal — it never appears as a word in these pages. We use
a single emoji badge for visual priority and let the Finding text carry
the actual meaning.
"""

from __future__ import annotations

from pathlib import Path

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.suites.base import SuiteResult


# Badges intentionally do NOT spell out severity words.
_BADGE = {
    Severity.CRITICAL: "🔴",
    Severity.MODERATE: "🟠",
    Severity.INFO:     "🔵",
    Severity.PASS:     "🟢",
    Severity.UNKNOWN:  "⚪",
}


def render_detail_pages(suite_result: SuiteResult, out_dir: Path) -> list[Path]:
    """Emit one Markdown file per diagnostic into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for diag_name, r in suite_result.results.items():
        path = out_dir / f"{r.axis.replace('.', '__')}.md"
        path.write_text(_format_markdown(r, diag_name))
        paths.append(path)
    return paths


def _format_markdown(r: DiagnosticResult, diag_name: str) -> str:
    badge = _BADGE[r.severity]

    lines: list[str] = [
        f"# {badge} {r.axis}",
        "",
        f"- **diagnostic**: `{diag_name}`",
        f"- **scalar score**: `{r.scalar_score:.4f}` ({r.direction.replace('_', ' ')})",
        f"- **model**: `{r.model_id}`",
        f"- **scene**: `{r.scene_id}`",
        "",
        "## Finding",
        "",
    ]
    if r.finding is not None:
        f = r.finding
        lines += [
            f"- **Observed**: {f.observed}",
            f"- **Meaning**: {f.meaning}",
            f"- **Next step**: {f.next_step}",
        ]
        if f.raw_numbers:
            lines += ["", "**Raw numbers**:", ""]
            for k, v in f.raw_numbers.items():
                lines.append(f"- `{k}`: {v}")
    else:
        # Legacy: pre-Finding diagnostic. Render explanation as-is.
        lines.append(r.explanation or "_(no verdict text)_")
    lines.append("")

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
