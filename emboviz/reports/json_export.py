"""JSON export of a suite run — for dashboards / CI / cross-run comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from emboviz.suites.base import SuiteResult


def export_json(suite_result: SuiteResult, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite_result.summary(), indent=2, default=_json_default))
    return out_path


def _json_default(o):
    """Coerce ndarray/tensor-ish things into list/float."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
