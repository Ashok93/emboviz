"""Probe serialization — JSON header + .npz weights."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from emboviz.probes.base import LinearProbe, ProbeSpec


def save_probe(probe: LinearProbe, out_path: Path) -> Path:
    """Save probe to `out_path.npz` with metadata JSON sidecar."""
    out_path = Path(out_path).with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, weights=probe.weights, bias=probe.bias)
    (out_path.with_suffix(".json")).write_text(
        json.dumps(asdict(probe.spec), indent=2)
    )
    return out_path


def load_probe(path: Path) -> LinearProbe:
    """Load probe from a path (with or without `.npz` extension)."""
    path = Path(path).with_suffix(".npz")
    arrs = np.load(path)
    spec_dict = json.loads(path.with_suffix(".json").read_text())
    spec = ProbeSpec(**spec_dict)
    return LinearProbe(
        spec=spec,
        weights=arrs["weights"].astype(np.float32),
        bias=arrs["bias"].astype(np.float32),
    )
