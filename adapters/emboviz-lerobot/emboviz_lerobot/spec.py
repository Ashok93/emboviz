"""AdapterSpec for the LeRobot dataset reader.

Discovered by emboviz core through the ``emboviz.readers`` entry-point
group declared in this package's ``pyproject.toml``. Carries the worker
launch target, the pip requirements for the isolated reader venv, and
the Python version that venv runs.

This module must stay IMPORT-LIGHT — emboviz core imports it from the
user's main venv to read SPEC. No lerobot, no torch.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="lerobot",
    server_module="emboviz_lerobot.server",
    # The reader venv needs lerobot pinned to 0.3.x — its codebase
    # version is v2.1, which reads the v2.0 AND v2.1 datasets every
    # shipped config uses (lerobot reads its own major + lower minors,
    # and refuses other majors). 0.3.x also keeps Python 3.11 (0.5
    # requires 3.12). lerobot pulls its own video-decode + torch stack;
    # we don't restate it (lerobot's pyproject is the source of truth).
    # rerun-sdk<0.23 comes in transitively and is HARMLESS here — the
    # reader never exports .rrd, so it cannot collide with core's modern
    # rerun. That isolation is the whole point of this package.
    runtime_pip=(
        "lerobot>=0.3,<0.4",
        "emboviz-wire",
        "emboviz-lerobot",
    ),
    description=(
        "LeRobot dataset reader (isolated). lerobot 0.3.x / codebase v2.1 "
        "→ reads LeRobot v2.0 and v2.1 datasets, HF Hub or local path."
    ),
    requires_python="3.11",
    needs_gpu=False,
)
