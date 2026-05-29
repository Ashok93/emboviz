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
    # The reader venv tracks the LATEST lerobot (>=0.5), whose dataset
    # format is v3.0 — the current official LeRobot standard. v3.0 is a
    # HARD break: lerobot >=0.4 no longer reads v2.0/v2.1 (it raises
    # BackwardCompatibilityError). So emboviz accepts v3.0 datasets ONLY
    # and tells users to convert older data once with lerobot's own
    # ``python -m lerobot.datasets.v30.convert_dataset_v21_to_v30``. We do
    # not ship an old reader to humour old data. lerobot >=0.5 needs
    # Python 3.12. lerobot pulls its own video-decode + torch stack; we
    # don't restate it (its pyproject is the source of truth). Any
    # rerun-sdk it drags in is isolated to this venv — the reader never
    # exports .rrd, so it can't collide with core's modern rerun.
    runtime_pip=(
        "lerobot>=0.5,<0.6",
        "emboviz-wire",
        "emboviz-lerobot",
    ),
    # lerobot pulls ``pynput`` (keyboard/mouse capture for live
    # teleoperation) → ``evdev``, whose C extension needs Python.h +
    # kernel input headers to build from sdist. A dataset READER never
    # teleoperates — it only touches ``lerobot.datasets`` — so pynput is
    # dead weight that also breaks the install on header-less boxes. Drop
    # it (uv --override, false marker): provider-driven, same "exclude
    # what this worker doesn't need" as gr00t's flash-attn.
    runtime_pip_exclude=("pynput",),
    description=(
        "LeRobot dataset reader (isolated). Latest lerobot (>=0.5) / "
        "codebase v3.0 → reads LeRobot v3.0 datasets, HF Hub or local path."
    ),
    requires_python="3.12",
    needs_gpu=False,
)
