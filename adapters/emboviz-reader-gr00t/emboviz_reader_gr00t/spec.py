"""AdapterSpec for the GR00T-format dataset reader.

Discovered by emboviz core through the ``emboviz.readers`` entry-point
group declared in this package's ``pyproject.toml``. Carries the worker
launch target, the pip requirements for the isolated reader venv, and the
Python version that venv runs.

This module must stay IMPORT-LIGHT — emboviz core imports it from the
user's main venv to read SPEC. No lerobot, no torch.

``name = "reader-gr00t"`` is DELIBERATELY distinct from the GR00T *model*
adapter's ``"gr00t"``. The runtime venv is ``venv_path(spec.name)``, so a
shared name would make this reader and the model adapter fight over one
venv (a light v2.1 lerobot install vs the multi-GB gr00t model stack).
Reader and model live in separate entry-point groups (``emboviz.readers``
vs ``emboviz.adapters``) but the same venv namespace — hence the distinct
name. The user-facing config key stays ``dataset.format: gr00t``; core's
``build_source`` maps that to ``connect_reader("reader-gr00t", ...)``.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="reader-gr00t",
    server_module="emboviz_reader_gr00t.server",
    # GR00T datasets are LeRobot **v2.1**. lerobot >=0.4 reads only the
    # v3.0 on-disk format and hard-refuses v2.x (BackwardCompatibilityError),
    # so this reader pins the last v2.1-capable release: 0.3.3 (0.4.0 already
    # flipped CODEBASE_VERSION to "v3.0"). lerobot pulls its own
    # video-decode + torch stack; we don't restate it (its pyproject is the
    # source of truth). This is a separate, lighter venv than the gr00t
    # MODEL adapter — it never imports the gr00t package.
    runtime_pip=(
        "lerobot>=0.3.3,<0.4",
        "emboviz-wire",
        "emboviz-reader-gr00t",
    ),
    # lerobot pulls ``pynput`` (keyboard/mouse capture for live
    # teleoperation) → ``evdev``, whose C extension needs Python.h + kernel
    # input headers to build from sdist. A dataset READER never
    # teleoperates — it only touches ``lerobot.datasets`` — so pynput is
    # dead weight that also breaks the install on header-less boxes. Drop
    # it (uv --override, false marker), same as the v3.0 lerobot reader. If
    # 0.3.x doesn't pull it, the override is a harmless no-op.
    runtime_pip_exclude=("pynput",),
    description=(
        "GR00T-format dataset reader (isolated). Pins lerobot 0.3.x "
        "(LeRobot v2.1) and reads LeRobot v2.1 + meta/modality.json "
        "datasets, HF Hub or local path."
    ),
    # Match the GR00T model venv's Python (3.11), a known-good interpreter
    # for this lerobot era with broad wheel availability.
    requires_python="3.11",
    needs_gpu=False,
)
