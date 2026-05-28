"""Shared install helper for the ``install-*`` subcommands.

Both ``emboviz install-pi0`` and ``emboviz install-gr00t`` install an
upstream git package into the *active* venv. We prefer ``uv pip install``
over ``python -m pip install`` because:

  • the project standard is uv (never bare pip);
  • uv shares a global wheel cache, so big wheels (torch, jax, …) already
    pulled for another model venv are REUSED instead of re-downloaded —
    the difference is minutes-of-cold-download vs near-instant;
  • uv resolves + downloads in parallel.

If ``uv`` isn't on PATH (e.g. the user installed emboviz with plain pip),
we fall back to ``python -m pip install``, bootstrapping pip via
``ensurepip`` if the venv lacks it (uv-created venvs do).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Optional

import click


def pip_install(
    packages: list[str],
    *,
    python: Optional[str] = None,
    no_deps: bool = False,
    force_reinstall: bool = False,
    env: Optional[dict] = None,
    label: str = "install",
) -> str:
    """Install ``packages`` into the venv owning ``python``.

    Prefers ``uv pip install --python <python>`` (fast, shares uv's wheel
    cache); falls back to ``python -m pip install``. Raises
    ``click.ClickException`` on failure. Returns the backend used
    ("uv" or "pip") so callers can message accordingly.
    """
    python = python or sys.executable
    uv = shutil.which("uv")

    if uv:
        cmd = [uv, "pip", "install", "--python", python]
        if no_deps:
            cmd.append("--no-deps")
        if force_reinstall:
            cmd.append("--reinstall")  # uv's spelling of --force-reinstall
        backend = "uv"
    else:
        _ensure_pip(python, label)
        cmd = [python, "-m", "pip", "install"]
        if no_deps:
            cmd.append("--no-deps")
        if force_reinstall:
            cmd.append("--force-reinstall")
        backend = "pip"

    cmd.extend(packages)
    click.echo(f"[{label}] {backend} pip install: {' '.join(packages)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise click.ClickException(
            f"{backend} install exited with code {result.returncode}"
        )
    return backend


def _ensure_pip(python: str, label: str) -> None:
    """Make sure ``python -m pip`` works (uv-created venvs ship without pip)."""
    r = subprocess.run(
        [python, "-m", "pip", "--version"], capture_output=True, text=True
    )
    if r.returncode == 0:
        return
    click.echo(f"[{label}] pip not available in this venv — bootstrapping via ensurepip")
    r = subprocess.run([python, "-m", "ensurepip", "--upgrade"])
    if r.returncode != 0:
        raise click.ClickException(
            "could not bootstrap pip in this venv, and `uv` is not on PATH. "
            "Install uv (https://docs.astral.sh/uv/) or activate a venv that has pip."
        )


