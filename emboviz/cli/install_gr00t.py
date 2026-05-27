"""`emboviz install-gr00t` — one-shot wrapper for NVIDIA's Isaac-GR00T install.

WHY this exists
---------------

NVIDIA's gr00t Python package is not on PyPI; it ships only as a GitHub
repo. We can't put it in the ``[gr00t]`` extra as a normal git+ URL
because gr00t's pyproject lists ``flash-attn`` as a build dependency.
flash-attn's setup imports torch BEFORE pip has installed it (the build
isolation contract is empty env at build time), so the install fails
with ``ModuleNotFoundError: torch``.

The workaround is to install gr00t with ``--no-deps``: flash-attn is
listed but never gets built, and our gr00t adapter falls back to SDPA /
eager attention anyway, so flash-attn isn't actually needed at run time.

This subcommand is a one-line wrapper for the user.
"""

from __future__ import annotations

import subprocess
import sys

import click

from emboviz.cli._install_common import pip_install


_DEFAULT_GR00T_REPO = "git+https://github.com/NVIDIA/Isaac-GR00T.git"


@click.command("install-gr00t")
@click.option(
    "--repo", default=_DEFAULT_GR00T_REPO,
    help="git+ URL to install gr00t from. Default is NVIDIA's upstream; "
         "override if you have a fork.",
)
@click.option(
    "--force-reinstall", is_flag=True, default=False,
    help="Pass --force-reinstall to pip, useful if a previous install "
         "stopped partway through.",
)
def install_gr00t_cmd(repo: str, force_reinstall: bool) -> None:
    """Install NVIDIA's gr00t package into the active venv.

    Run this once after ``uv pip install 'emboviz[gr00t]'``. Subsequent
    ``emboviz analyze --model gr00t ...`` calls then work.

    Why this isn't a transitive dep of the ``[gr00t]`` extra: gr00t's
    pyproject lists flash-attn as a hard dep, but flash-attn fails to
    build under pip's standard build-isolation (it imports torch at
    build time but pip hasn't installed it yet). The fix is
    ``--no-deps``; emboviz's gr00t adapter uses SDPA/eager attention so
    flash-attn isn't actually needed at runtime.

    Example:

    \b
        uv venv
        uv pip install 'emboviz[gr00t]'
        emboviz install-gr00t
        emboviz analyze --model gr00t \\
            --model-kwargs '{"camera_mapping": {...}}' \\
            --dataset droid-sample --episodes 1 \\
            --target "the blue block" --output ./report
    """
    # Locate the venv's pip — uv-created venvs may not have pip on
    # the path (uv installs without it). Use ``python -m pip`` and
    # ensure pip is present in the active venv.
    python = sys.executable
    click.echo(f"[install-gr00t] using python at: {python}")

    click.echo("[install-gr00t] installing gr00t with --no-deps")
    click.echo("[install-gr00t] (--no-deps because gr00t's flash-attn build dep")
    click.echo("                 fails under pip build isolation; gr00t adapter")
    click.echo("                 falls back to SDPA/eager attention.)\n")
    backend = pip_install(
        [repo], python=python, no_deps=True, force_reinstall=force_reinstall,
        label="install-gr00t",
    )
    click.echo(f"[install-gr00t] (used {backend} as the installer)")

    # Verify import.
    click.echo("\n[install-gr00t] verifying import")
    r = subprocess.run(
        [python, "-c",
         "import gr00t, gr00t.data; print('  gr00t at', gr00t.__file__)"],
        capture_output=False,
    )
    if r.returncode != 0:
        raise click.ClickException(
            "install completed but ``import gr00t`` fails. Check the install log."
        )

    click.echo("\n[install-gr00t] DONE")
    click.echo(
        "Next: emboviz analyze --model gr00t "
        "--model-kwargs '{\"camera_mapping\": {...}}' ..."
    )
