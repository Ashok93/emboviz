"""``emboviz install-<adapter>`` — materialise an adapter's runtime venv.

Each adapter shim (``emboviz-openvla``, ``emboviz-oft``, ``emboviz-pi0``,
``emboviz-gr00t``, ``emboviz-sam3``) registers its :class:`AdapterSpec`
via entry points. This command reads that spec and:

  1. Creates an isolated venv at ``~/.emboviz/venvs/<name>`` using
     ``uv venv --python <spec.requires_python>``.
  2. Runs ``uv pip install`` inside that venv with the heavy
     ``spec.runtime_pip`` requirements (torch + transformers +
     lerobot + the model checkpoint code) and the adapter's own
     env vars (``GIT_LFS_SKIP_SMUDGE`` for π0, etc.).
  3. Runs a one-line ``python -c "from emboviz_<adapter>.model
     import *"`` import sanity check so failures surface here
     instead of mid-analyze.

Because the dev-path and user-path are the same (CLAUDE.md "Dev path
is the user path"), the scripts/setup/0N_install_<name>_venv.sh dev
recipes just call this command — no separate provisioning logic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from emboviz.adapters import find_adapter, install_venv
from emboviz.adapters.lifecycle import venv_path, venv_python


def _build_install_cmd(name: str, description: str) -> click.Command:
    """Factory: produce ``install-<name>`` Click command for one adapter."""

    @click.command(
        f"install-{name}",
        help=(
            f"Materialise the runtime venv for the {description} "
            f"adapter.\n\nRun ONCE after `uv pip install emboviz-"
            f"{name}`. Creates ~/.emboviz/venvs/{name} and installs "
            "the heavy model deps into it."
        ),
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Delete and recreate the venv even if it already exists.",
    )
    @click.option(
        "--check-import",
        is_flag=True,
        default=True,
        help="Run a sanity import after install (default: on).",
    )
    def _cmd(force: bool, check_import: bool) -> None:
        spec = find_adapter(name)
        click.echo(f"[install-{name}] adapter spec: {spec.actor_import_path}")
        click.echo(f"[install-{name}] runtime python: {spec.requires_python}")
        click.echo(
            f"[install-{name}] runtime pip ({len(spec.runtime_pip)}):"
        )
        for req in spec.runtime_pip:
            click.echo(f"    {req}")

        path = install_venv(spec, force=force)
        click.echo(f"[install-{name}] venv ready: {path}")

        if check_import:
            actor_module = spec.actor_import_path.split(":", 1)[0]
            check = (
                f"import {actor_module} as m; "
                "print('  actor module:', m.__name__)"
            )
            try:
                py = venv_python(name)
                subprocess.run(
                    [str(py), "-c", check],
                    check=True,
                )
                click.echo(f"[install-{name}] sanity import: OK")
            except subprocess.CalledProcessError as e:
                click.echo(
                    f"[install-{name}] sanity import FAILED ({e}). "
                    f"The venv at {path} was created but the actor "
                    "module didn't import — check the runtime_pip "
                    "in the adapter's spec.py.",
                    err=True,
                )
                sys.exit(1)

    return _cmd


# One subcommand per built-in adapter alias the user CAN install via
# this CLI. The list is hard-coded (rather than discovered from the
# entry-point registry) so ``emboviz install-openvla`` errors with
# "install emboviz-openvla first" when the shim isn't present, instead
# of silently not showing up under ``emboviz --help``.
INSTALLABLE_ADAPTERS = {
    "openvla":  "OpenVLA-7B",
    "oft":      "OpenVLA-OFT (LIBERO fine-tunes)",
    "pi0":      "Physical Intelligence π0 / π0.5",
    "gr00t":    "NVIDIA GR00T N1.5 / N1.7",
    "sam3":     "Meta Segment Anything 3 (text-prompted detector)",
}


def register_install_commands(main_group: click.Group) -> None:
    """Attach all ``install-<adapter>`` subcommands to the root group."""
    for name, desc in INSTALLABLE_ADAPTERS.items():
        main_group.add_command(_build_install_cmd(name, desc))
