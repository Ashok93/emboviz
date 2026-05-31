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
  3. Runs a one-line ``python -c "import <server_module>"`` sanity
     check inside the runtime venv so failures surface here instead
     of mid-analyze.
"""

from __future__ import annotations

import subprocess
import sys

import click

from emboviz.adapters import find_adapter, find_reader, install_venv
from emboviz.adapters.lifecycle import venv_python


def _find_installable_spec(name: str):
    """Resolve an install-able :class:`AdapterSpec` from EITHER the model-
    adapter (``emboviz.adapters``) or the dataset-reader
    (``emboviz.readers``) entry-point group. Both kinds build their
    runtime venv identically; they differ only in the registry they
    register under."""
    try:
        return find_adapter(name)
    except KeyError:
        return find_reader(name)


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
        spec = _find_installable_spec(name)
        click.echo(f"[install-{name}] adapter server: {spec.server_module}")
        click.echo(f"[install-{name}] runtime python: {spec.requires_python}")
        click.echo(
            f"[install-{name}] runtime pip ({len(spec.runtime_pip)}):"
        )
        for req in spec.runtime_pip:
            click.echo(f"    {req}")
        if spec.runtime_pip_exclude:
            click.echo(
                f"[install-{name}] excluded (provider-declared, not "
                f"installed) ({len(spec.runtime_pip_exclude)}):"
            )
            for pkg in spec.runtime_pip_exclude:
                click.echo(f"    {pkg}")

        path = install_venv(spec, force=force)
        click.echo(f"[install-{name}] venv ready: {path}")

        if check_import:
            check = (
                f"import {spec.server_module} as m; "
                "print('  server module:', m.__name__)"
            )
            try:
                py = venv_python(name)
                subprocess.run([str(py), "-c", check], check=True)
                click.echo(f"[install-{name}] sanity import: OK")
                click.echo(
                    f"[install-{name}] start the worker with:\n"
                    f"    {path}/bin/{spec.console_script} serve"
                )
            except subprocess.CalledProcessError as e:
                click.echo(
                    f"[install-{name}] sanity import FAILED ({e}). "
                    f"The venv at {path} was created but the server "
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
    "lama":     "LaMa big-lama inpainting (on-manifold memorization fill)",
    # Dataset readers register under the ``emboviz.readers`` group but
    # install into a runtime venv identically — same command, same flow.
    "lerobot":      "LeRobot dataset reader (isolated; reads LeRobot v3.0 datasets)",
    "reader-gr00t": "GR00T-format dataset reader (isolated; LeRobot v2.1 + modality.json)",
}


def register_install_commands(main_group: click.Group) -> None:
    """Attach all ``install-<adapter>`` subcommands to the root group."""
    for name, desc in INSTALLABLE_ADAPTERS.items():
        main_group.add_command(_build_install_cmd(name, desc))
