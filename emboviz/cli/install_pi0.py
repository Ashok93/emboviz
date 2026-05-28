"""`emboviz install-pi0` — one-shot wrapper for Physical Intelligence's openpi.

WHY this exists
---------------

π0 / π0.5 inference is provided by ``openpi`` (Physical-Intelligence/openpi).
openpi is **not on PyPI** — it ships only as a GitHub repo — and, critically,
it **cannot be installed without** ``GIT_LFS_SKIP_SMUDGE=1`` set in the
environment. openpi's own README documents this:

    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    # "GIT_LFS_SKIP_SMUDGE=1 is needed to pull LeRobot as a dependency."

openpi pins ``lerobot`` to an old git commit whose tests/artifacts/*.safetensors
LFS fixtures are no longer fetchable from the remote ("remote missing object").
Those fixtures are test data, not part of the built lerobot wheel, so skipping
the smudge is correct: git-lfs leaves pointer files and the wheel builds from
the Python sources.

We therefore CANNOT ship ``openpi`` as a normal ``openpi @ git+...`` entry in
the ``[pi0]`` extra: a plain ``uv pip install 'emboviz[pi0]'`` resolves that
transitive git dependency with smudge ENABLED and crashes, and there is no way
for a user to inject an env var into a transitive dependency build. So openpi
lives behind this subcommand, which sets the env var in the install subprocess
exactly as openpi documents.

(This mirrors ``emboviz install-gr00t``, which exists for a different upstream
packaging quirk — gr00t's flash-attn build dep.)
"""

from __future__ import annotations

import os
import subprocess
import sys

import click

from emboviz.cli._install_common import pip_install


_DEFAULT_OPENPI_REPO = "openpi @ git+https://github.com/Physical-Intelligence/openpi.git"
# openpi's gemma wrapper targets transformers 4.53.x; openpi's own metadata
# does not pin it tightly, so we pin it after install to the version the
# integration test validates against.
_DEFAULT_TRANSFORMERS_PIN = "transformers==4.53.2"


@click.command("install-pi0")
@click.option(
    "--repo", default=_DEFAULT_OPENPI_REPO,
    help="PEP 508 spec to install openpi from. Default is PI's upstream "
         "(git+); override if you have a fork.",
)
@click.option(
    "--transformers-pin", default=_DEFAULT_TRANSFORMERS_PIN,
    help="transformers version to pin after installing openpi. Set to an "
         "empty string to skip pinning.",
)
@click.option(
    "--force-reinstall", is_flag=True, default=False,
    help="Pass --force-reinstall to pip, useful if a previous install "
         "stopped partway through.",
)
def install_pi0_cmd(repo: str, transformers_pin: str, force_reinstall: bool) -> None:
    """Install Physical Intelligence's openpi into the active venv.

    Run this once after ``uv pip install 'emboviz[pi0]'``. Subsequent
    ``emboviz analyze --model pi0 ...`` calls then work.

    Why this isn't a transitive dep of the ``[pi0]`` extra: openpi is
    git-only and its documented install REQUIRES ``GIT_LFS_SKIP_SMUDGE=1``
    (it pins an old lerobot commit whose git-lfs test fixtures are no
    longer fetchable). A transitive ``openpi @ git+...`` would resolve
    with smudge enabled and crash, and a user can't set an env var on a
    transitive build. This subcommand sets it correctly.

    Example:

    \b
        uv venv
        uv pip install 'emboviz[pi0]'
        emboviz install-pi0
        # optional 3rd step for the attention diagnostic (PyTorch backend):
        emboviz convert-pi0 pi0_libero
        emboviz analyze --model pi0 --dataset pi-libero \\
            --episodes 0 --target "the white mug" --output ./report
    """
    python = sys.executable
    click.echo(f"[install-pi0] using python at: {python}")
    click.echo(
        "[install-pi0] heads up: openpi is a heavy install (its own JAX + "
        "PyTorch stack — several GB). Prefer running this in a uv-managed "
        "venv so the wheel cache is reused.\n"
    )

    # openpi's documented install: GIT_LFS_SKIP_SMUDGE=1 in the environment
    # so the transitive lerobot git checkout doesn't try to smudge fixtures.
    env = dict(os.environ)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    click.echo("[install-pi0] installing openpi with GIT_LFS_SKIP_SMUDGE=1")
    click.echo("[install-pi0] (that env var is openpi's documented requirement —")
    click.echo("               it pins an old lerobot commit whose git-lfs test")
    click.echo("               fixtures are no longer fetchable.)\n")
    backend = pip_install(
        [repo], python=python, force_reinstall=force_reinstall,
        env=env, label="install-pi0",
    )
    click.echo(f"[install-pi0] (used {backend} as the installer)")

    # Pin transformers to the version openpi's gemma wrapper expects.
    if transformers_pin:
        click.echo(f"\n[install-pi0] pinning {transformers_pin}")
        pip_install(
            [transformers_pin], python=python, env=env, label="install-pi0",
        )

    # Verify import.
    click.echo("\n[install-pi0] verifying import")
    r = subprocess.run(
        [python, "-c", "import openpi; print('  openpi at', openpi.__file__)"],
    )
    if r.returncode != 0:
        raise click.ClickException(
            "install completed but ``import openpi`` fails. Check the install log."
        )

    click.echo("\n[install-pi0] DONE")
    click.echo(
        "Next: emboviz analyze --model pi0 --dataset pi-libero ...  "
        "(run `emboviz convert-pi0 pi0_libero` first if you want the "
        "attention diagnostic, which needs the PyTorch backend.)"
    )
