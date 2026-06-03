"""``emboviz convert-pi0 <config>`` — convert a JAX π0 checkpoint to
PyTorch for attention extraction.

WHY a wrapper
-------------
Physical Intelligence trains π0 in JAX. The PyTorch backend the emboviz
adapter uses for the attention diagnostic reads a separately-converted
checkpoint. The conversion needs ``openpi`` + ``transformers``, which live
ONLY in the π0 runtime venv (``~/.emboviz/venvs/pi0``) — never in the host
venv. So this host command is a thin wrapper: it resolves the pi0 venv's
interpreter and runs ``python -m emboviz_pi0.convert`` there (where the
real work happens), forwarding the adapter's env vars (e.g.
``GIT_LFS_SKIP_SMUDGE``). The conversion logic itself lives in
``emboviz_pi0.convert`` inside the adapter package.

When the user does NOT need this
--------------------------------
Only π0's attention diagnostic requires the PyTorch backend. The other
four diagnostics run on the default JAX path with no conversion. Run this
once per checkpoint, then set ``use_pytorch: true`` in the run config.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import click


@click.command("convert-pi0")
@click.argument("config_name")
@click.option(
    "--jax-checkpoint", type=click.Path(), default=None,
    help="Path to the JAX checkpoint dir. Defaults to "
         "~/.cache/openpi/openpi-assets/checkpoints/<config_name>.",
)
@click.option(
    "--output", "output_path", type=click.Path(), default=None,
    help="Path for the converted PyTorch checkpoint. Defaults to "
         "~/.cache/openpi/openpi-assets/checkpoints/<config_name>_pytorch.",
)
@click.option(
    "--precision", type=click.Choice(["float32", "bfloat16", "float16"]),
    default="bfloat16",
    help="Floating-point precision for the converted weights. Default "
         "bfloat16 (matches openpi's recommended inference setting).",
)
@click.option(
    "--skip-transformers-patch", is_flag=True, default=False,
    help="Skip applying openpi's transformers_replace patch. Use only "
         "if you have already applied it manually.",
)
@click.option(
    "--download-jax/--no-download-jax", default=True,
    help="If the JAX checkpoint is not present locally, download it "
         "via openpi's download helper first (default: yes).",
)
def convert_pi0_cmd(
    config_name: str, jax_checkpoint: Optional[str], output_path: Optional[str],
    precision: str, skip_transformers_patch: bool, download_jax: bool,
) -> None:
    """Convert a JAX π0 checkpoint to PyTorch (runs inside the π0 venv).

    Example:

    \b
        emboviz install-pi0          # creates the pi0 runtime venv (with openpi)
        emboviz convert-pi0 pi0_libero
        emboviz analyze --config pi0
    """
    from emboviz.adapters.lifecycle import venv_python
    from emboviz.adapters.registry import find_adapter

    # The pi0 shim must be installed (so we know the adapter + its env vars).
    spec = find_adapter("pi0")

    # The conversion runs in the pi0 RUNTIME venv, where openpi lives.
    try:
        py = venv_python("pi0")
    except FileNotFoundError as e:
        raise click.ClickException(
            f"{e}\nThe π0 runtime venv (with openpi + transformers) doesn't "
            "exist yet — run `emboviz install-pi0` first, then re-run "
            "`emboviz convert-pi0`."
        )

    cmd = [str(py), "-m", "emboviz_pi0.convert", config_name, "--precision", precision]
    if jax_checkpoint:
        cmd += ["--jax-checkpoint", jax_checkpoint]
    if output_path:
        cmd += ["--output", output_path]
    if skip_transformers_patch:
        cmd += ["--skip-transformers-patch"]
    cmd += ["--download-jax"] if download_jax else ["--no-download-jax"]

    # Forward the adapter's env vars (GIT_LFS_SKIP_SMUDGE etc.) — the same
    # set the worker is spawned with — so the checkpoint download behaves.
    env = dict(os.environ)
    env.update(spec.runtime_env_vars or {})

    click.echo(f"[convert-pi0] delegating to the pi0 runtime venv: {py}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise click.ClickException(
            f"convert-pi0 failed (exit code {result.returncode}); see the "
            "output above for the underlying error."
        )
