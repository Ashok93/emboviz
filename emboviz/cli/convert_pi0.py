"""`emboviz convert-pi0 <config>` — one-shot helper for the JAX→PyTorch
checkpoint conversion that π0's PyTorch backend requires.

WHY this exists
---------------

Physical Intelligence trains π0 in JAX. Their PyTorch backend (which
the emboviz adapter uses for the attention-extraction diagnostic) reads
a separately-converted checkpoint format. The conversion is documented
upstream as a 3-step manual dance:

  1. Apply openpi's vendored "transformers_replace" patch on top of the
     installed transformers package.
  2. Run ``openpi/examples/convert_jax_model_to_pytorch.py`` with the
     right ``--config-name`` flag.
  3. Manually ``cp -r assets/`` from the JAX checkpoint to the PyTorch
     output directory (the convert script doesn't move norm_stats.json).

This subcommand does all 3 in one call. After it succeeds, the
adapter's ``use_pytorch=True`` path Just Works on this checkpoint.

When the user does NOT need this
--------------------------------

emboviz's attention-extraction diagnostic for π0 is the only one that
requires the PyTorch backend. If the user is OK skipping attention
(the other 4 diagnostics still run on JAX), this subcommand is
optional. The adapter's default ``use_pytorch=False`` path works
without any conversion.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click


def _openpi_root() -> Optional[Path]:
    """Locate the openpi source root.

    The user installed ``emboviz[pi0]`` which pulled
    ``openpi @ git+...`` into the active venv as editable / wheel.
    The convert script lives in the source repo, not the wheel —
    so we look up ``openpi``'s ``__file__`` and walk up to the
    repo root that contains ``examples/convert_jax_model_to_pytorch.py``.
    """
    try:
        import openpi
    except ImportError:
        return None
    p = Path(openpi.__file__).resolve()
    # Walk up looking for the examples directory.
    for ancestor in [p.parent, *p.parents]:
        candidate = ancestor / "examples" / "convert_jax_model_to_pytorch.py"
        if candidate.exists():
            return ancestor
    return None


def _hf_cache_root() -> Path:
    """openpi's expected checkpoint cache location."""
    import os
    base = os.environ.get("OPENPI_CACHE_DIR") or os.path.expanduser(
        "~/.cache/openpi/openpi-assets/checkpoints"
    )
    return Path(base)


def _apply_transformers_replace(openpi_root: Path) -> None:
    """openpi monkey-patches transformers' gemma modeling code. The
    convert script + the PyTorch backend both refuse to run unless
    these patches are applied. Copy them into the active venv's
    transformers/ install.
    """
    import transformers
    transformers_dir = Path(transformers.__file__).parent
    patch_dir = openpi_root / "src" / "openpi" / "models_pytorch" / "transformers_replace"
    if not patch_dir.exists():
        raise click.ClickException(
            f"openpi's transformers_replace patch dir not found at {patch_dir} "
            f"— openpi layout has changed; update emboviz/cli/convert_pi0.py."
        )

    click.echo(f"[convert-pi0] applying transformers_replace patch")
    click.echo(f"             from: {patch_dir}")
    click.echo(f"             to:   {transformers_dir}")
    for src in patch_dir.rglob("*"):
        if src.is_file():
            rel = src.relative_to(patch_dir)
            dst = transformers_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


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
    """Convert a JAX π0 checkpoint to PyTorch for attention extraction.

    Example:

    \b
        # In a venv where ``emboviz[pi0]`` is installed
        emboviz convert-pi0 pi0_libero
        emboviz analyze --model pi0 --dataset pi-libero \\
            --episodes 0 --target "the white mug" --output ./report
    """
    openpi_root = _openpi_root()
    if openpi_root is None:
        raise click.ClickException(
            "openpi is not installed in this venv. Install it first:\n"
            "    uv pip install 'emboviz[pi0]'"
        )
    click.echo(f"[convert-pi0] found openpi at: {openpi_root}")

    convert_script = openpi_root / "examples" / "convert_jax_model_to_pytorch.py"
    if not convert_script.exists():
        raise click.ClickException(
            f"openpi convert script not found at {convert_script} "
            f"— openpi layout has changed; update emboviz/cli/convert_pi0.py."
        )

    cache_root = _hf_cache_root()
    jax_dir = Path(jax_checkpoint) if jax_checkpoint else cache_root / config_name
    out_dir = Path(output_path) if output_path else cache_root / f"{config_name}_pytorch"

    # 1) Apply transformers_replace patch (idempotent).
    if not skip_transformers_patch:
        _apply_transformers_replace(openpi_root)

    # 2) Download JAX checkpoint if missing.
    if not jax_dir.exists() and download_jax:
        click.echo(f"[convert-pi0] downloading JAX checkpoint for '{config_name}'")
        try:
            from openpi.shared import download
            resolved = download.maybe_download(
                f"gs://openpi-assets/checkpoints/{config_name}"
            )
            click.echo(f"             JAX checkpoint at: {resolved}")
            if Path(resolved) != jax_dir:
                jax_dir = Path(resolved)
        except Exception as e:
            raise click.ClickException(
                f"failed to download JAX checkpoint via openpi.shared.download: {e}"
            ) from e
    elif not jax_dir.exists():
        raise click.ClickException(
            f"JAX checkpoint not found at {jax_dir} and --no-download-jax was set."
        )

    # 3) Run the convert script.
    cache_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"[convert-pi0] running convert script ({precision})")
    click.echo(f"             jax_dir:  {jax_dir}")
    click.echo(f"             out_dir:  {out_dir}")
    cmd = [
        sys.executable, str(convert_script),
        "--checkpoint-dir", str(jax_dir),
        "--config-name", config_name,
        "--output-path", str(out_dir),
        "--precision", precision,
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise click.ClickException(
            f"openpi convert script exited with code {result.returncode}"
        )

    # 4) Copy assets/ over (norm_stats.json + per-dataset metadata).
    assets_src = jax_dir / "assets"
    assets_dst = out_dir / "assets"
    if assets_src.exists():
        if assets_dst.exists():
            click.echo(f"[convert-pi0] assets/ already present in {out_dir}, leaving as-is")
        else:
            click.echo(f"[convert-pi0] copying assets/ from JAX checkpoint")
            shutil.copytree(assets_src, assets_dst)
    else:
        click.echo(
            f"[convert-pi0] WARN: no assets/ in {jax_dir} — adapter may fail "
            f"on norm_stats lookup. Check the JAX checkpoint structure."
        )

    click.echo(f"[convert-pi0] DONE → {out_dir}")
    click.echo(
        f"\nNext: emboviz analyze --model pi0 "
        f"--model-kwargs '{{\"config_name\": \"{config_name}\", \"use_pytorch\": true}}' ..."
    )
