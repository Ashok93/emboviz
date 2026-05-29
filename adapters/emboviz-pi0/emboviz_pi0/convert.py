"""``python -m emboviz_pi0.convert <config>`` — JAX→PyTorch checkpoint
conversion for π0's PyTorch attention backend.

This module runs INSIDE the π0 runtime venv, where ``openpi`` and
``transformers`` are installed. The host-side ``emboviz convert-pi0``
command is a thin wrapper that invokes it with this venv's interpreter —
the host venv has no ``openpi`` and could never do the conversion itself.

Physical Intelligence trains π0 in JAX; the PyTorch backend the emboviz
adapter uses for attention extraction reads a separately-converted
checkpoint. Upstream documents this as a 3-step manual dance, which this
module performs in one call:

  1. Apply openpi's vendored ``transformers_replace`` patch onto the
     installed ``transformers`` package.
  2. Run ``openpi/examples/convert_jax_model_to_pytorch.py``.
  3. Copy ``assets/`` (``norm_stats.json`` + per-dataset metadata) from
     the JAX checkpoint to the PyTorch output dir — the convert script
     doesn't move it and the adapter needs it for de-normalization.

It uses only the standard library + ``openpi``/``transformers``; the CLI
is argparse so it has no dependency beyond what the runtime venv already
ships.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    """Print an error and exit non-zero (the host wrapper surfaces it)."""
    print(f"[convert-pi0] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _openpi_root() -> Optional[Path]:
    """Locate the openpi source root.

    The convert script lives in the openpi source repo (not the wheel),
    so we resolve ``openpi.__file__`` and walk up to the directory that
    contains ``examples/convert_jax_model_to_pytorch.py``.
    """
    try:
        import openpi
    except ImportError:
        return None
    p = Path(openpi.__file__).resolve()
    for ancestor in [p.parent, *p.parents]:
        if (ancestor / "examples" / "convert_jax_model_to_pytorch.py").exists():
            return ancestor
    return None


def _hf_cache_root() -> Path:
    """openpi's expected checkpoint cache location."""
    base = os.environ.get("OPENPI_CACHE_DIR") or os.path.expanduser(
        "~/.cache/openpi/openpi-assets/checkpoints"
    )
    return Path(base)


def _apply_transformers_replace(openpi_root: Path) -> None:
    """Copy openpi's vendored gemma-modeling patch into the installed
    ``transformers`` package. The convert script and the PyTorch backend
    both require it."""
    import transformers

    transformers_dir = Path(transformers.__file__).parent
    patch_dir = openpi_root / "src" / "openpi" / "models_pytorch" / "transformers_replace"
    if not patch_dir.exists():
        _fail(
            f"openpi's transformers_replace patch dir not found at {patch_dir} "
            "— openpi's layout has changed; update emboviz_pi0/convert.py."
        )
    print("[convert-pi0] applying transformers_replace patch", flush=True)
    print(f"             from: {patch_dir}", flush=True)
    print(f"             to:   {transformers_dir}", flush=True)
    for src in patch_dir.rglob("*"):
        if src.is_file():
            dst = transformers_dir / src.relative_to(patch_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m emboviz_pi0.convert",
        description="Convert a JAX π0 checkpoint to PyTorch for attention extraction.",
    )
    parser.add_argument("config_name", help="openpi config name, e.g. pi0_libero")
    parser.add_argument(
        "--jax-checkpoint", default=None,
        help="JAX checkpoint dir. Default: "
             "~/.cache/openpi/openpi-assets/checkpoints/<config_name>.",
    )
    parser.add_argument(
        "--output", dest="output_path", default=None,
        help="Output dir for the PyTorch checkpoint. Default: "
             "~/.cache/openpi/openpi-assets/checkpoints/<config_name>_pytorch.",
    )
    parser.add_argument(
        "--precision", choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Converted-weight precision (default: bfloat16, openpi's "
             "recommended inference setting).",
    )
    parser.add_argument(
        "--skip-transformers-patch", action="store_true",
        help="Skip applying openpi's transformers_replace patch.",
    )
    parser.add_argument(
        "--download-jax", action=argparse.BooleanOptionalAction, default=True,
        help="Download the JAX checkpoint first if it's not present locally "
             "(default: yes).",
    )
    args = parser.parse_args()

    openpi_root = _openpi_root()
    if openpi_root is None:
        _fail(
            "openpi is not importable in this venv. The π0 runtime venv "
            "should already have it — try `emboviz install-pi0 --force`."
        )
    print(f"[convert-pi0] found openpi at: {openpi_root}", flush=True)

    convert_script = openpi_root / "examples" / "convert_jax_model_to_pytorch.py"
    if not convert_script.exists():
        _fail(
            f"openpi convert script not found at {convert_script} "
            "— openpi's layout has changed; update emboviz_pi0/convert.py."
        )

    cache_root = _hf_cache_root()
    jax_dir = Path(args.jax_checkpoint) if args.jax_checkpoint else cache_root / args.config_name
    out_dir = Path(args.output_path) if args.output_path else cache_root / f"{args.config_name}_pytorch"

    # 1) Apply the transformers_replace patch (idempotent).
    if not args.skip_transformers_patch:
        _apply_transformers_replace(openpi_root)

    # 2) Download the JAX checkpoint if missing.
    if not jax_dir.exists() and args.download_jax:
        print(f"[convert-pi0] downloading JAX checkpoint for '{args.config_name}'", flush=True)
        try:
            from openpi.shared import download
            resolved = download.maybe_download(
                f"gs://openpi-assets/checkpoints/{args.config_name}"
            )
        except Exception as e:  # surfaced to the user; not swallowed
            _fail(f"failed to download JAX checkpoint via openpi.shared.download: {e}")
        print(f"             JAX checkpoint at: {resolved}", flush=True)
        jax_dir = Path(resolved)
    elif not jax_dir.exists():
        _fail(f"JAX checkpoint not found at {jax_dir} and --no-download-jax was set.")

    # 3) Run the convert script with THIS venv's interpreter (the pi0 venv).
    cache_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[convert-pi0] running convert script ({args.precision})", flush=True)
    print(f"             jax_dir:  {jax_dir}", flush=True)
    print(f"             out_dir:  {out_dir}", flush=True)
    result = subprocess.run([
        sys.executable, str(convert_script),
        "--checkpoint-dir", str(jax_dir),
        "--config-name", args.config_name,
        "--output-path", str(out_dir),
        "--precision", args.precision,
    ])
    if result.returncode != 0:
        _fail(f"openpi convert script exited with code {result.returncode}")

    # 4) Copy assets/ (norm_stats.json + per-dataset metadata) over.
    assets_src = jax_dir / "assets"
    assets_dst = out_dir / "assets"
    if assets_src.exists():
        if assets_dst.exists():
            print(f"[convert-pi0] assets/ already present in {out_dir}, leaving as-is", flush=True)
        else:
            print("[convert-pi0] copying assets/ from JAX checkpoint", flush=True)
            shutil.copytree(assets_src, assets_dst)
    else:
        print(
            f"[convert-pi0] WARN: no assets/ in {jax_dir} — the adapter may fail "
            "on norm_stats lookup. Check the JAX checkpoint structure.",
            flush=True,
        )

    print(f"[convert-pi0] DONE -> {out_dir}", flush=True)
    print(
        "\nNext: point your run config at the PyTorch backend and analyze:\n"
        "    model:\n"
        "      adapter: pi0\n"
        "      kwargs:\n"
        f"        config_name: {args.config_name}\n"
        "        use_pytorch: true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
