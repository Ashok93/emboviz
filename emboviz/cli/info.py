"""`emboviz list-models`, `list-datasets`, `version` — discovery commands.

These never need to load a model. They enumerate what's installed in
the user's environment + show the install command for missing
adapters.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import click

from emboviz.adapters import list_adapters


# Human-readable descriptions for each registered adapter alias. The
# alias itself comes from the entry-point registry — this dict only
# stores what to show next to it. Adapters not in this dict still
# appear; they just print without a description.
_ADAPTER_DESCRIPTIONS: dict[str, str] = {
    "openvla":  "OpenVLA-7B (Stanford). LLaMA-2 7B + SigLIP+DINOv2 vision tower.",
    "oft":      "OpenVLA-OFT — parallel decoding + L1 action head; LIBERO fine-tunes.",
    "pi0":      "Physical Intelligence π0 / π0.5 via openpi.",
    "gr00t":    "NVIDIA GR00T-N1.7 (3B) — Qwen3-VL backbone + diffusion expert.",
    "sam3":     "Meta SAM 3 — text→mask detector for memorization & target-aware diagnostics.",
}

# Built-in in-process models that don't go through the ZMQ adapter path.
_LEGACY_INPROC_MODELS: dict[str, str] = {
    "mock":    "Deterministic mock policy for testing (no GPU).",
    "lerobot": "Stock LeRobotDataset policy (CPU OK).",
}

# The three self-describing dataset input formats (config `dataset.format`).
# emboviz reads each format's own schema (info.json / array shapes / TFDS
# feature spec) — these are the only `dataset.format` values a run config
# accepts. (Rerun/MCAP are recording-viz formats, not dataset inputs.)
_DATASET_FORMAT_EXTRAS: dict[str, tuple[str, str]] = {
    "lerobot":  ("LeRobot v2/v3 (BridgeV2, LIBERO, DROID, ALOHA, custom HF)", "lerobot"),
    "hdf5":     ("Robomimic / ALOHA / Isaac Lab Mimic HDF5",                  ""),
    "rlds":     ("RLDS / TFDS (Open-X-Embodiment, RT-X, Octo)",              "rlds"),
}


def _import_check(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _dataset_extra_installed(extra: str) -> bool:
    if not extra:
        return True
    primary = {
        "lerobot":  "lerobot",
        "rlds":     "tensorflow_datasets",
        "hdf5":     "h5py",
        "mcap":     "mcap",
        "rerun":    "rerun",
        "viz":      "matplotlib",
    }.get(extra)
    if primary is None:
        return False
    return _import_check(primary)


@click.command("list-models")
def list_models_cmd() -> None:
    """Show the model adapters this install can drive.

    For each adapter we report:

      ✓  the adapter shim is installed (its entry point is discoverable)
      ·  not installed — shows the ``uv pip install`` + ``emboviz install-<name>``
         commands to get it
    """
    installed = list_adapters()
    click.echo("Available model adapters (ZMQ workers):")

    all_known = sorted(set(_ADAPTER_DESCRIPTIONS) | set(installed))
    for name in all_known:
        desc = _ADAPTER_DESCRIPTIONS.get(name, "")
        mark = "✓" if name in installed else "·"
        if name in installed:
            click.echo(f"  {mark} {name:<10} {desc}")
        else:
            click.echo(
                f"  {mark} {name:<10} {desc}\n"
                f"           install:  uv pip install emboviz-{name}\n"
                f"                      emboviz install-{name}\n"
                f"                      emboviz-{name} serve &"
            )

    click.echo()
    click.echo("In-process built-ins:")
    for name, desc in _LEGACY_INPROC_MODELS.items():
        click.echo(f"  ✓ {name:<10} {desc}")

    click.echo()
    click.echo("Use: emboviz analyze --config <file>   (templates under configs/)")


@click.command("list-datasets")
def list_datasets_cmd() -> None:
    """Show the dataset input formats this install can read.

    A run config's ``dataset.format`` is one of these three self-describing
    formats; emboviz reads dims/per-dim names from each format's own schema.
    """
    click.echo("Dataset input formats (config `dataset.format`):")
    for fmt, (desc, extra) in _DATASET_FORMAT_EXTRAS.items():
        ok = _dataset_extra_installed(extra) if extra else True
        mark = "✓" if ok else "·"
        hint = "" if ok else f"  (install with: uv pip install 'emboviz[{extra}]')"
        click.echo(f"  {mark} {fmt:<10} {desc}{hint}")
    click.echo()
    click.echo("Each config maps camera roles / state convention / gripper for its")
    click.echo("format uniformly. See configs/ for ready-made model+dataset templates,")
    click.echo("and configs/README.md for the full field reference.")


@click.command("version")
def version_cmd() -> None:
    """Print the installed emboviz version."""
    from emboviz import __version__
    click.echo(f"emboviz {__version__}")
    click.echo(f"python {sys.version.split()[0]}")
