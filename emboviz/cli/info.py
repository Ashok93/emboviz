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

from emboviz.adapters import list_adapters, list_world_models


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

# The only in-process model: a GPU-free deterministic policy for testing
# the diagnostic side. Every real model is an isolated ZMQ adapter worker.
_INPROC_MODELS: dict[str, str] = {
    "mock": "Deterministic mock policy for testing (no GPU).",
}

# The three self-describing dataset input formats (config `dataset.format`).
# Each entry: (description, module-that-indicates-availability, install hint).
# LeRobot is read by the ISOLATED emboviz-lerobot worker, so its presence
# is indicated by the reader SHIM in the host (emboviz_lerobot), not an
# in-host lerobot. HDF5's h5py ships in core. RLDS needs the rlds extra.
_DATASET_FORMATS: dict[str, tuple[str, str, str]] = {
    "lerobot": ("LeRobot v3.0 (BridgeV2, LIBERO, DROID, ALOHA, custom HF)",
                "emboviz_lerobot", ""),       # ships with emboviz core
    "gr00t":   ("GR00T format — LeRobot v2.1 + modality.json (NVIDIA Isaac-GR00T)",
                "emboviz_reader_gr00t", ""),  # ships with emboviz core
    "hdf5":    ("Robomimic / ALOHA / Isaac Lab Mimic HDF5", "h5py", ""),
    "rlds":    ("RLDS / TFDS (Open-X-Embodiment, RT-X, Octo)",
                "tensorflow_datasets", "uv sync --extra rlds"),
}


def _import_check(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


@click.command("list-models")
def list_models_cmd() -> None:
    """Show the model adapters this install can drive.

    For each adapter we report:

      ✓  the adapter shim is installed (its entry point is discoverable)
      ·  not installed — shows the ``uv sync --extra <name>`` command to get it
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
                f"           install:  uv sync --extra {name}"
            )

    click.echo()
    click.echo("In-process built-ins:")
    for name, desc in _INPROC_MODELS.items():
        click.echo(f"  ✓ {name:<10} {desc}")

    click.echo()
    click.echo("Use: emboviz analyze --config <file>   (templates under configs/)")


_WORLD_MODEL_DESCRIPTIONS: dict[str, str] = {
    "cosmos3": "NVIDIA Cosmos3-Nano — action-conditioned forward dynamics via vLLM-Omni.",
}


@click.command("list-world-models")
def list_world_models_cmd() -> None:
    """Show the world models this install can drive.

    World models predict future frames from a conditioning frame + actions
    (the substrate for trust-calibration and rollout diagnostics). Like the
    VLA adapters, each runs as an isolated worker; the heavy model lives in a
    separate inference server.

      ✓  the world-model shim is installed (its entry point is discoverable)
      ·  not installed — shows the ``uv sync --extra <name>`` command to get it
    """
    installed = list_world_models()
    click.echo("Available world models (ZMQ workers):")

    all_known = sorted(set(_WORLD_MODEL_DESCRIPTIONS) | set(installed))
    for name in all_known:
        desc = _WORLD_MODEL_DESCRIPTIONS.get(name, "")
        mark = "✓" if name in installed else "·"
        if name in installed:
            click.echo(f"  {mark} {name:<10} {desc}")
        else:
            click.echo(
                f"  {mark} {name:<10} {desc}\n"
                f"           install:  uv sync --extra {name}"
            )


@click.command("list-datasets")
def list_datasets_cmd() -> None:
    """Show the dataset input formats this install can read.

    A run config's ``dataset.format`` is one of these three self-describing
    formats; emboviz reads dims/per-dim names from each format's own schema.
    """
    click.echo("Dataset input formats (config `dataset.format`):")
    for fmt, (desc, indicator, install) in _DATASET_FORMATS.items():
        ok = _import_check(indicator)
        mark = "✓" if ok else "·"
        hint = "" if ok or not install else f"  (install with: {install})"
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
