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

_DATASET_EXTRAS: dict[str, tuple[str, str]] = {
    "bridge":           ("BridgeV2 (lerobot v2.0)",                       "lerobot"),
    "libero-spatial":   ("LIBERO-spatial (lerobot)",                      "lerobot"),
    "libero-object":    ("LIBERO-object (lerobot)",                       "lerobot"),
    "libero-goal":      ("LIBERO-goal (lerobot)",                         "lerobot"),
    "libero-10":        ("LIBERO-10 (lerobot)",                           "lerobot"),
    "pi-libero":        ("Physical Intelligence's LIBERO conversion",     "lerobot"),
    "droid-100":        ("DROID 100-episode subset (lerobot)",            "lerobot"),
    "droid-full":       ("DROID 1.0.1 full (76K episodes; lerobot)",      "lerobot"),
    "droid-sample":     ("droid_sample (3 demo episodes; needs gr00t)",   ""),
    "aloha-transfer":   ("ALOHA sim transfer cube (lerobot)",             "lerobot"),
    "aloha-insertion":  ("ALOHA sim insertion (lerobot)",                 "lerobot"),
}


_DATASET_FORMAT_EXTRAS: dict[str, tuple[str, str]] = {
    "hdf5":         ("Robomimic / ALOHA / Isaac Lab Mimic HDF5",          ""),
    "rlds":         ("RLDS / TFDS (Open-X-Embodiment, RT-X, ...)",        "rlds"),
    "mcap":         ("MCAP deployment recording (ROS 2 / Isaac SIM)",     ""),
    "rerun-rrd":    ("Rerun .rrd deployment recording",                   ""),
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
    click.echo("Use: emboviz analyze --model <alias> ...")


@click.command("list-datasets")
def list_datasets_cmd() -> None:
    """Show the dataset / recording adapters this install can read."""
    click.echo("Pre-shipped dataset aliases (use: --dataset <alias>):")
    for alias, (desc, extra) in _DATASET_EXTRAS.items():
        ok = _dataset_extra_installed(extra)
        mark = "✓" if ok else "·"
        hint = "" if (ok or not extra) else f"  (install with: uv pip install 'emboviz[{extra}]')"
        click.echo(f"  {mark} {alias:<18} {desc}{hint}")
    click.echo()
    click.echo("Generic data formats (use: --dataset-format <fmt> --dataset-path <path>):")
    for fmt, (desc, extra) in _DATASET_FORMAT_EXTRAS.items():
        ok = _dataset_extra_installed(extra) if extra else True
        mark = "✓" if ok else "·"
        hint = "" if ok else f"  (install with: uv pip install 'emboviz[{extra}]')"
        click.echo(f"  {mark} {fmt:<14} {desc}{hint}")
    click.echo()
    click.echo("Pass adapter-specific options (camera_keys, builder_name, topic_map, ...)")
    click.echo("via --dataset-kwargs '<JSON>'.")


@click.command("version")
def version_cmd() -> None:
    """Print the installed emboviz version."""
    from emboviz import __version__
    click.echo(f"emboviz {__version__}")
    click.echo(f"python {sys.version.split()[0]}")
