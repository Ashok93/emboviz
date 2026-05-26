"""`emboviz list-models`, `list-datasets`, `version` — discovery commands.

These never need to load a model. They just enumerate what's available
in the framework + tell the user how to install missing extras.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import click


# Per-adapter extra mapping — for showing the user "install with pip install emboviz[<extra>]".
_MODEL_EXTRAS: dict[str, tuple[str, str]] = {
    # alias  →  (description, extra)
    "openvla":     ("OpenVLA-7B base + fine-tunes (HF AutoModel)",        "openvla"),
    "openvla-oft": ("OpenVLA-OFT (moojink fork; needs separate repo)",    "oft"),
    "pi0":         ("π0 / π0.5 via Physical-Intelligence's openpi repo",  "pi0"),
    "gr00t":       ("NVIDIA Isaac GR00T-N1.7 (needs Isaac-GR00T repo)",   "gr00t"),
    "mock":        ("Deterministic mock policy for testing (no GPU)",     ""),
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
    "droid-sample":     ("droid_sample (3 demo episodes; needs gr00t)",   "gr00t"),
    "aloha-transfer":   ("ALOHA sim transfer cube (lerobot)",             "lerobot"),
    "aloha-insertion":  ("ALOHA sim insertion (lerobot)",                 "lerobot"),
    # planned (Phase 7)
    "rlds":             ("RLDS / Open-X-Embodiment (planned, Phase 7)",   "rlds"),
    "hdf5":             ("Robomimic / ALOHA HDF5 (planned, Phase 7)",     "hdf5"),
    # deployment recordings (Phase 8)
    "mcap":             ("MCAP deployment recording (planned, Phase 8)",  "mcap"),
    "rerun":            ("Rerun .rrd recording (planned, Phase 8)",       "rerun"),
}


def _extra_installed(extra: str) -> bool:
    """Probe whether a given extra's primary package is importable."""
    if not extra:
        return True   # core / no extra needed
    primary = {
        "openvla":  "transformers",
        "oft":      "transformers",
        "pi0":      "transformers",
        "gr00t":    "transformers",   # Isaac-GR00T must also be cloned; we only probe transformers
        "lerobot":  "lerobot",
        "rlds":     "tensorflow_datasets",
        "hdf5":     "h5py",
        "mcap":     "mcap",
        "rerun":    "rerun",
        "viz":      "matplotlib",
    }.get(extra)
    if primary is None:
        return False
    return importlib.util.find_spec(primary) is not None


@click.command("list-models")
def list_models_cmd() -> None:
    """Show the model adapters this install can drive."""
    click.echo("Available model adapters:")
    for alias, (desc, extra) in _MODEL_EXTRAS.items():
        installed = _extra_installed(extra)
        mark = "✓" if installed else "·"
        hint = "" if (installed or not extra) else f"  (install with: pip install 'emboviz[{extra}]')"
        click.echo(f"  {mark} {alias:<14} {desc}{hint}")
    click.echo()
    click.echo("Use: emboviz analyze --model <alias> ...")


@click.command("list-datasets")
def list_datasets_cmd() -> None:
    """Show the dataset / recording adapters this install can read."""
    click.echo("Available dataset / recording adapters:")
    for alias, (desc, extra) in _DATASET_EXTRAS.items():
        installed = _extra_installed(extra)
        mark = "✓" if installed else "·"
        hint = "" if (installed or not extra) else f"  (install with: pip install 'emboviz[{extra}]')"
        click.echo(f"  {mark} {alias:<18} {desc}{hint}")
    click.echo()
    click.echo("Use: emboviz analyze --dataset <alias> ...")


@click.command("version")
def version_cmd() -> None:
    """Print the installed emboviz version."""
    from emboviz import __version__
    click.echo(f"emboviz {__version__}")
    click.echo(f"python {sys.version.split()[0]}")
