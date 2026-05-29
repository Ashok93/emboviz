"""Episode source adapters — one per dataset input format.

The framework reads three self-describing dataset formats:
  • LeRobot v2.x   — read by the ISOLATED ``emboviz-lerobot`` worker
                     (its venv pins lerobot; core never imports it). The
                     host gets a ``ZMQReaderClient`` via
                     ``emboviz.adapters.connect_reader`` — see
                     ``emboviz.datasets.manifest``.
  • HDF5           — Robomimic / ALOHA / Isaac Lab Mimic; read in-process.
  • RLDS / TFDS    — Open-X-Embodiment / RT-X / Octo; read in-process.

The in-process readers (HDF5, RLDS) are lazy-imported via PEP 562
``__getattr__`` so ``import emboviz.datasets`` works in a core-only
install; the heavy extra is only pulled when the reader is actually
accessed. The base ``EpisodeSource`` contract lives in ``emboviz-wire``
(re-exported here via :mod:`emboviz.datasets.base`).
"""

from __future__ import annotations

from emboviz.datasets.base import EpisodeSource

__all__ = [
    "EpisodeSource",
    "HDF5EpisodeSource",   # extra: hdf5 (h5py — in core)
    "RLDSEpisodeSource",   # extra: rlds (tensorflow_datasets)
]


# Map of lazily-resolvable names → (module_path, attribute_name, extra_name).
# LeRobot is intentionally absent: it is not an in-process reader — it runs
# in the isolated ``emboviz-lerobot`` worker (see manifest._build_lerobot).
_LAZY_IMPORTS: dict[str, tuple[str, str, str]] = {
    "HDF5EpisodeSource": ("emboviz.datasets.hdf5", "HDF5EpisodeSource", "hdf5"),
    "RLDSEpisodeSource": ("emboviz.datasets.rlds", "RLDSEpisodeSource", "rlds"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access — import an in-process reader on first
    use, suggesting its pip extra if the import fails."""
    entry = _LAZY_IMPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name, extra = entry
    try:
        import importlib
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"emboviz.datasets.{name} requires the '{extra}' extra. "
            f"Install with: pip install 'emboviz[{extra}]'.  "
            f"Underlying ImportError: {e}"
        ) from e
    value = getattr(module, attr_name)
    globals()[name] = value  # cache for next time
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
