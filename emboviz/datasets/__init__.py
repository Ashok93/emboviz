"""Episode source adapters — one per data format.

Every team's rollouts live in one of:
  • LeRobot v2/v3   — dominant open-source standard
  • RLDS / TFDS     — Google's standard, foundation of Open-X-Embodiment
                      (planned; Phase 7)
  • HDF5            — Robomimic / ALOHA / NVIDIA Isaac Lab + MimicGen
                      (planned; Phase 7)
  • MCAP            — ROS 2 / NVIDIA Isaac SIM default (deployment
                      recordings; planned; Phase 8)
  • Rerun .rrd      — code-first deployment recordings (planned; Phase 8)
  • HuggingFace generic — any other HF-hosted dataset

Every adapter except ``EpisodeSource`` is lazy-imported via PEP 562
``__getattr__``. This means ``import emboviz.datasets`` works in a
core-only install (just numpy + PIL); only when a user actually accesses
``emboviz.datasets.BridgeEpisodeSource`` does the ``lerobot`` extra get
pulled. If the extra isn't installed, the resulting ImportError tells
them exactly which ``pip install emboviz[<extra>]`` they're missing.

The base ``EpisodeSource`` protocol is the only eager import — it's
pure stdlib and tells external callers what interface their custom
adapters must satisfy.
"""

from __future__ import annotations

from emboviz.datasets.base import EpisodeSource

__all__ = [
    "EpisodeSource",
    # ── lerobot adapter family (extra: lerobot) ───────────────────────
    "HuggingFaceEpisodeSource",
    "LeRobotEpisodeSource",
    "BridgeEpisodeSource",
    "BRIDGE_PROFILE",
    "ALOHA_BIMANUAL_PROFILE_1CAM",
    "ALOHA_BIMANUAL_PROFILE_4CAM",
    "AlohaSimTransferCubeSource",
    "AlohaSimInsertionSource",
    "AlohaStatic4CamSource",
    "LIBERO_PROFILE",
    "LiberoSpatialSource",
    "LiberoObjectSource",
    "LiberoGoalSource",
    "Libero10Source",
    "DROID_PROFILE",
    "Droid100Source",
    "DroidFullSource",
    "GR00T_DROID_PROFILE",
    "GR00TDroidSampleSource",
    "PI_LIBERO_PROFILE",
    "PhysicalIntelligenceLiberoSource",
    # ── deployment recording adapters (extras: rerun, mcap) ───────────
    "RerunEpisodeSource",
    "FoxgloveEpisodeSource",
]


# Map of lazily-resolvable names → (module_path, attribute_name, extra_name).
# Each entry says: "if user accesses emboviz.datasets.X, import it from
# this module; if the import fails, suggest this pip extra."
_LAZY_IMPORTS: dict[str, tuple[str, str, str]] = {
    # lerobot family
    "HuggingFaceEpisodeSource": ("emboviz.datasets.huggingface", "HuggingFaceEpisodeSource", "lerobot"),
    "LeRobotEpisodeSource":     ("emboviz.datasets.lerobot",     "LeRobotEpisodeSource",     "lerobot"),
    "BridgeEpisodeSource":      ("emboviz.datasets.lerobot_bridge", "BridgeEpisodeSource",   "lerobot"),
    "BRIDGE_PROFILE":           ("emboviz.datasets.lerobot_bridge", "BRIDGE_PROFILE",        "lerobot"),
    "ALOHA_BIMANUAL_PROFILE_1CAM": ("emboviz.datasets.lerobot_aloha", "ALOHA_BIMANUAL_PROFILE_1CAM", "lerobot"),
    "ALOHA_BIMANUAL_PROFILE_4CAM": ("emboviz.datasets.lerobot_aloha", "ALOHA_BIMANUAL_PROFILE_4CAM", "lerobot"),
    "AlohaSimTransferCubeSource":  ("emboviz.datasets.lerobot_aloha", "AlohaSimTransferCubeSource",  "lerobot"),
    "AlohaSimInsertionSource":     ("emboviz.datasets.lerobot_aloha", "AlohaSimInsertionSource",     "lerobot"),
    "AlohaStatic4CamSource":       ("emboviz.datasets.lerobot_aloha", "AlohaStatic4CamSource",       "lerobot"),
    "LIBERO_PROFILE":              ("emboviz.datasets.lerobot_libero", "LIBERO_PROFILE",              "lerobot"),
    "LiberoSpatialSource":         ("emboviz.datasets.lerobot_libero", "LiberoSpatialSource",         "lerobot"),
    "LiberoObjectSource":          ("emboviz.datasets.lerobot_libero", "LiberoObjectSource",          "lerobot"),
    "LiberoGoalSource":            ("emboviz.datasets.lerobot_libero", "LiberoGoalSource",            "lerobot"),
    "Libero10Source":              ("emboviz.datasets.lerobot_libero", "Libero10Source",              "lerobot"),
    "DROID_PROFILE":               ("emboviz.datasets.lerobot_droid",  "DROID_PROFILE",               "lerobot"),
    "Droid100Source":              ("emboviz.datasets.lerobot_droid",  "Droid100Source",              "lerobot"),
    "DroidFullSource":             ("emboviz.datasets.lerobot_droid",  "DroidFullSource",             "lerobot"),
    "GR00T_DROID_PROFILE":         ("emboviz.datasets.lerobot_droid",  "GR00T_DROID_PROFILE",         "lerobot"),
    "GR00TDroidSampleSource":      ("emboviz.datasets.lerobot_droid",  "GR00TDroidSampleSource",      "lerobot"),
    "PI_LIBERO_PROFILE":           ("emboviz.datasets.lerobot_libero", "PI_LIBERO_PROFILE",           "lerobot"),
    "PhysicalIntelligenceLiberoSource": ("emboviz.datasets.lerobot_libero", "PhysicalIntelligenceLiberoSource", "lerobot"),
    # deployment recording adapters
    "RerunEpisodeSource":     ("emboviz.datasets.rerun",    "RerunEpisodeSource",    "rerun"),
    "FoxgloveEpisodeSource":  ("emboviz.datasets.foxglove", "FoxgloveEpisodeSource", "mcap"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access.

    Resolves an adapter on first access by importing only the module
    that defines it. The result is cached in this module's globals
    so subsequent accesses are zero-cost dict lookups.
    """
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
    """Make tab-completion + IDE inspection list lazy adapters too."""
    return sorted(set(__all__) | set(globals()))
