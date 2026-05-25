"""Dataset adapters — load episodes from various sources as `Scene` objects.

Datasets are decoupled from models — same Scene flows through any VLA.
"""

from emboviz.datasets.base import EpisodeSource

__all__ = ["EpisodeSource"]
