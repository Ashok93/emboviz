"""Shared CLI helpers — load a model, load a scene, build a suite by name."""

from __future__ import annotations

from typing import Optional

from emboviz.core.types import Scene, Trajectory
from emboviz.models.protocol import VLAModel
from emboviz.models.registry import get_model
from emboviz.suites.base import Suite


def load_model(name: str, **kwargs) -> VLAModel:
    factory = get_model(name)
    return factory(**kwargs)


def load_scene_bridge(episode: int, frame_offset: int = 0) -> Scene:
    """Load one Scene from BridgeV2."""
    from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource
    src = BridgeEpisodeSource()
    scenes = src.load_episode(str(episode))
    if not scenes:
        raise ValueError(f"Episode {episode} produced no scenes")
    return scenes[min(frame_offset, len(scenes) - 1)]


def load_scene(scene_spec: str) -> Scene:
    """Parse a `scene_spec` like 'bridge:0:5' (dataset:episode:frame) or
    raise if unsupported."""
    parts = scene_spec.split(":")
    if parts[0] == "bridge":
        ep = int(parts[1])
        frame = int(parts[2]) if len(parts) > 2 else 0
        return load_scene_bridge(ep, frame)
    raise ValueError(f"Unknown scene spec: {scene_spec}")


def load_trajectory(traj_spec: str) -> Trajectory:
    """Parse a trajectory spec like 'bridge:0' (dataset:episode)."""
    parts = traj_spec.split(":")
    if parts[0] == "bridge":
        from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource
        return BridgeEpisodeSource().load_trajectory(int(parts[1]))
    raise ValueError(f"Unknown trajectory spec: {traj_spec}")


def build_suite(name: str) -> Suite:
    if name == "language_grounding":
        from emboviz.suites.language_grounding import build_language_grounding_suite
        return build_language_grounding_suite()
    if name == "visual_robustness":
        from emboviz.suites.visual_robustness import build_visual_robustness_suite
        return build_visual_robustness_suite()
    if name == "full_profile":
        from emboviz.suites.full_profile import build_full_profile
        return build_full_profile()
    if name == "quick_smoke":
        from emboviz.suites.quick_smoke import build_quick_smoke
        return build_quick_smoke()
    raise ValueError(f"Unknown suite: {name}")
