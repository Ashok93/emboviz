"""Marginal-distribution sample pools for modality dropout.

For each input modality (image-per-camera, state, gripper,
action_history, instruction), build a pool of ``n_samples`` real values
drawn from the dataset's empirical marginal — sampled from EPISODES
OTHER than the one under test, so the substitution is uncorrelated with
the current trajectory.

This is the SHAP / Janzing-Minorics-Blöbaum prescription for causally-
interpretable feature attribution:

  • Marginal sampling = do-intervention semantics (Pearl). The
    substitute value comes from the population, not from the joint with
    the held-out modalities. See:
      - Janzing, Minorics, Blöbaum 2020 "Feature relevance quantification
        in explainable AI: A causal problem" (AISTATS, arXiv:1910.13413)
      - Štrumbelj & Kononenko 2014, KIS

  • NEVER use zeros / midpoints / single-trajectory substitutes — those
    are NOT samples from the marginal and cause false "ignored" verdicts
    (the substitute happens to coincide with the model's null prior).

  • Excluding the current episode avoids Hooker & Mentch's "permutation
    forces extrapolation" failure mode for autocorrelated time series:
      - Hooker & Mentch 2019 (arXiv:1905.03151) — same-trajectory
        permutations push the model to extrapolate; cross-episode
        samples are still in-distribution.

Pool size: 20-50 per modality per query frame is the practical balance
between variance (O(1/√N) per RISE / Monte-Carlo Shapley) and inference
cost. We default to 20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from emboviz.core.types import Scene


@dataclass
class ModalityPool:
    """Per-modality empirical samples + a "minimum meaningful intervention"
    threshold derived from intra-pool pairwise distances.

    ``ref_distance[modality]`` is the 25th percentile of pairwise
    distances within the pool. If a candidate substitution's distance
    to the current frame's value is BELOW this, the substitution is too
    similar to count as a real intervention and the diagnostic must
    abstain rather than report "modality ignored."
    """

    state_samples: list[np.ndarray]          = field(default_factory=list)
    gripper_samples: list[float]             = field(default_factory=list)
    action_history_samples: list[np.ndarray] = field(default_factory=list)
    instruction_samples: list[str]           = field(default_factory=list)
    image_samples: dict[str, list[Any]]      = field(default_factory=dict)   # camera → list of PIL.Image
    ref_distance: dict[str, float]           = field(default_factory=dict)
    metadata: dict                           = field(default_factory=dict)

    def has(self, modality: str) -> bool:
        """Whether this pool has any samples for the requested modality.

        Accepts:
          - "state"
          - "gripper"
          - "action_history"
          - "instruction"
          - "image:<camera>"
        """
        if modality == "state":          return bool(self.state_samples)
        if modality == "gripper":        return bool(self.gripper_samples)
        if modality == "action_history": return bool(self.action_history_samples)
        if modality == "instruction":    return bool(self.instruction_samples)
        if modality.startswith("image:"):
            cam = modality.split(":", 1)[1]
            return cam in self.image_samples and bool(self.image_samples[cam])
        raise ValueError(f"Unknown modality '{modality}'")

    def sample(
        self, modality: str, k: int, rng: np.random.Generator,
        current_value: Any = None,
    ) -> list:
        """Draw ``k`` substitutions for ``modality`` from the pool.

        We always sample without replacement when the pool is large
        enough; if k > pool size we sample WITH replacement and warn via
        metadata. Drawn samples that are exactly equal to
        ``current_value`` (np.array_equal / str equality) are filtered
        out — the marginal distribution may legitimately produce a
        sample identical to the current frame for binary gripper or
        repeated instructions.
        """
        pool_attr = {
            "state":          self.state_samples,
            "gripper":        self.gripper_samples,
            "action_history": self.action_history_samples,
            "instruction":    self.instruction_samples,
        }
        if modality in pool_attr:
            pool = pool_attr[modality]
        elif modality.startswith("image:"):
            cam = modality.split(":", 1)[1]
            pool = self.image_samples.get(cam, [])
        else:
            raise ValueError(f"Unknown modality '{modality}'")

        if not pool:
            return []

        # Filter exact duplicates of current_value
        def _eq(a, b):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return a.shape == b.shape and bool(np.array_equal(a, b))
            return a == b

        if current_value is not None:
            filtered = [p for p in pool if not _eq(p, current_value)]
        else:
            filtered = list(pool)
        if not filtered:
            return []

        if k > len(filtered):
            replace = True
        else:
            replace = False
        idx = rng.choice(len(filtered), size=k, replace=replace)
        return [filtered[int(i)] for i in idx]


def _distance(modality: str, a: Any, b: Any) -> float:
    """Natural per-modality distance.

    state/action_history → L2.
    gripper              → absolute difference.
    instruction          → 1 - Jaccard similarity over whitespace tokens
                           (lightweight; no embedding model).
    image:<cam>          → mean pixel-L2 across the image, normalized to
                           [0, 1] by sqrt(3)*255 (max possible).
    """
    if modality in ("state", "action_history"):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    if modality == "gripper":
        return float(abs(float(a) - float(b)))
    if modality == "instruction":
        ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
        if not ta and not tb:
            return 0.0
        inter, union = len(ta & tb), len(ta | tb)
        return float(1.0 - (inter / max(union, 1)))
    if modality.startswith("image:"):
        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
        if arr_a.shape != arr_b.shape:
            raise ValueError(
                f"image distance: shape mismatch {arr_a.shape} vs {arr_b.shape}"
            )
        diff = arr_a - arr_b
        per_pixel = np.linalg.norm(diff, axis=-1)
        max_per_pixel = float(np.sqrt(arr_a.shape[-1]) * 255.0)
        return float(per_pixel.mean() / max_per_pixel)
    raise ValueError(f"distance: unknown modality '{modality}'")


def build_modality_pool(
    dataset,
    current_episode: int,
    declared_modalities: dict,
    *,
    n_samples: int = 20,
    cameras: Optional[list[str]] = None,
    seed: int = 0,
    instruction_must_differ_from_task: Optional[str] = None,
) -> ModalityPool:
    """Build a ModalityPool sampled from episodes OTHER than current_episode.

    Args:
        dataset: ``EpisodeSource`` (must implement ``list_episodes`` and
            ``load_trajectory``).
        current_episode: episode under test; we exclude it from sampling.
        declared_modalities: dict of which modalities to build pools for.
            Example: ``{"state": True, "gripper": True,
                        "action_history": False, "instruction": True,
                        "images": ["primary", "wrist"]}``.
        n_samples: target pool size per modality (uses fewer if dataset
            has fewer episodes).
        cameras: override which cameras to build image pools for.
        seed: deterministic RNG seed.
        instruction_must_differ_from_task: if set, drops sampled
            instructions that match this exact string (used by the
            runner to ensure substitutions differ from the current task).

    Returns:
        Populated :class:`ModalityPool` with ``ref_distance[modality]``
        equal to the 25th percentile of intra-pool pairwise distances.
        Used by ``ModalityDropoutDiagnostic`` to decide whether a
        candidate substitution constitutes a "real intervention."
    """
    rng = np.random.default_rng(seed)

    all_episodes = [int(e) for e in dataset.list_episodes()]
    other_episodes = [e for e in all_episodes if e != current_episode]
    if not other_episodes:
        raise ValueError(
            f"build_modality_pool: dataset has only {len(all_episodes)} "
            f"episode(s); cannot sample from OTHER episodes for marginal "
            f"distribution. Modality dropout needs at least 2 episodes."
        )

    # Sample one random frame from each of up to n_samples other episodes.
    pick = min(n_samples, len(other_episodes))
    chosen = rng.choice(other_episodes, size=pick, replace=False).tolist()

    pool = ModalityPool()
    pool.metadata["sampled_episodes"] = [int(e) for e in chosen]
    pool.metadata["n_requested"]      = int(n_samples)
    pool.metadata["n_available"]      = int(len(other_episodes))

    want_state    = bool(declared_modalities.get("state"))
    want_gripper  = bool(declared_modalities.get("gripper"))
    want_history  = bool(declared_modalities.get("action_history"))
    want_instr    = bool(declared_modalities.get("instruction"))
    want_cams     = list(declared_modalities.get("images") or []) if cameras is None else list(cameras)
    for cam in want_cams:
        pool.image_samples[cam] = []

    for ep_idx in chosen:
        traj = dataset.load_trajectory(int(ep_idx))
        if not traj.frames:
            continue
        # Pick a random frame from this episode.
        fi = int(rng.choice(len(traj.frames)))
        scene = traj.frames[fi]
        obs = scene.observations

        if want_state and obs.state is not None:
            pool.state_samples.append(
                np.asarray(obs.state.values, dtype=np.float32).copy()
            )
        if want_gripper and obs.gripper is not None:
            pool.gripper_samples.append(float(obs.gripper.value))
        if want_history and obs.action_history is not None:
            pool.action_history_samples.append(
                np.asarray(obs.action_history.actions, dtype=np.float32).copy()
            )
        if want_instr and scene.instruction:
            if (instruction_must_differ_from_task is None
                    or scene.instruction != instruction_must_differ_from_task):
                pool.instruction_samples.append(str(scene.instruction))
        for cam in want_cams:
            if cam in obs.images:
                pool.image_samples[cam].append(obs.images[cam].data)

    # Compute reference distances (25th percentile of intra-pool pairwise
    # distances). The "minimum meaningful intervention" threshold —
    # candidate substitutions with distance below this are flagged as
    # uninformative and the diagnostic abstains.
    def _ref(pool_list, modality):
        if len(pool_list) < 2:
            return 0.0
        # All-pairs distance can be O(N²); for N=20 that's 190 pairs. Fine.
        ds = []
        for i in range(len(pool_list)):
            for j in range(i + 1, len(pool_list)):
                ds.append(_distance(modality, pool_list[i], pool_list[j]))
        if not ds:
            return 0.0
        return float(np.percentile(np.asarray(ds), 25))

    pool.ref_distance["state"]          = _ref(pool.state_samples,          "state")
    pool.ref_distance["gripper"]        = _ref(pool.gripper_samples,        "gripper")
    pool.ref_distance["action_history"] = _ref(pool.action_history_samples, "action_history")
    pool.ref_distance["instruction"]    = _ref(pool.instruction_samples,    "instruction")
    for cam in want_cams:
        pool.ref_distance[f"image:{cam}"] = _ref(pool.image_samples[cam], f"image:{cam}")

    return pool
