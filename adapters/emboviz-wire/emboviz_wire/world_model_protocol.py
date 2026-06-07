"""The WorldModel protocol.

The third worker contract, alongside :class:`emboviz_wire.model_protocol.
VLAModel` (a policy: Scene → action) and :class:`emboviz_wire.reader_
protocol.EpisodeSource` (a dataset: id → Scenes). A **world model** is the
inverse of a policy: given a conditioning frame and a sequence of actions,
it predicts the *future frames* those actions would produce.

    VLAModel:      observation                      → action
    WorldModel:    observation + action sequence    → future observations

Diagnostics and the runner call only the methods declared here; they never
import a specific world-model package, so swapping one (Cosmos, a learned
latent model, a future foundation model) is a one-line change in the CLI —
the same isolation the other two contracts give.

The prediction is returned as a :class:`emboviz_wire.types.Trajectory` — the
same universal frame-sequence type a dataset reader produces — so a predicted
rollout and a recorded episode are byte-identical on the wire and downstream
(divergence metrics, the Rerun exporter) treats them uniformly. This is what
makes the trust-calibration comparison — predicted rollout vs recorded
episode — a plain operation on two Trajectories.

Two dimensions of declarative metadata travel on every world model:

- ``capabilities``: which prediction directions it supports
  (``FORWARD_DYNAMICS`` today; ``INVERSE_DYNAMICS`` reserved). A caller
  checks the flag before invoking and gets a clean ``NotSupported`` if absent.

- ``supported_domains`` / ``action_dim`` / ``conditioning_camera``: the
  embodiment contract. A world model is trained for specific embodiments
  (Cosmos calls these *domains*, e.g. ``bridge_orig_lerobot``); it consumes a
  fixed-dimension action vector and conditions on one named camera view. The
  caller validates a rollout request against these before paying for an
  expensive generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Optional

import numpy as np

from emboviz_wire.model_protocol import NotSupported
from emboviz_wire.types import Scene, Trajectory


class WorldModelCapability(Flag):
    """What prediction directions a world model exposes.

    Declared by the adapter in ``__init__``; callers gate on these the
    same way diagnostics gate on :class:`emboviz_wire.model_protocol.
    Capability`.
    """

    FORWARD_DYNAMICS = auto()   # rollout(): conditioning frame + actions → future frames
    INVERSE_DYNAMICS = auto()   # actions_from_video(): frames → the actions that produced them


class WorldModel(ABC):
    """The interface every world-model adapter implements.

    Adapters are responsible for:
      • Loading their model checkpoint (and any quantized variant).
      • Mapping a conditioning :class:`Scene` to the model's native image
        conditioning, and the dataset's native action vector to the model's
        expected action encoding (normalization, chunking) — the adapter
        owns every model-specific quirk so the contract stays clean.
      • Returning the predicted future as a :class:`Trajectory` whose frames
        carry the generated images.
    """

    # ----- identification ---------------------------------------------------

    @property
    @abstractmethod
    def model_id(self) -> str:
        """A short stable identifier — e.g. ``'cosmos3-nano'``."""

    @property
    @abstractmethod
    def capabilities(self) -> WorldModelCapability:
        """OR'd flags describing which prediction directions this adapter
        implements."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action vector this world model conditions on,
        in the embodiment's native action space — one row of the ``actions``
        array passed to :meth:`rollout`."""

    @property
    @abstractmethod
    def supported_domains(self) -> frozenset[str]:
        """Embodiment domains this checkpoint was trained for (the model's
        own naming, e.g. ``{'bridge_orig_lerobot'}``). Informational and
        used for a clear error when a caller requests an unsupported one."""

    @property
    def conditioning_camera(self) -> str:
        """The camera role whose image the model conditions on. Defaults to
        ``'primary'``; multi-view world models override."""
        return "primary"

    # ----- forward dynamics: the core method -------------------------------

    @abstractmethod
    def rollout(
        self,
        init: Scene,
        actions: np.ndarray,
        *,
        num_frames: Optional[int] = None,
    ) -> Trajectory:
        """Predict the future frames produced by ``actions`` from ``init``.

        Parameters
        ----------
        init
            The conditioning timestep. The adapter reads the image at
            :attr:`conditioning_camera` from ``init.observations.images``.
        actions
            ``(T, action_dim)`` array in the embodiment's native action
            space — typically the real logged actions of a recorded episode.
            The adapter applies any model-specific normalization / chunking.
        num_frames
            How many future frames to generate. ``None`` lets the adapter
            choose its native default (e.g. one frame per supplied action).

        Returns
        -------
        Trajectory
            The predicted rollout: one frame per generated timestep, each a
            :class:`Scene` carrying the generated image(s). ``metadata``
            records the generation settings (domain, denoise steps, the
            quantization variant) so a downstream verdict can disclose them.

        Capability: ``FORWARD_DYNAMICS``.
        """

    # ----- episode → conditioning actions ----------------------------------

    def prepare_actions(
        self,
        episode: Trajectory,
        *,
        frame_start: int = 0,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """Encode the actions this world model conditions on, from a recorded
        episode — the input to :meth:`rollout`.

        How an episode maps to conditioning actions is **model-specific** (raw
        logged actions, or a per-embodiment encoding such as normalized pose
        deltas), so it lives behind this method rather than in the runner. The
        runner calls it polymorphically and never encodes a model's action
        format itself.

        Default: the per-frame logged actions from
        ``scene.metadata["expert_action"]`` (which the readers populate from the
        dataset's action key), starting at ``frame_start`` and limited to
        ``n_actions`` (all available if ``None``). Adapters whose model consumes
        a different representation override this. Raises if a frame lacks a
        logged action rather than inventing one.
        """
        rows: list[list[float]] = []
        for i in range(max(0, frame_start), len(episode.frames)):
            a = episode.frames[i].metadata.get("expert_action")
            if a is None:
                raise ValueError(
                    f"frame {i} has no 'expert_action' in metadata — the reader did "
                    "not expose logged actions. Configure the dataset's `action` key, "
                    "or use a world model that encodes actions from state."
                )
            rows.append(list(a))
            if n_actions is not None and len(rows) >= int(n_actions):
                break
        if not rows:
            raise ValueError(f"no actions available from frame_start={frame_start}")
        actions = np.asarray(rows, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"prepared actions are not 2-D (T, action_dim): {actions.shape}")
        return actions

    # ----- inverse dynamics (capability-gated; reserved) -------------------

    def actions_from_video(self, frames: Trajectory) -> np.ndarray:
        """Infer the ``(T, action_dim)`` actions that would produce ``frames``.

        Capability: ``INVERSE_DYNAMICS``. The default raises; adapters that
        advertise the flag override it.
        """
        raise NotSupported(f"{self.model_id} does not support inverse dynamics.")

    # ----- request validation ----------------------------------------------

    def validate_rollout(self, init: Scene, actions: np.ndarray) -> Optional[str]:
        """Return None if ``(init, actions)`` is a well-formed rollout
        request, else a human-readable reason.

        Checked at the boundary — a loud error before an expensive
        generation, never a silently reshaped or truncated request.
        """
        cam = self.conditioning_camera
        if cam not in init.observations.images:
            return (
                f"world model conditions on camera '{cam}' but it is missing "
                f"from init.observations.images (have: "
                f"{sorted(init.observations.images)})"
            )
        actions = np.asarray(actions)
        if actions.ndim != 2:
            return f"actions must be 2-D (T, action_dim), got shape {actions.shape}"
        if actions.shape[0] < 1:
            return "actions must contain at least one timestep"
        if actions.shape[1] != self.action_dim:
            return (
                f"actions have action_dim {actions.shape[1]} but this world "
                f"model conditions on action_dim {self.action_dim}"
            )
        return None

    # ----- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Optional: release resources (GPU memory, file handles)."""
