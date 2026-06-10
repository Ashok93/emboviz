"""Tests for the world-model history surface of the wire — handler-level, no ZMQ.

A mock ``WorldModel`` that conditions on history is driven through
``WorldModelHandler`` with codec-round-tripped arguments, verifying that the
``history`` Trajectory (including per-frame pose, gripper, and a numpy latent
in metadata) survives encode→decode, and that ``conditions_on_history`` rides
the static metadata.

Run::

    uv run python adapters/emboviz-wire/tests/test_world_model_history_wire.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire import WorldModelHandler, wire
from emboviz_wire.observations import GripperState, Proprioception, RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel, WorldModelCapability


def _scene(value: int, with_latent: bool) -> Scene:
    meta = {"ctrlworld_latent": np.full((4, 72, 40), value, np.float16)} if with_latent else {}
    return Scene(
        observations=Observations(
            images={"primary": RGBImage(data=np.full((6, 6, 3), value, np.uint8), camera_id="primary")},
            state=Proprioception(values=np.arange(6, dtype=np.float32) + value, convention="ee_pose"),
            gripper=GripperState(value=0.25),
        ),
        instruction="pick the marker",
        metadata=meta,
    )


class _HistoryWM(WorldModel):
    def __init__(self) -> None:
        self.seen = {}

    @property
    def model_id(self) -> str:
        return "mock-history-wm"

    @property
    def capabilities(self) -> WorldModelCapability:
        return WorldModelCapability.FORWARD_DYNAMICS

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def supported_domains(self) -> frozenset:
        return frozenset({"mock"})

    @property
    def conditions_on_history(self) -> bool:
        return True

    def rollout(self, init, actions, *, history=None, num_frames=None) -> Trajectory:
        self.seen = {"init": init, "actions": np.asarray(actions), "history": history}
        return Trajectory(frames=[_scene(9, with_latent=True)], fps=5.0)


def test_history_round_trips_through_handler() -> None:
    wm = _HistoryWM()
    handler = WorldModelHandler(wm)

    meta = handler.methods["static_metadata"]({})
    assert meta["conditions_on_history"] is True

    history = Trajectory(frames=[_scene(0, with_latent=False), _scene(1, with_latent=True)], fps=0.0)
    args = wire.unpack(wire.pack({
        "init": wire.encode_scene(_scene(1, with_latent=True)),
        "actions": np.zeros((4, 7), np.float32),
        "history": wire.encode_trajectory(history),
        "num_frames": None,
    }))
    result = handler.methods["rollout"](args)

    got = wm.seen["history"]
    assert got is not None and len(got.frames) == 2
    seed = got.frames[0]
    assert seed.observations.state.convention == "ee_pose"
    assert float(seed.observations.gripper.value) == 0.25
    assert "ctrlworld_latent" not in seed.metadata
    anchor = got.frames[1]
    latent = np.asarray(anchor.metadata["ctrlworld_latent"])
    assert latent.shape == (4, 72, 40) and latent.dtype == np.float16

    # The reply (a generated frame with its latent) survives the codec too.
    out = wire.decode_trajectory(wire.unpack(wire.pack(result)))
    assert np.asarray(out.frames[0].metadata["ctrlworld_latent"]).dtype == np.float16


def test_history_none_stays_none() -> None:
    wm = _HistoryWM()
    handler = WorldModelHandler(wm)
    handler.methods["rollout"]({
        "init": wire.encode_scene(_scene(1, with_latent=True)),
        "actions": np.zeros((4, 7), np.float32),
        "num_frames": None,
    })
    assert wm.seen["history"] is None


def _run_all() -> None:
    test_history_round_trips_through_handler()
    test_history_none_stays_none()
    print("OK: all world-model history wire checks passed")


if __name__ == "__main__":
    _run_all()
