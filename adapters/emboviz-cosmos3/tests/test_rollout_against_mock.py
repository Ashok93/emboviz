"""End-to-end test of Cosmos3WorldModel.rollout against a mock vLLM-Omni server.

Proves the adapter's HTTP plumbing without a GPU: request shaping (multipart
form + extra_params), autoregressive chunk splitting, MP4 decoding, dropping
the conditioning frame, and Trajectory assembly. The mock stands in for
``POST /v1/videos/sync``: it returns the conditioning frame (solid red) followed
by ``action_chunk_size`` generated frames (solid grayscale), so the test can
verify the red conditioning frame is dropped and one generated frame is kept
per action.

Run (deps live in the worker venv, not core, so pull them in for the test)::

    uv run --with requests --with imageio --with av --with pillow \
        python adapters/emboviz-cosmos3/tests/test_rollout_against_mock.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory

from emboviz_cosmos3.model import Cosmos3WorldModel


# ── multipart parsing (server side) ─────────────────────────────────────────


def _parse_multipart(body: bytes, boundary: str) -> dict[str, bytes]:
    """Split a multipart/form-data body into ``{field_name: raw_bytes}``."""
    delim = ("--" + boundary).encode()
    fields: dict[str, bytes] = {}
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        head, _, payload = part.partition(b"\r\n\r\n")
        head_text = head.decode("latin-1")
        name = None
        for token in head_text.split(";"):
            token = token.strip()
            if token.startswith('name="'):
                name = token[len('name="'):].rstrip('"')
                break
        if name is not None:
            fields[name] = payload
    return fields


def _encode_mp4(frames: np.ndarray, fps: int) -> bytes:
    import imageio.v3 as iio

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.mp4"
        iio.imwrite(path, frames, plugin="pyav", codec="libx264",
                    fps=fps, out_pixel_format="yuv420p")
        return path.read_bytes()


# ── the mock server ─────────────────────────────────────────────────────────


class _MockCosmos:
    """A threaded HTTP stand-in for the vLLM-Omni forward-dynamics endpoint."""

    def __init__(self) -> None:
        self.requested_chunk_sizes: list[int] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):  # silence
                pass

            def do_POST(self):
                ctype = self.headers["Content-Type"] or ""
                boundary = ctype.split("boundary=", 1)[1]
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                fields = _parse_multipart(body, boundary)

                extra = json.loads(fields["extra_params"].decode())
                chunk_size = int(extra["action_chunk_size"])
                width, height = (int(x) for x in fields["size"].decode().split("x"))
                num_frames = int(fields["num_frames"].decode())
                assert num_frames == chunk_size + 1
                outer.requested_chunk_sizes.append(chunk_size)

                # Frame 0: solid red conditioning frame (must be dropped).
                # Frames 1..k: solid grayscale, distinct per index.
                frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
                frames[0] = (255, 0, 0)
                for j in range(1, num_frames):
                    level = min(20 + j * 12, 240)
                    frames[j] = level
                mp4 = _encode_mp4(frames, fps=int(fields["fps"].decode()))

                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(mp4)))
                self.end_headers()
                self.wfile.write(mp4)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_MockCosmos":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


# ── helpers ──────────────────────────────────────────────────────────────────


def _scene(width: int = 64, height: int = 48) -> Scene:
    img = np.full((height, width, 3), (10, 200, 60), dtype=np.uint8)  # greenish
    return Scene(
        observations=Observations(images={"primary": RGBImage(data=img, camera_id="primary")}),
        instruction="pick up the cup",
    )


def _model(port: int, **kw) -> Cosmos3WorldModel:
    return Cosmos3WorldModel(
        server_url=f"http://127.0.0.1:{port}",
        domain_name="agibotworld",
        action_dim=29,
        action_chunk_size=8,
        **kw,
    )


# ── tests ─────────────────────────────────────────────────────────────────────


def test_rollout_shapes_and_chunking() -> None:
    with _MockCosmos() as mock:
        model = _model(mock.port)
        actions = np.random.RandomState(0).randn(20, 29).astype(np.float32)
        traj = model.rollout(_scene(), actions)

        assert isinstance(traj, Trajectory)
        # 20 actions -> 20 generated frames (one per action).
        assert len(traj.frames) == 20
        # Chunked 20 into [8, 8, 4] -> three server requests.
        assert mock.requested_chunk_sizes == [8, 8, 4]
        assert traj.metadata["n_chunks"] == 3
        assert traj.metadata["domain_name"] == "agibotworld"

        # Every kept frame is grayscale (R≈G≈B): the red conditioning frame
        # at index 0 of each chunk was dropped, not kept.
        for scene in traj.frames:
            f = np.asarray(scene.observations.images["primary"].data).astype(int)
            assert f.dtype != object and f.shape == (48, 64, 3)
            spread = f.max(axis=2) - f.min(axis=2)
            assert spread.mean() < 25, "a red conditioning frame leaked through"


def test_num_frames_truncates_actions() -> None:
    with _MockCosmos() as mock:
        model = _model(mock.port)
        actions = np.zeros((20, 29), dtype=np.float32)
        traj = model.rollout(_scene(), actions, num_frames=5)
        assert len(traj.frames) == 5
        assert mock.requested_chunk_sizes == [5]


def test_action_normalizer_applied() -> None:
    seen: list[np.ndarray] = []

    def normalizer(a: np.ndarray) -> np.ndarray:
        seen.append(a)
        return a * 0.0  # map everything to zero, element-wise

    with _MockCosmos() as mock:
        model = _model(mock.port, action_normalizer=normalizer)
        model.rollout(_scene(), np.ones((4, 29), dtype=np.float32))
        assert seen and np.allclose(seen[0], 1.0)


def test_validate_rejects_wrong_action_dim() -> None:
    model = _model(8000)  # no server contacted — validation fails first
    try:
        model.rollout(_scene(), np.zeros((4, 7), dtype=np.float32))
    except ValueError as e:
        assert "action_dim" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched action_dim")


def test_missing_conditioning_camera_rejected() -> None:
    model = Cosmos3WorldModel(
        server_url="http://127.0.0.1:8000",
        domain_name="agibotworld", action_dim=29, conditioning_camera="wrist",
    )
    try:
        model.rollout(_scene(), np.zeros((4, 29), dtype=np.float32))
    except ValueError as e:
        assert "wrist" in str(e)
    else:
        raise AssertionError("expected ValueError for missing conditioning camera")


def test_worldmodel_handler_roundtrip() -> None:
    """The server-side WorldModelHandler encodes a rollout the wire codec
    reconstructs as a Trajectory — the new world-model wire surface."""
    from emboviz_wire import WorldModelHandler, wire

    with _MockCosmos() as mock:
        handler = WorldModelHandler(_model(mock.port))

        meta = handler.methods["static_metadata"]({})
        assert meta["model_id"] == "cosmos3-nano"
        assert meta["action_dim"] == 29
        assert meta["supported_domains"] == ["agibotworld"]
        assert meta["conditioning_camera"] == "primary"

        actions = np.zeros((12, 29), dtype=np.float32)
        encoded = handler.methods["rollout"]({
            "init": wire.encode_scene(_scene()),
            "actions": actions,
            "num_frames": None,
        })
        traj = wire.decode_trajectory(encoded)
        assert isinstance(traj, Trajectory)
        assert len(traj.frames) == 12
        assert traj.metadata["world_model"] == "cosmos3-nano"


def test_closed_loop_joint_dream_end_to_end() -> None:
    """The whole π0-DROID closed loop offline: real Franka forward kinematics +
    a mock joint-space policy + the mock Cosmos server. Exercises every link the
    real run uses except the GPU policy and the GPU world model — concat split,
    joint tracking, FK, droid_lerobot encoding, the autoregressive request, and
    Trajectory assembly.
    """
    from emboviz_robot import load_kinematics
    from emboviz_wire.types import ActionResult, Scene

    from emboviz.world_models.simulate import closed_loop_rollout
    from emboviz_cosmos3.bridge import JointStateTracker
    from emboviz_cosmos3.concat_view import build_concat_view
    from emboviz_cosmos3.dream_step import PolicyDreamStepper

    kin = load_kinematics("franka_panda")
    seed_joints = np.array([0.09, -0.28, 0.24, -2.08, -0.16, 1.57, -0.01], dtype=np.float32)

    seen = {}

    def mock_pi0(scene: Scene) -> ActionResult:
        seen["state_dim"] = int(np.asarray(scene.observations.state.values).size)
        seen["convention"] = scene.observations.state.convention
        seen["instruction"] = scene.instruction
        seen["cameras"] = sorted(scene.observations.images)
        chunk = np.zeros((10, 8), dtype=np.float32)   # [joint_delta(7), gripper(1)]
        chunk[:, 1] = 0.005                            # small +joint1 each step
        chunk[:, 7] = 0.4                              # absolute gripper
        return ActionResult(action=chunk[0], action_chunk=chunk)

    stepper = PolicyDreamStepper(
        mock_pi0,
        tracker=JointStateTracker(seed_joints, 0.0, kin),
        camera_map={"primary": "exterior_left", "wrist_left": "wrist"},
        instruction="pick the marker from the cup",
        n_actions=10,
    )

    seed_concat = build_concat_view(
        np.zeros((12, 16, 3), np.uint8),
        np.full((8, 10, 3), 60, np.uint8),
        np.full((8, 10, 3), 180, np.uint8),
    )

    with _MockCosmos() as mock:
        model = Cosmos3WorldModel(
            server_url=f"http://127.0.0.1:{mock.port}",
            domain_name="droid_lerobot", action_dim=10, action_chunk_size=16,
        )
        dream = closed_loop_rollout(model, seed_concat, stepper, n_steps=1)

    assert len(dream.trajectory.frames) == 10          # 10 actions -> 10 dream frames
    assert mock.requested_chunk_sizes == [10]          # one chunk of the full π0 horizon
    assert seen["state_dim"] == 7 and seen["convention"] == "joint_angles"
    assert seen["cameras"] == ["primary", "wrist_left"]
    assert seen["instruction"] == "pick the marker from the cup"
    # The tracked joint vector advanced by +0.005 * 10 on joint1.
    assert np.isclose(stepper.tracker.joints[1], seed_joints[1] + 0.05, atol=1e-4)


def _run_all() -> None:
    test_rollout_shapes_and_chunking()
    test_num_frames_truncates_actions()
    test_action_normalizer_applied()
    test_validate_rejects_wrong_action_dim()
    test_missing_conditioning_camera_rejected()
    test_worldmodel_handler_roundtrip()
    test_closed_loop_joint_dream_end_to_end()
    print("OK: all cosmos3 rollout-against-mock checks passed")


if __name__ == "__main__":
    _run_all()
