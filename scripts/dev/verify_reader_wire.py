"""Verify the dataset-reader wire path end-to-end WITHOUT lerobot.

Stands up a DatasetReaderHandler wrapping a mock EpisodeSource on a real
ZMQ socket, connects a ZMQReaderClient, and exercises every method —
proving the handler↔client↔codec path (incl. profile + Trajectory
round-trip) works exactly as the lerobot worker will use it. Also checks
core imports and that the `emboviz.readers` entry point resolves.

Run: uv run --no-sync python scripts/dev/verify_reader_wire.py
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

# ── 1. core + wire imports across the refactor ───────────────────────
import emboviz.datasets.manifest          # noqa: F401
import emboviz.datasets.base              # noqa: F401
from emboviz.adapters import (
    connect_reader, find_reader, list_readers,
    DatasetReaderHandler, ZMQReaderClient,
)
from emboviz_wire.server import _serve_async
from emboviz_wire.reader_protocol import EpisodeSource
from emboviz_wire.types import Scene, Observations, Trajectory
from emboviz_wire.observations import RGBImage, Proprioception, GripperState
from emboviz_wire.profile import RobotProfile, ActionSpec, StateSpec, GripperSpec
print("[1] core+wire imports OK")

# ── 2. entry-point discovery of the lerobot reader ───────────────────
readers = list_readers()
assert "lerobot" in readers, f"lerobot reader not discovered: {sorted(readers)}"
spec = find_reader("lerobot")
assert spec.server_module == "emboviz_lerobot.server"
assert spec.needs_gpu is False and spec.requires_python == "3.11"
print(f"[2] entry-point discovery OK: find_reader('lerobot') → {spec.name}, "
      f"runtime_pip={spec.runtime_pip}")


# ── 3. a mock EpisodeSource (no lerobot) exercising the default
#       load_episodes / load_trajectory built on load_episode ─────────
class MockReader(EpisodeSource):
    name = "lerobot:mock-dataset"

    def list_episodes(self):
        return [str(i) for i in range(5)]

    def load_episode(self, episode_id):
        i = int(episode_id)
        img = np.full((8, 8, 3), i % 256, np.uint8)
        prof = RobotProfile(
            name="mockbot",
            state=StateSpec(dim=7, convention="ee_pose"),
            gripper=GripperSpec(kind="parallel_jaw", units="unit", range=(0.0, 1.0)),
            action=ActionSpec(dim=7, dim_names=["dx", "dy", "dz", "rx", "ry", "rz", "g"]),
        )
        obs = Observations(
            images={"primary": RGBImage(data=img, camera_id="primary")},
            state=Proprioception(values=np.arange(7, dtype=np.float32), convention="ee_pose"),
            gripper=GripperState(value=float(i) / 10, kind="parallel_jaw", units="unit"),
        )
        return [Scene(observations=obs, instruction=f"episode {i} task",
                      profile=prof, metadata={"fps": 5.0, "episode_index": i},
                      scene_id=f"lerobot:mock:{i}:0")]

    def all_instructions(self):
        return ["pick the cube", "open drawer", "wipe table"]


SOCK = "ipc:///tmp/emboviz_verify_reader.sock"


def _serve_in_thread():
    handler = DatasetReaderHandler(MockReader())
    asyncio.run(_serve_async(handler, SOCK, n_workers=1))


t = threading.Thread(target=_serve_in_thread, daemon=True)
t.start()

client = ZMQReaderClient("lerobot", endpoint=SOCK)
# wait for the worker to bind
for _ in range(50):
    if client.ping(timeout_ms=300):
        break
    time.sleep(0.1)
else:
    raise SystemExit("mock reader worker never became ready")
print("[3] mock reader worker up; client connected")

# ── 4. exercise every EpisodeSource method over the wire ─────────────
assert client.name == "lerobot:mock-dataset", client.name           # static_metadata
assert client.list_episodes() == [str(i) for i in range(5)]
assert client.all_instructions() == ["pick the cube", "open drawer", "wipe table"]

traj = client.load_trajectory(2)
assert isinstance(traj, Trajectory) and len(traj.frames) == 1
fr = traj.frames[0]
assert fr.instruction == "episode 2 task"
assert fr.profile is not None and fr.profile.action.dim_names == ["dx","dy","dz","rx","ry","rz","g"]
assert fr.profile.state.convention == "ee_pose"
assert np.asarray(fr.observations.images["primary"].data).max() == 2     # img filled with i=2
assert fr.observations.state.values.tolist() == list(range(7))
assert abs(fr.observations.gripper.value - 0.2) < 1e-6

eps = client.load_episodes([1, 3])
assert set(eps) == {1, 3}
assert np.asarray(eps[1][0].observations.images["primary"].data).max() == 1
assert np.asarray(eps[3][0].observations.images["primary"].data).max() == 3
assert eps[3][0].profile.action.dim == 7
print("[4] all EpisodeSource methods round-trip over the wire: "
      "list_episodes, all_instructions, load_trajectory, load_episodes "
      "— Scenes carry profile + state + gripper intact")

client.shutdown()
client.close()
t.join(timeout=5)
print("\nREADER WIRE PATH VERIFIED (no lerobot needed)")
