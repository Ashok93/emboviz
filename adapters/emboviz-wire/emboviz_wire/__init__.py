"""emboviz-wire — the minimal ZeroMQ connector + shared wire contracts.

This is the ONLY emboviz package a model worker venv installs. It
carries the things that cross the socket and the things a model
adapter implements, and **nothing else** — no datasets, no diagnostics,
no rerun/viz/export. Its dependency floor is deliberately tiny
(pyzmq + msgpack + msgpack-numpy + numpy + pydantic, all loosely
pinned) so it never fights a model's own pins.

Two processes (a model worker and the emboviz host) talk purely through
**bytes** (msgpack over a ZMQ Unix socket). Because the wire format is
standardized, the worker and host may run completely different
numpy / pyzmq / msgpack versions — only the bytes are
shared, and the bytes are version-stable. The single guarantee is that
both sides run the *same emboviz-wire* (same codec + same schemas).

Contents:
  • ``server`` / ``serve``       — the ZMQ ROUTER + Service-Handler dispatch
  • ``wire``                     — the msgpack encode/decode codec
  • ``handler`` / ``AdapterSpec``— the worker handler contract + adapter spec
  • ``types``                    — Scene / Observations / ActionResult / …
  • ``model_protocol``           — VLAModel / Capability / RequiredInputs
  • ``observations`` / ``profile`` / ``distances`` — the typed data model
"""

from __future__ import annotations

# ── ZMQ connector — server end (worker) + client end (host) ──────────
from emboviz_wire.server import (
    DatasetReaderHandler,
    ServiceHandler,
    VLAModelHandler,
    serve,
)
from emboviz_wire.client import (
    AdapterRpcError,
    RpcClient,
    ZMQAdapterClient,
    ZMQReaderClient,
    default_endpoint,
)
from emboviz_wire.handler import AdapterSpec
from emboviz_wire import wire

# ── shared wire data model ───────────────────────────────────────────
from emboviz_wire.types import (
    ActionResult,
    AttentionMaps,
    AttentionTrace,
    FFNActivations,
    HiddenStates,
    ImageLike,
    Observations,
    PerturbedScene,
    Scene,
    TokenSelector,
    Trajectory,
    resolve_cameras,
)
from emboviz_wire.model_protocol import (
    Capability,
    NotSupported,
    RequiredInputs,
    VLAModel,
)
from emboviz_wire.observations import (
    ActionHistory,
    DepthMap,
    ForceTorque,
    GripperState,
    Proprioception,
    RGBImage,
    TactileReading,
)
from emboviz_wire.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)
from emboviz_wire.reader_protocol import EpisodeSource
from emboviz_wire.dataset_build import (
    build_profile,
    make_gripper_extractor,
    parse_lerobot_names,
)

__all__ = [
    # connector
    "serve",
    "ServiceHandler",
    "VLAModelHandler",
    "DatasetReaderHandler",
    "RpcClient",
    "ZMQAdapterClient",
    "ZMQReaderClient",
    "AdapterRpcError",
    "default_endpoint",
    "AdapterSpec",
    "wire",
    # dataset-reader contract + shared construction helpers
    "EpisodeSource",
    "build_profile",
    "make_gripper_extractor",
    "parse_lerobot_names",
    # types
    "ActionResult",
    "AttentionMaps",
    "AttentionTrace",
    "FFNActivations",
    "HiddenStates",
    "ImageLike",
    "Observations",
    "PerturbedScene",
    "Scene",
    "TokenSelector",
    "Trajectory",
    "resolve_cameras",
    # model protocol
    "Capability",
    "NotSupported",
    "RequiredInputs",
    "VLAModel",
    # observations
    "ActionHistory",
    "DepthMap",
    "ForceTorque",
    "GripperState",
    "Proprioception",
    "RGBImage",
    "TactileReading",
    # profile
    "ActionSpec",
    "CameraSpec",
    "GripperSpec",
    "RobotProfile",
    "StateSpec",
]
