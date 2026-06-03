"""ZeroMQ DEALER clients for adapter IPC.

Two layers:

* :class:`RpcClient` — the transport. Knows about ZMQ DEALER sockets,
  msgpack framing, request IDs, timeouts, and error frames. **Knows
  nothing** about what the methods mean.
* :class:`ZMQAdapterClient` — extends :class:`RpcClient` and adds the
  VLA-side encode / decode shim so the class satisfies the
  :class:`emboviz_wire.model_protocol.VLAModel` ABC.

Non-VLA adapter clients (SAM 3 and any future detector / perception
worker) inherit from :class:`RpcClient` directly and ship their own
typed method wrappers in the adapter package.

Wire protocol (matches :mod:`emboviz_wire.server`)::

    [req_id (8 bytes), msgpack({"method": str, "args": dict})]

Reply::

    [req_id (8 bytes), msgpack({"ok": bool, "result"?: ..., "error"?: str})]

``req_id`` increments per outbound call; the reader discards any reply
whose id doesn't match the one we're waiting for, which keeps the
socket clean across timeout retries.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zmq

from emboviz_wire import wire
from emboviz_wire.types import (
    ActionResult,
    AttentionMaps,
    FFNActivations,
    HiddenStates,
    Scene,
    TokenSelector,
)
from emboviz_wire.model_protocol import Capability, RequiredInputs, VLAModel
from emboviz_wire.reader_protocol import EpisodeSource
from emboviz_wire.world_model_protocol import WorldModel, WorldModelCapability
from emboviz_wire.types import Trajectory


# Default per-call timeout. Diagnostics that need to cold-load a large
# model use ``static_metadata`` first, which can take a while on the
# first call (the server has to finish initializing the model). Size
# accordingly; override with ``EMBOVIZ_RPC_TIMEOUT_MS``.
_DEFAULT_TIMEOUT_MS = int(os.environ.get("EMBOVIZ_RPC_TIMEOUT_MS", "600000"))


# ─────────────────────────────────────────────────────────────────────
# Endpoint resolution (mirror of server_base._resolve_endpoint)
# ─────────────────────────────────────────────────────────────────────


def default_endpoint(name: str) -> str:
    """Where the client looks for the server when no override is set.

    Reads ``EMBOVIZ_<NAME>_ENDPOINT`` first; otherwise defaults to the
    Unix-socket convention shared with :mod:`emboviz.adapters.lifecycle`
    and :mod:`emboviz.adapters.server_base`.
    """
    override = os.environ.get(f"EMBOVIZ_{name.upper()}_ENDPOINT")
    if override:
        return override
    if sys.platform.startswith("win"):
        return f"tcp://127.0.0.1:{8800 + (sum(map(ord, name)) % 100)}"
    sock_path = Path.home() / ".emboviz" / "sockets" / f"{name}.sock"
    return f"ipc://{sock_path}"


# ─────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────


class AdapterRpcError(RuntimeError):
    """Raised when the remote worker returns an explicit error frame.

    The message is the remote ``ClassName: message`` string; the
    optional ``remote_traceback`` attribute carries the worker-side
    stack trace for debugging.
    """

    def __init__(self, message: str, remote_traceback: Optional[str] = None):
        super().__init__(message)
        self.remote_traceback = remote_traceback


# ─────────────────────────────────────────────────────────────────────
# Transport — the only file that touches ZMQ sockets.
# ─────────────────────────────────────────────────────────────────────


class RpcClient:
    """ZMQ DEALER client over msgpack-framed wire.

    Subclasses add typed method wrappers; this base only knows how to
    send/receive opaque msgpack-encoded ``(method, args)`` pairs.

    Constructor connects to the running worker but does NOT block
    waiting for the model to load; the first :meth:`request` is the
    one that pays for cold-load latency.
    """

    def __init__(
        self,
        name: str,
        *,
        endpoint: Optional[str] = None,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        self._name = name
        self._endpoint = endpoint or default_endpoint(name)
        self._timeout_ms = int(timeout_ms)

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        # Random identity per client — lets the server's ROUTER socket
        # tell concurrent clients apart.
        self._sock.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
        # LINGER=0: closing the socket drops pending sends instead of
        # blocking. We never want a stuck client on process exit.
        self._sock.setsockopt(zmq.LINGER, 0)
        # RCVTIMEO is enforced by the poll loop in :meth:`request`; this
        # is a safety net for any direct recv we might add.
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.connect(self._endpoint)

        self._next_id = 0
        self._closed = False

    # ----- low-level RPC --------------------------------------------------

    def request(self, method: str, args: Optional[dict] = None) -> Any:
        """Send one request, return the deserialized result.

        Raises :class:`AdapterRpcError` if the server reports a method
        failure, or :class:`TimeoutError` if the reply doesn't arrive
        within ``self._timeout_ms``.
        """
        if self._closed:
            raise RuntimeError(f"adapter client '{self._name}' is closed")

        self._next_id = (self._next_id + 1) & 0xFFFFFFFFFFFFFFFF
        req_id = self._next_id.to_bytes(8, "big")
        body = wire.pack({"method": method, "args": args or {}})
        self._sock.send_multipart([req_id, body])

        # Explicit poll so we honor a single overall deadline AND can
        # discard stray frames left behind from a previously timed-out
        # call (keeping the socket usable for the next request).
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        deadline_ms = self._timeout_ms
        while True:
            socks = dict(poller.poll(deadline_ms))
            if self._sock not in socks:
                raise TimeoutError(
                    f"adapter '{self._name}' method {method!r} timed out "
                    f"after {self._timeout_ms}ms (endpoint={self._endpoint})"
                )
            rep_id, rep_body = self._sock.recv_multipart()
            if rep_id != req_id:
                # Stale frame from a previously timed-out request — drop
                # and keep polling for the one we actually want.
                continue
            reply = wire.unpack(rep_body)
            if not reply.get("ok"):
                raise AdapterRpcError(
                    reply.get("error") or "remote error",
                    remote_traceback=reply.get("traceback"),
                )
            return reply.get("result")

    # ----- protocol-level helpers (handled by the server loop) -----------

    def ping(self, timeout_ms: int = 5000) -> bool:
        """Cheap synchronous reachability check used by lifecycle's
        readiness probe. Returns True iff the worker replied within
        ``timeout_ms``."""
        old = self._timeout_ms
        self._timeout_ms = int(timeout_ms)
        try:
            self.request("ping")
            return True
        except (TimeoutError, zmq.ZMQError, AdapterRpcError):
            return False
        finally:
            self._timeout_ms = old

    def shutdown(self) -> None:
        """Ask the worker to exit cleanly. Returns once the worker
        acknowledges the request (which is BEFORE it has actually
        torn down — the worker keeps in-flight handlers running)."""
        try:
            self.request("shutdown")
        except (TimeoutError, AdapterRpcError):
            pass

    # ----- lifecycle -----------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
        self._closed = True

    def __enter__(self) -> "RpcClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────
# VLA adapter client — typed wrappers around :meth:`RpcClient.request`.
# ─────────────────────────────────────────────────────────────────────


class ZMQAdapterClient(RpcClient, VLAModel):
    """VLAModel facade over an :class:`RpcClient`.

    Every VLAModel protocol method is implemented as one
    :meth:`request` call with the matching wire encode / decode. The
    server side is :class:`emboviz_wire.server.VLAModelHandler`
    — the two files must be kept on the SAME method list.
    """

    def __init__(
        self,
        name: str,
        *,
        endpoint: Optional[str] = None,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        super().__init__(name, endpoint=endpoint, timeout_ms=timeout_ms)
        # Static metadata is fetched lazily on first property access.
        self._meta: Optional[dict] = None

    def _ensure_meta(self) -> dict:
        if self._meta is None:
            self._meta = self.request("static_metadata")
        return self._meta

    # ----- VLAModel ABC: identification ---------------------------------

    @property
    def model_id(self) -> str:
        return self._ensure_meta()["model_id"]

    @property
    def capabilities(self) -> Capability:
        return Capability(self._ensure_meta()["capabilities"])

    @property
    def required_inputs(self) -> RequiredInputs:
        return wire.decode_required_inputs(self._ensure_meta()["required_inputs"])

    @property
    def action_dim(self) -> int:
        return int(self._ensure_meta()["action_dim"])

    @property
    def action_scale(self) -> Optional[np.ndarray]:
        s = self._ensure_meta().get("action_scale")
        return None if s is None else np.asarray(s)

    @property
    def num_layers(self) -> Optional[int]:
        return self._ensure_meta().get("num_layers")

    @property
    def num_heads(self) -> Optional[int]:
        return self._ensure_meta().get("num_heads")

    @property
    def hidden_dim(self) -> Optional[int]:
        return self._ensure_meta().get("hidden_dim")

    # ----- VLAModel ABC: inference --------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        result = self.request("predict", {"scene": wire.encode_scene(scene)})
        return wire.decode_action_result(result)

    def predict_batch(
        self, scenes: list[Scene], n_samples: int = 1,
    ) -> list[ActionResult]:
        result = self.request("predict_batch", {
            "scenes": [wire.encode_scene(s) for s in scenes],
            "n_samples": int(n_samples),
        })
        return [wire.decode_action_result(r) for r in result]

    # ----- VLAModel ABC: internal inspection ----------------------------

    def extract_attention(self, scene: Scene, query: TokenSelector) -> AttentionMaps:
        result = self.request("extract_attention", {
            "scene": wire.encode_scene(scene),
            "query": wire.encode_token_selector(query),
        })
        return wire.decode_attention_maps(result)

    def extract_hidden_states(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> HiddenStates:
        result = self.request("extract_hidden_states", {
            "scene": wire.encode_scene(scene),
            "query": wire.encode_token_selector(query),
            "layer_indices": list(layer_indices),
        })
        return wire.decode_hidden_states(result)

    def extract_ffn_activations(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> FFNActivations:
        result = self.request("extract_ffn_activations", {
            "scene": wire.encode_scene(scene),
            "query": wire.encode_token_selector(query),
            "layer_indices": list(layer_indices),
        })
        return wire.decode_ffn_activations(result)

    # ----- VLAModel ABC: vocab projection -------------------------------

    def get_ffn_value_vector_norms(self, layer_indices: list[int]) -> dict[int, np.ndarray]:
        result = self.request("get_ffn_value_vector_norms", {
            "layer_indices": list(layer_indices),
        })
        return {int(k): np.asarray(v) for k, v in result.items()}

    def project_to_vocab(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        result = self.request("project_to_vocab", {
            "vector": np.asarray(vector),
            "top_k": int(top_k),
        })
        return [(str(t), float(s)) for t, s in result]

    # ----- VLAModel ABC: interventions ----------------------------------

    def predict_with_neuron_ablation(
        self, scene: Scene, ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        # Tuples don't round-trip through msgpack — send pairs as lists.
        result = self.request("predict_with_neuron_ablation", {
            "scene": wire.encode_scene(scene),
            "ablations": [(list(k), float(v)) for k, v in ablations.items()],
        })
        return wire.decode_action_result(result)

    def predict_with_residual_patch(
        self, scene: Scene, patches: dict[int, np.ndarray],
        patch_position: Optional[int] = None,
    ) -> ActionResult:
        result = self.request("predict_with_residual_patch", {
            "scene": wire.encode_scene(scene),
            "patches": {int(k): np.asarray(v) for k, v in patches.items()},
            "patch_position": patch_position,
        })
        return wire.decode_action_result(result)

    # ----- VLAModel ABC: tokenization -----------------------------------

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return list(self.request("find_token_positions", {
            "instruction": instruction, "word": word,
        }))


# ─────────────────────────────────────────────────────────────────────
# Dataset-reader client — typed wrappers, satisfies the EpisodeSource ABC.
# ─────────────────────────────────────────────────────────────────────


class ZMQReaderClient(RpcClient, EpisodeSource):
    """EpisodeSource facade over an :class:`RpcClient`.

    The dataset-side analogue of :class:`ZMQAdapterClient`. The host
    talks to an isolated reader worker (its own venv with the heavy
    dataset library) exactly as it talks to a model worker — same wire,
    same lifecycle — but the methods are the EpisodeSource contract. The
    server side is :class:`emboviz_wire.server.DatasetReaderHandler`; the
    two must be kept on the SAME method list.
    """

    def __init__(
        self,
        name: str,
        *,
        endpoint: Optional[str] = None,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        super().__init__(name, endpoint=endpoint, timeout_ms=timeout_ms)
        self._meta: Optional[dict] = None

    def _ensure_meta(self) -> dict:
        if self._meta is None:
            self._meta = self.request("static_metadata")
        return self._meta

    @property
    def name(self) -> str:            # type: ignore[override]
        """Dataset identity (e.g. ``"lerobot:<repo_id>"``), from the
        worker — NOT the socket name. The modality-pool cache keys off
        this, so it must reflect the actual dataset."""
        return self._ensure_meta().get("name") or self._name

    def list_episodes(self) -> list[str]:
        return [str(e) for e in self.request("list_episodes")]

    def load_episode(self, episode_id: str) -> list[Scene]:
        result = self.request("load_episode", {"episode_id": str(episode_id)})
        return [wire.decode_scene(s) for s in result]

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        result = self.request("load_episodes", {
            "episode_indices": [int(i) for i in episode_indices],
        })
        return {int(k): [wire.decode_scene(s) for s in v] for k, v in result.items()}

    def episode_lengths(self, episode_indices: list[int]) -> dict[int, int]:
        result = self.request("episode_lengths", {
            "episode_indices": [int(i) for i in episode_indices],
        })
        return {int(k): int(v) for k, v in result.items()}

    def sample_frames(self, episode_offsets: dict[int, int]) -> dict[int, Scene]:
        result = self.request("sample_frames", {
            "episode_offsets": {int(k): int(v) for k, v in episode_offsets.items()},
        })
        return {int(k): wire.decode_scene(v) for k, v in result.items()}

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        result = self.request("load_trajectory", {"episode_idx": int(episode_idx)})
        return wire.decode_trajectory(result)

    def all_instructions(self) -> list[str]:
        return [str(x) for x in self.request("all_instructions")]


# ─────────────────────────────────────────────────────────────────────
# World-model client — typed wrappers, satisfies the WorldModel ABC.
# ─────────────────────────────────────────────────────────────────────


class ZMQWorldModelClient(RpcClient, WorldModel):
    """WorldModel facade over an :class:`RpcClient`.

    The third sibling of :class:`ZMQAdapterClient` (policy) and
    :class:`ZMQReaderClient` (dataset). The host talks to an isolated
    world-model worker — its own venv with the heavy generative stack —
    exactly as it talks to the other two: same wire, same lifecycle. The
    server side is :class:`emboviz_wire.server.WorldModelHandler`; the two
    must be kept on the SAME method list.
    """

    def __init__(
        self,
        name: str,
        *,
        endpoint: Optional[str] = None,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        super().__init__(name, endpoint=endpoint, timeout_ms=timeout_ms)
        self._meta: Optional[dict] = None

    def _ensure_meta(self) -> dict:
        if self._meta is None:
            self._meta = self.request("static_metadata")
        return self._meta

    # ----- WorldModel ABC: identification -------------------------------

    @property
    def model_id(self) -> str:
        return self._ensure_meta()["model_id"]

    @property
    def capabilities(self) -> WorldModelCapability:
        return WorldModelCapability(self._ensure_meta()["capabilities"])

    @property
    def action_dim(self) -> int:
        return int(self._ensure_meta()["action_dim"])

    @property
    def supported_domains(self) -> frozenset[str]:
        return frozenset(self._ensure_meta()["supported_domains"])

    @property
    def conditioning_camera(self) -> str:
        return self._ensure_meta().get("conditioning_camera") or "primary"

    # ----- WorldModel ABC: prediction -----------------------------------

    def rollout(
        self,
        init: Scene,
        actions: np.ndarray,
        *,
        num_frames: Optional[int] = None,
    ) -> Trajectory:
        result = self.request("rollout", {
            "init": wire.encode_scene(init),
            "actions": np.asarray(actions),
            "num_frames": None if num_frames is None else int(num_frames),
        })
        return wire.decode_trajectory(result)

    def actions_from_video(self, frames: Trajectory) -> np.ndarray:
        result = self.request("actions_from_video", {
            "frames": wire.encode_trajectory(frames),
        })
        return np.asarray(result)
