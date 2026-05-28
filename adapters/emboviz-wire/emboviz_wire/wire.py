"""Wire-format encoders / decoders for adapter IPC.

All the typed values that flow between the core process and an
adapter worker (``Scene``, ``ActionResult``, ``AttentionMaps``,
``HiddenStates``, ``FFNActivations``, ``TokenSelector``) get converted
to/from msgpack-friendly Python dicts here. The on-wire payload is
``msgpack.packb(<dict>)`` with ``msgpack-numpy``'s codec installed so
numpy arrays travel as native typed buffers (no JSON / base64 bloat).

Why this is small and stable:

* msgpack handles primitives, lists, dicts, bytes natively.
* ``msgpack-numpy`` handles ``np.ndarray`` natively (zero-copy on
  decode where the buffer protocol allows).
* The Pillow ``Image`` we carry inside an ``RGBImage`` is converted to
  a uint8 ``np.ndarray`` for transport (PIL is on both ends but its
  internal pixel buffer isn't a stable wire shape — numpy is).
* Frozen-dataclass-with-numpy-fields is the only nontrivial shape and
  every one of them is rebuilt by name from the decoded dict so we
  don't depend on cross-Python-version pickle of dataclass objects.

This file is the SCHEMA — keep both core and every adapter package on
the same version of it. Adding a new typed observation modality means
adding a small ``_enc_<modality>`` / ``_dec_<modality>`` pair here.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# msgpack-numpy patches msgpack so np.ndarray round-trips natively.
# Calling .patch() at import time is the standard upstream pattern;
# it only affects packb/unpackb in this process (idempotent).
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


# Tag bytes used as the leading byte of every framed payload.  The
# receiver can use these as a quick "is this what I expected?" check
# before unpacking. We do not currently rely on them; they document
# the protocol.
TAG_REQUEST = 0x01
TAG_REPLY_OK = 0x02
TAG_REPLY_ERR = 0x03


# ─────────────────────────────────────────────────────────────────────
# Helpers: dataclass / PIL <-> dict
# ─────────────────────────────────────────────────────────────────────


def _img_to_ndarray(img: Any) -> np.ndarray:
    """Materialize a PIL Image (or already-an-ndarray) as uint8 HWC."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        # Tolerate float [0,1]; cast at boundary. We never assume.
        if arr.dtype.kind == "f" and arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def _ndarray_to_pil(arr: np.ndarray):
    """Rebuild a PIL Image from a uint8 HWC ndarray."""
    from PIL import Image

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────
# Observation modalities — each is a tiny frozen dataclass + ndarray.
# ─────────────────────────────────────────────────────────────────────


def _enc_rgb_image(im) -> dict:
    return {"data": _img_to_ndarray(im.data), "camera_id": im.camera_id}


def _dec_rgb_image(d: dict):
    from emboviz_wire.observations import RGBImage
    return RGBImage(data=_ndarray_to_pil(d["data"]), camera_id=d.get("camera_id", "primary"))


def _enc_depth(dm) -> dict:
    return {"data": np.asarray(dm.data), "camera_id": dm.camera_id, "units": dm.units}


def _dec_depth(d: dict):
    from emboviz_wire.observations import DepthMap
    return DepthMap(data=d["data"], camera_id=d.get("camera_id", "primary"),
                    units=d.get("units", "meters"))


def _enc_state(st) -> Optional[dict]:
    if st is None:
        return None
    return {"values": np.asarray(st.values), "convention": st.convention}


def _dec_state(d: Optional[dict]):
    if d is None:
        return None
    from emboviz_wire.observations import Proprioception
    return Proprioception(values=d["values"], convention=d["convention"])


def _enc_gripper(g) -> Optional[dict]:
    if g is None:
        return None
    return {
        "value": float(g.value),
        "kind": g.kind,
        "units": g.units,
        "joint_angles": (np.asarray(g.joint_angles) if g.joint_angles is not None else None),
    }


def _dec_gripper(d: Optional[dict]):
    if d is None:
        return None
    from emboviz_wire.observations import GripperState
    return GripperState(value=d["value"], kind=d.get("kind", "parallel_jaw"),
                        units=d.get("units", "unit"), joint_angles=d.get("joint_angles"))


def _enc_action_history(ah) -> Optional[dict]:
    if ah is None:
        return None
    return {
        "actions": np.asarray(ah.actions),
        "source": ah.source,
        "timesteps_back": int(ah.timesteps_back),
    }


def _dec_action_history(d: Optional[dict]):
    if d is None:
        return None
    from emboviz_wire.observations import ActionHistory
    return ActionHistory(actions=d["actions"], source=d["source"],
                         timesteps_back=d.get("timesteps_back", d["actions"].shape[0]))


def _enc_force(ft) -> Optional[dict]:
    if ft is None:
        return None
    return {"wrench": np.asarray(ft.wrench), "frame": ft.frame, "units": ft.units}


def _dec_force(d: Optional[dict]):
    if d is None:
        return None
    from emboviz_wire.observations import ForceTorque
    return ForceTorque(wrench=d["wrench"], frame=d.get("frame", "ee"),
                       units=d.get("units", "N_Nm"))


def _enc_tactile(tac) -> Optional[dict]:
    if tac is None:
        return None
    return {"data": np.asarray(tac.data), "sensor_id": tac.sensor_id}


def _dec_tactile(d: Optional[dict]):
    if d is None:
        return None
    from emboviz_wire.observations import TactileReading
    return TactileReading(data=d["data"], sensor_id=d.get("sensor_id", "primary"))


# ─────────────────────────────────────────────────────────────────────
# Scene  ↔  dict
# ─────────────────────────────────────────────────────────────────────


def encode_scene(scene) -> dict:
    obs = scene.observations
    return {
        "images": {cam: _enc_rgb_image(im) for cam, im in obs.images.items()},
        "state": _enc_state(obs.state),
        "gripper": _enc_gripper(obs.gripper),
        "action_history": _enc_action_history(obs.action_history),
        "depth": ({c: _enc_depth(dm) for c, dm in obs.depth.items()} if obs.depth else None),
        "force_torque": _enc_force(obs.force_torque),
        "tactile": _enc_tactile(obs.tactile),
        "extras": dict(obs.extras) if obs.extras else {},
        "instruction": scene.instruction,
        "scene_id": scene.scene_id,
        "metadata": dict(scene.metadata) if scene.metadata else {},
    }


def decode_scene(d: dict):
    """Rebuild a Scene from a wire dict. ``profile`` is not transported
    (it's a per-adapter concept that lives on the worker side); if a
    downstream consumer needs it, it should be reconstructed from the
    metadata blob."""
    from emboviz_wire.types import Observations, Scene

    obs = Observations(
        images={cam: _dec_rgb_image(im) for cam, im in d["images"].items()},
        state=_dec_state(d.get("state")),
        gripper=_dec_gripper(d.get("gripper")),
        action_history=_dec_action_history(d.get("action_history")),
        depth=({c: _dec_depth(dm) for c, dm in d["depth"].items()} if d.get("depth") else None),
        force_torque=_dec_force(d.get("force_torque")),
        tactile=_dec_tactile(d.get("tactile")),
        extras=d.get("extras") or {},
    )
    return Scene(
        observations=obs,
        instruction=d.get("instruction"),
        metadata=d.get("metadata") or {},
        scene_id=d.get("scene_id", ""),
    )


# ─────────────────────────────────────────────────────────────────────
# ActionResult  ↔  dict
# ─────────────────────────────────────────────────────────────────────


def encode_action_result(ar) -> dict:
    return {
        "action": np.asarray(ar.action),
        "action_dim": int(ar.action_dim or 0),
        "action_tokens": ar.action_tokens,
        "action_distribution": ar.action_distribution,
        "confidence": (float(ar.confidence) if ar.confidence is not None else None),
        "action_chunk": (np.asarray(ar.action_chunk) if ar.action_chunk is not None else None),
        "metadata": dict(ar.metadata) if ar.metadata else {},
    }


def decode_action_result(d: dict):
    from emboviz_wire.types import ActionResult
    return ActionResult(
        action=d["action"],
        action_dim=d.get("action_dim", int(np.asarray(d["action"]).shape[-1])),
        action_tokens=d.get("action_tokens"),
        action_distribution=d.get("action_distribution"),
        confidence=d.get("confidence"),
        action_chunk=d.get("action_chunk"),
        metadata=d.get("metadata") or {},
    )


# ─────────────────────────────────────────────────────────────────────
# TokenSelector  ↔  dict
# ─────────────────────────────────────────────────────────────────────


def encode_token_selector(ts) -> dict:
    return {"position": ts.position, "relative": ts.relative, "word": ts.word}


def decode_token_selector(d: dict):
    from emboviz_wire.types import TokenSelector
    return TokenSelector(position=d.get("position"), relative=d.get("relative"), word=d.get("word"))


# ─────────────────────────────────────────────────────────────────────
# AttentionMaps  ↔  dict
# ─────────────────────────────────────────────────────────────────────


def encode_attention_maps(am) -> dict:
    return {
        "weights": np.asarray(am.weights),
        "query_position": int(am.query_position),
        "n_keys": int(am.n_keys),
        # tuple keys → list-of-list for wire neutrality
        "image_token_ranges": {cam: [list(r) for r in ranges]
                               for cam, ranges in am.image_token_ranges.items()},
        "image_grid_sides": {cam: int(side) for cam, side in am.image_grid_sides.items()},
        "layer_indices": list(am.layer_indices) if am.layer_indices is not None else None,
        "metadata": dict(am.metadata) if am.metadata else {},
    }


def decode_attention_maps(d: dict):
    from emboviz_wire.types import AttentionMaps
    return AttentionMaps(
        weights=d["weights"],
        query_position=int(d["query_position"]),
        n_keys=int(d["n_keys"]),
        image_token_ranges={cam: [tuple(r) for r in ranges]
                            for cam, ranges in d["image_token_ranges"].items()},
        image_grid_sides={cam: int(side) for cam, side in d["image_grid_sides"].items()},
        layer_indices=d.get("layer_indices"),
        metadata=d.get("metadata") or {},
    )


# ─────────────────────────────────────────────────────────────────────
# HiddenStates / FFNActivations  ↔  dict
# ─────────────────────────────────────────────────────────────────────


def encode_hidden_states(hs) -> dict:
    return {
        "states": np.asarray(hs.states),
        "query_position": int(hs.query_position),
        "layer_indices": list(hs.layer_indices),
        "hidden_dim": int(hs.hidden_dim),
        "metadata": dict(hs.metadata) if hs.metadata else {},
    }


def decode_hidden_states(d: dict):
    from emboviz_wire.types import HiddenStates
    return HiddenStates(
        states=d["states"],
        query_position=int(d["query_position"]),
        layer_indices=list(d["layer_indices"]),
        hidden_dim=int(d.get("hidden_dim", d["states"].shape[-1])),
        metadata=d.get("metadata") or {},
    )


def encode_ffn_activations(fa) -> dict:
    # by_layer keys are ints — msgpack supports them natively.
    return {
        "by_layer": {int(k): np.asarray(v) for k, v in fa.by_layer.items()},
        "query_position": int(fa.query_position),
        "metadata": dict(fa.metadata) if fa.metadata else {},
    }


def decode_ffn_activations(d: dict):
    from emboviz_wire.types import FFNActivations
    return FFNActivations(
        by_layer={int(k): v for k, v in d["by_layer"].items()},
        query_position=int(d["query_position"]),
        metadata=d.get("metadata") or {},
    )


# ─────────────────────────────────────────────────────────────────────
# RequiredInputs  ↔  dict
#
# RequiredInputs is a frozen dataclass with frozensets — we transport
# as lists; the decoder rebuilds frozensets so downstream consumers see
# the original immutable shape (and ``.validate()`` works).
# ─────────────────────────────────────────────────────────────────────


def encode_required_inputs(ri) -> dict:
    return {
        "cameras": sorted(ri.cameras),
        "instruction": bool(ri.instruction),
        "state": bool(ri.state),
        "gripper": bool(ri.gripper),
        "action_history": bool(ri.action_history),
        "depth": bool(ri.depth),
        "force_torque": bool(ri.force_torque),
        "tactile": bool(ri.tactile),
        "extras": sorted(ri.extras),
    }


def decode_required_inputs(d: dict):
    from emboviz_wire.model_protocol import RequiredInputs
    return RequiredInputs(
        cameras=frozenset(d.get("cameras") or []),
        instruction=d.get("instruction", True),
        state=d.get("state", False),
        gripper=d.get("gripper", False),
        action_history=d.get("action_history", False),
        depth=d.get("depth", False),
        force_torque=d.get("force_torque", False),
        tactile=d.get("tactile", False),
        extras=frozenset(d.get("extras") or []),
    )


# ─────────────────────────────────────────────────────────────────────
# Pack / unpack wrappers — single import surface for client + server.
# ─────────────────────────────────────────────────────────────────────


def pack(obj: Any) -> bytes:
    """Serialize a Python value (already msgpack-friendly) to bytes."""
    return msgpack.packb(obj, use_bin_type=True)


def unpack(buf: bytes) -> Any:
    """Deserialize msgpack bytes back to a Python value.

    ``strict_map_key=False`` is required because we use ``int`` keys
    for layer-indexed dicts (``FFNActivations.by_layer``, residual
    patches keyed by layer, etc.).
    """
    return msgpack.unpackb(buf, raw=False, strict_map_key=False)
