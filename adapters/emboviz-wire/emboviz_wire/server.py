"""ZeroMQ ROUTER dispatch loop adapters use to expose a service.

Architecture: **Service Handler pattern.** Each adapter ships a small
"Handler" class whose ``methods`` property returns the explicit
``{wire-method-name → handler-function}`` dispatch table for that
adapter. The serve() loop looks up each incoming method against that
table and dispatches it; unknown names raise a clean error. There is
**no** ``getattr`` magic, **no** introspection-based discovery, **no**
fallback path — what's exposed over the wire is exactly what the
handler enumerated.

This is the same shape every serious RPC framework uses:

* gRPC service stubs (each ``rpc`` in ``.proto`` → one method)
* xmlrpc.server's ``register_function``
* JSON-RPC libraries (Jupyter, LSP) with explicit method registration
* Pyro5's ``@expose`` decorator (non-decorated methods aren't callable)
* Ray Serve / BentoML / FastAPI decorators

For VLA adapters we ship a ready-made :class:`VLAModelHandler` that
enumerates every :class:`emboviz_wire.model_protocol.VLAModel` method
once. Most adapters just write ``serve(my_factory, name="x")`` and
inherit that dispatch table. Non-VLA adapters (SAM 3, future
detectors) ship their own small Handler.

The protocol-level methods ``ping`` and ``shutdown`` are handled by
the loop itself — they're transport concerns, not model concerns, and
that separation prevents an adapter author from accidentally
shadowing them.

Wire protocol: ``[client_identity, req_id, msgpack(request)]`` frames
over a DEALER↔ROUTER pair. ``request`` = ``{"method": str, "args":
dict}``; the reply body is ``{"ok": True, "result": ...}`` or
``{"ok": False, "error": str, "traceback": str}``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import click
import numpy as np

import zmq
import zmq.asyncio

from emboviz_wire import wire


# ─────────────────────────────────────────────────────────────────────
# Protocol the serve() loop expects from any Handler.
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class ServiceHandler(Protocol):
    """A handler exposes an explicit method-dispatch table.

    Implementations live in adapter packages and ship the encode/decode
    glue for each method they expose. The serve() loop knows nothing
    about the underlying model.
    """

    @property
    def methods(self) -> dict[str, Callable[[dict], Any]]:
        """``{wire-method-name → handler(args_dict) -> msgpack-friendly result}``."""
        ...

    def close(self) -> None:                                 # pragma: no cover
        """Optional teardown invoked on graceful server shutdown."""
        ...


# ─────────────────────────────────────────────────────────────────────
# Standard VLA-model handler — every VLA adapter uses this.
# ─────────────────────────────────────────────────────────────────────


class VLAModelHandler:
    """Wire-method dispatcher for any :class:`emboviz_wire.model_protocol.VLAModel`.

    This is the boundary between msgpack bytes and the typed Scene /
    ActionResult / AttentionMaps / HiddenStates objects. Every
    VLAModel protocol method has exactly one entry in :attr:`methods`;
    each entry handles its own decode-args / call-model / encode-result
    sequence.

    Adapter packages that wrap a VLAModel just write::

        serve(lambda **kw: VLAModelHandler(OpenVLAAdapter(**kw)),
              name="openvla")

    and the rest of the dispatch table is provided here.
    """

    def __init__(self, model):
        self._m = model

    # ----- the dispatch table ----------------------------------------

    @property
    def methods(self) -> dict[str, Callable[[dict], Any]]:
        return {
            "static_metadata":               self._static_metadata,
            "predict":                       self._predict,
            "predict_batch":                 self._predict_batch,
            "extract_attention":             self._extract_attention,
            "extract_hidden_states":         self._extract_hidden_states,
            "extract_ffn_activations":       self._extract_ffn_activations,
            "find_token_positions":          self._find_token_positions,
            "get_ffn_value_vector_norms":    self._get_ffn_value_vector_norms,
            "project_to_vocab":              self._project_to_vocab,
            "predict_with_neuron_ablation":  self._predict_with_neuron_ablation,
            "predict_with_residual_patch":   self._predict_with_residual_patch,
        }

    # ----- handlers ---------------------------------------------------

    def _static_metadata(self, _: dict) -> dict:
        m = self._m
        action_scale = m.action_scale
        return {
            "model_id":        str(m.model_id),
            "capabilities":    int(m.capabilities.value),
            "required_inputs": wire.encode_required_inputs(m.required_inputs),
            "action_dim":      int(m.action_dim),
            "action_scale":    (np.asarray(action_scale) if action_scale is not None else None),
            "num_layers":      m.num_layers,
            "num_heads":       m.num_heads,
            "hidden_dim":      m.hidden_dim,
        }

    def _predict(self, args: dict) -> dict:
        scene = wire.decode_scene(args["scene"])
        return wire.encode_action_result(self._m.predict(scene))

    def _predict_batch(self, args: dict) -> list:
        scenes = [wire.decode_scene(s) for s in args["scenes"]]
        n_samples = int(args.get("n_samples", 1))
        results = self._m.predict_batch(scenes, n_samples)
        return [wire.encode_action_result(r) for r in results]

    def _extract_attention(self, args: dict) -> dict:
        scene = wire.decode_scene(args["scene"])
        query = wire.decode_token_selector(args["query"])
        return wire.encode_attention_maps(self._m.extract_attention(scene, query))

    def _extract_hidden_states(self, args: dict) -> dict:
        scene = wire.decode_scene(args["scene"])
        query = wire.decode_token_selector(args["query"])
        layer_indices = list(args.get("layer_indices") or [])
        return wire.encode_hidden_states(
            self._m.extract_hidden_states(scene, layer_indices, query)
        )

    def _extract_ffn_activations(self, args: dict) -> dict:
        scene = wire.decode_scene(args["scene"])
        query = wire.decode_token_selector(args["query"])
        layer_indices = list(args.get("layer_indices") or [])
        return wire.encode_ffn_activations(
            self._m.extract_ffn_activations(scene, layer_indices, query)
        )

    def _find_token_positions(self, args: dict) -> list[int]:
        return list(self._m.find_token_positions(args["instruction"], args["word"]))

    def _get_ffn_value_vector_norms(self, args: dict) -> dict[int, np.ndarray]:
        out = self._m.get_ffn_value_vector_norms(list(args.get("layer_indices") or []))
        return {int(k): np.asarray(v) for k, v in out.items()}

    def _project_to_vocab(self, args: dict) -> list:
        vector = np.asarray(args["vector"])
        top_k = int(args.get("top_k", 20))
        out = self._m.project_to_vocab(vector, top_k)
        return [(str(t), float(s)) for t, s in out]

    def _predict_with_neuron_ablation(self, args: dict) -> dict:
        scene = wire.decode_scene(args["scene"])
        # Tuple keys arrive as 2-element lists on the wire — rebuild.
        ablations = {tuple(k): float(v) for k, v in args["ablations"]}
        return wire.encode_action_result(
            self._m.predict_with_neuron_ablation(scene, ablations)
        )

    def _predict_with_residual_patch(self, args: dict) -> dict:
        scene = wire.decode_scene(args["scene"])
        patches = {int(k): np.asarray(v) for k, v in args["patches"].items()}
        patch_position = args.get("patch_position")
        return wire.encode_action_result(
            self._m.predict_with_residual_patch(scene, patches, patch_position)
        )

    # ----- optional teardown -----------------------------------------

    def close(self) -> None:
        close = getattr(self._m, "close", None)
        if callable(close):
            close()


# ─────────────────────────────────────────────────────────────────────
# Standard dataset-reader handler — every dataset reader uses this.
# ─────────────────────────────────────────────────────────────────────


class DatasetReaderHandler:
    """Wire-method dispatcher for any :class:`emboviz_wire.reader_protocol.
    EpisodeSource` — the dataset-side analogue of :class:`VLAModelHandler`.

    Lives in the isolated reader worker venv (which has the heavy dataset
    library, e.g. lerobot). It is the boundary between msgpack bytes and
    the typed Scene / Trajectory objects: it encodes whatever the reader
    produces and ships it to the host. Reader packages just write::

        serve(lambda **kw: DatasetReaderHandler(build_my_source(**kw)),
              name="lerobot")
    """

    def __init__(self, source):
        self._s = source

    @property
    def methods(self) -> dict[str, Callable[[dict], Any]]:
        return {
            "static_metadata":  self._static_metadata,
            "list_episodes":    self._list_episodes,
            "load_episode":     self._load_episode,
            "load_episodes":    self._load_episodes,
            "load_trajectory":  self._load_trajectory,
            "all_instructions": self._all_instructions,
        }

    def _static_metadata(self, _: dict) -> dict:
        return {"name": str(getattr(self._s, "name", "") or "")}

    def _list_episodes(self, _: dict) -> list:
        return [str(e) for e in self._s.list_episodes()]

    def _load_episode(self, args: dict) -> list:
        scenes = self._s.load_episode(str(args["episode_id"]))
        return [wire.encode_scene(s) for s in scenes]

    def _load_episodes(self, args: dict) -> dict:
        indices = [int(i) for i in args["episode_indices"]]
        out = self._s.load_episodes(indices)
        return {int(k): [wire.encode_scene(s) for s in v] for k, v in out.items()}

    def _load_trajectory(self, args: dict) -> dict:
        traj = self._s.load_trajectory(int(args["episode_idx"]))
        return wire.encode_trajectory(traj)

    def _all_instructions(self, _: dict) -> list:
        return [str(x) for x in self._s.all_instructions()]

    def close(self) -> None:
        close = getattr(self._s, "close", None)
        if callable(close):
            close()


# ─────────────────────────────────────────────────────────────────────
# Async ROUTER dispatch loop.
# ─────────────────────────────────────────────────────────────────────


async def _handle_request(
    handler: ServiceHandler,
    ident: bytes,
    req_id: bytes,
    body: bytes,
    sock: "zmq.asyncio.Socket",
    pool: ThreadPoolExecutor,
    shutdown_event: asyncio.Event,
) -> None:
    """Decode one inbound message, dispatch via the handler's method
    table, send the reply.

    Blocking handler invocations run in the thread pool so the event
    loop stays free to accept the next inbound request from a different
    caller. ``ping`` and ``shutdown`` are transport-level — they live
    in this function (not in the handler) so adapter authors cannot
    accidentally shadow them.

    Errors are reported as ``{"ok": False, "error": ..., "traceback":
    ...}`` inside the reply frame; the wire never silently drops a
    request.
    """
    try:
        req = wire.unpack(body)
        method = req.get("method")
        args = req.get("args") or {}
        if not isinstance(method, str):
            raise ValueError("request missing 'method' field")

        if method == "ping":
            result = {"ok": True}
        elif method == "shutdown":
            shutdown_event.set()
            result = {"shutting_down": True}
        else:
            table = handler.methods
            if method not in table:
                raise KeyError(
                    f"unknown wire method {method!r} (registered: "
                    f"{sorted(table) + ['ping', 'shutdown']})"
                )
            handler_fn = table[method]
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(pool, handler_fn, args)

        reply = wire.pack({"ok": True, "result": result})
    except Exception as e:
        tb = traceback.format_exc(limit=8)
        reply = wire.pack({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb,
        })

    await sock.send_multipart([ident, req_id, reply])


async def _serve_async(handler: ServiceHandler, sock_endpoint: str,
                       *, n_workers: int) -> None:
    """Bind the ROUTER socket and run the dispatch loop until SIGTERM,
    a ``shutdown`` request, or KeyboardInterrupt."""

    pool = ThreadPoolExecutor(
        max_workers=n_workers, thread_name_prefix="adapter-worker",
    )
    shutdown_event = asyncio.Event()

    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(sock_endpoint)
    print(f"[server] bound on {sock_endpoint}", flush=True)

    # SIGTERM / SIGINT flip the shutdown event so a clean ``kill`` from
    # systemd / docker / lifecycle releases the GPU before exit.
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown_event.set)
        except (NotImplementedError, RuntimeError):
            # Windows / restricted environments: fall through to the
            # default signal handling (KeyboardInterrupt path).
            pass

    inflight: set[asyncio.Task] = set()
    print(f"[server] ready (methods: {sorted(handler.methods) + ['ping', 'shutdown']})",
          flush=True)

    try:
        while not shutdown_event.is_set():
            # Race the next inbound message against the shutdown event
            # so we exit promptly. ``recv_multipart()`` already returns
            # an asyncio.Future via zmq.asyncio's I/O integration.
            recv_fut = sock.recv_multipart()
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            await asyncio.wait(
                [recv_fut, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown_event.is_set():
                recv_fut.cancel()
                break
            shutdown_task.cancel()
            ident, req_id, body = recv_fut.result()

            task = asyncio.create_task(
                _handle_request(handler, ident, req_id, body, sock,
                                pool, shutdown_event)
            )
            inflight.add(task)
            task.add_done_callback(inflight.discard)
    finally:
        # Let in-flight requests finish so we don't leave clients hanging.
        if inflight:
            await asyncio.wait(inflight, timeout=30)
        sock.close(linger=0)
        ctx.term()
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            close = getattr(handler, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        print("[server] shut down cleanly", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Endpoint resolution
# ─────────────────────────────────────────────────────────────────────


def _resolve_endpoint(name: str, sock: str | None, tcp: str | None) -> str:
    """Map CLI ``--sock`` / ``--tcp`` (or env-var defaults) to a ZMQ
    endpoint string (``ipc://...`` or ``tcp://host:port``)."""
    if tcp:
        return f"tcp://{tcp}"
    if sock:
        Path(sock).expanduser().parent.mkdir(parents=True, exist_ok=True)
        return f"ipc://{Path(sock).expanduser().absolute()}"

    # Default convention — matches what :mod:`lifecycle` looks for.
    if sys.platform.startswith("win"):
        # No Unix sockets on Windows — bind localhost TCP on an auto-port.
        # We pick the port deterministically from name so the client can
        # find it. Best-effort; users should set EMBOVIZ_<NAME>_ENDPOINT.
        return f"tcp://127.0.0.1:{8800 + (sum(map(ord, name)) % 100)}"
    sock_path = Path.home() / ".emboviz" / "sockets" / f"{name}.sock"
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    return f"ipc://{sock_path}"


# ─────────────────────────────────────────────────────────────────────
# Public ``serve()`` entry-point each adapter's ``server.py`` uses.
# ─────────────────────────────────────────────────────────────────────


def serve(
    handler_factory: Callable[..., ServiceHandler],
    *,
    name: str,
    default_kwargs: dict | None = None,
) -> None:
    """Build the handler, bind ZMQ, run the dispatch loop.

    Parameters
    ----------
    handler_factory
        Callable that returns a ServiceHandler. Called with kwargs
        taken from the ``--kwargs`` JSON option (merged on top of
        ``default_kwargs``). For VLA adapters this is usually
        ``lambda **kw: VLAModelHandler(MyAdapter(**kw))``.
    name
        Adapter name — used to derive the default socket path under
        ``~/.emboviz/sockets/<name>.sock`` and the ``EMBOVIZ_<NAME>_*``
        env-var prefix the user can override with.
    default_kwargs
        Constructor kwargs applied if the user didn't override on the
        command line. Typically the same values the adapter's
        ``AdapterSpec.default_actor_kwargs`` carries.
    """

    @click.group(invoke_without_command=False,
                 help=f"emboviz-{name} ZeroMQ worker.")
    def root() -> None:
        """Root group so ``emboviz-<name> serve`` mirrors the verb-
        first convention used by vllm / ollama / triton. Without a
        subcommand we print help instead of silently launching."""

    @root.command("serve",
                  help=f"Bind a ZMQ ROUTER and serve the {name} model.")
    @click.option("--sock", default=None,
                  help="Unix domain socket path. Default: "
                       "~/.emboviz/sockets/<name>.sock")
    @click.option("--tcp", default=None,
                  help="Bind on TCP <host:port> instead of a Unix socket "
                       "(e.g. ``--tcp 127.0.0.1:8815``). Useful on Windows "
                       "or for cross-machine setups.")
    @click.option("--kwargs", "kwargs_json", default="",
                  help="JSON dict of constructor kwargs forwarded to the "
                       "handler factory. Merged on top of the adapter's "
                       "defaults.")
    @click.option("--workers", default=1, type=int,
                  help="Thread-pool size for blocking handler calls. Most "
                       "VLA workloads should leave this at 1 because the "
                       "GPU is the bottleneck.")
    def _entry(sock: str | None, tcp: str | None,
               kwargs_json: str, workers: int) -> None:
        import json

        cli_kwargs = json.loads(kwargs_json) if kwargs_json else {}
        kwargs = {**(default_kwargs or {}), **cli_kwargs}

        # Constructor argument-validation: catch typos in kwargs at
        # process-start time, instead of mid-pipeline TypeError.
        sig = None
        try:
            sig = inspect.signature(handler_factory)
        except (TypeError, ValueError):
            pass
        if sig is not None and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            unknown = set(kwargs) - set(sig.parameters)
            if unknown:
                raise click.ClickException(
                    f"{name}: unknown constructor kwargs {sorted(unknown)}. "
                    f"Accepts: {sorted(sig.parameters)}"
                )

        endpoint = _resolve_endpoint(name, sock, tcp)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

        print(f"[server] loading {name} ...", flush=True)
        handler = handler_factory(**kwargs)
        if not hasattr(handler, "methods"):
            raise click.ClickException(
                f"{name}: handler_factory did not return a ServiceHandler "
                f"(missing .methods property). Got: {type(handler).__name__}"
            )
        print(f"[server] {name} loaded; entering serve loop", flush=True)

        try:
            asyncio.run(_serve_async(handler, endpoint,
                                     n_workers=max(1, workers)))
        except KeyboardInterrupt:
            print("[server] interrupted; bye", flush=True)

    root()
