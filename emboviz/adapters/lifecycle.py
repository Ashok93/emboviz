"""Spawn / connect / shutdown Ray actors for adapter execution.

Each VLA adapter runs as a long-lived Ray actor in its OWN venv. The
actor is launched with::

    ray.remote(actor_cls).options(
        runtime_env={
            "py_executable": "/path/to/adapter/venv/bin/python",
            "env_vars":      spec.runtime_env_vars,
        },
        name=actor_name,
        lifetime="detached" if persistent else None,
    ).remote(**kwargs)

The ``py_executable`` runtime_env field is the documented
Ray-recommended way to run a single actor in a different interpreter
without paying for conda/docker provisioning. See:

    https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
    https://docs.ray.io/en/latest/ray-core/api/runtime-env.html

The runtime venv path is fixed at ``~/.emboviz/venvs/<name>``. The
``install-<name>`` CLI subcommand populates it; this module only reads.
"""

from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from emboviz.adapters.protocol import AdapterSpec
from emboviz.adapters.registry import find_adapter


def venv_root() -> Path:
    """Where the per-adapter runtime venvs live.

    ``EMBOVIZ_VENVS_DIR`` overrides for testing / multi-user systems.
    Default is ``~/.emboviz/venvs``. Per-adapter subdir = ``<name>``.
    """
    override = os.environ.get("EMBOVIZ_VENVS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".emboviz" / "venvs"


def venv_path(name: str) -> Path:
    return venv_root() / name


def venv_python(name: str) -> Path:
    """Path to the Python interpreter inside the adapter's runtime venv.

    Raises FileNotFoundError with a clear remediation if the venv has
    not been created yet (the user hasn't run ``emboviz install-<name>``).
    """
    p = venv_path(name) / "bin" / "python"
    if not p.exists():
        # Windows layout fallback (defensive — emboviz targets Linux but
        # someone WILL try this on macOS / WSL):
        alt = venv_path(name) / "Scripts" / "python.exe"
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Adapter '{name}' runtime venv not found at {venv_path(name)}. "
            f"Run `emboviz install-{name}` first to create it."
        )
    return p


@dataclass
class ActorHandle:
    """A live Ray actor + the spec it was spawned from.

    Held by :class:`emboviz.adapters.client.RayVLAClient`. The
    underlying actor object is opaque (Ray's ActorHandle) — callers
    should go through the client.
    """

    spec: AdapterSpec
    actor: Any                                   # ray.actor.ActorHandle
    name: str                                    # the named-actor name
    persistent: bool


def _ensure_ray_initialized() -> None:
    """Start Ray if it isn't running. Idempotent.

    We use a local single-node cluster: heavy actors live in the same
    physical box but in their own OS processes (different venvs). For
    multi-node deployments callers should ``ray.init(address=...)``
    themselves before calling :func:`connect`.
    """
    import ray

    if not ray.is_initialized():
        # ``ignore_reinit_error=True`` lets multiple emboviz commands in
        # the same process call connect() without tripping over each
        # other. ``log_to_driver=True`` keeps actor stdout/stderr in the
        # foreground — critical for the model-load progress prints that
        # users need to see during the first cold start.
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=True,
            namespace="emboviz",
            # Don't reserve all the cores for Ray task scheduling — most
            # of the work is inside one or two big GPU actors anyway.
            include_dashboard=False,
        )


def _actor_class(spec: AdapterSpec):
    """Import the actor CLASS from the adapter package.

    The import happens in the CALLER'S process (core), not in the
    runtime venv — Ray then re-imports the class inside the venv when
    materialising the actor. Either:

      (a) The actor module is import-light enough to load in core's
          venv too (recommended — keep heavy deps behind lazy imports
          in ``__init__``), OR
      (b) The adapter's spec module re-exports a thin shim.

    Pattern (a) is what the bundled adapters do.
    """
    module_path, _, class_name = spec.actor_import_path.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def connect(
    name: str,
    *,
    actor_kwargs: Optional[dict] = None,
    actor_name: Optional[str] = None,
    persistent: bool = False,
    num_gpus: float = 1.0,
) -> ActorHandle:
    """Spawn (or attach to a named) actor for the adapter ``name``.

    Parameters
    ----------
    name
        Adapter CLI alias (``"openvla"``, ``"pi0"``, ...).
    actor_kwargs
        Forwarded to the actor's ``__init__``. Merged on top of
        :attr:`AdapterSpec.default_actor_kwargs`.
    actor_name
        Optional named-actor handle. When set, repeated connects with
        the same name re-attach to the SAME running actor (model stays
        warm in GPU memory between CLI calls). Without it, every
        ``emboviz analyze`` cold-loads the model.
    persistent
        If True, the actor outlives this driver process (``lifetime=
        "detached"``). Combined with ``actor_name``, this gives the
        "warm model server" pattern.
    num_gpus
        How many GPUs to claim. Default 1.0 — VLAs use all of one GPU.

    Returns
    -------
    ActorHandle
        Wrapped in :class:`emboviz.adapters.client.RayVLAClient` by the
        caller (CLI / runner).
    """
    spec = find_adapter(name)
    import ray

    _ensure_ray_initialized()

    # If a named persistent actor already exists, reuse it. Saves
    # the 30-90s VLA model load on every CLI invocation.
    if actor_name is not None:
        try:
            handle = ray.get_actor(actor_name, namespace="emboviz")
            return ActorHandle(spec=spec, actor=handle, name=actor_name,
                               persistent=persistent)
        except ValueError:
            pass  # not found — fall through to create

    py = venv_python(spec.name)
    cls = _actor_class(spec)

    options: dict = {
        "runtime_env": {
            "py_executable": str(py),
            "env_vars": dict(spec.runtime_env_vars),
        },
        "num_gpus": num_gpus,
    }
    if actor_name is not None:
        options["name"] = actor_name
    if persistent:
        options["lifetime"] = "detached"

    kwargs = {**spec.default_actor_kwargs, **(actor_kwargs or {})}
    handle = ray.remote(cls).options(**options).remote(**kwargs)

    return ActorHandle(
        spec=spec,
        actor=handle,
        name=actor_name or f"<anon:{spec.name}>",
        persistent=persistent,
    )


def shutdown(handle: ActorHandle, *, kill: bool = False) -> None:
    """Release the actor.

    Detached (persistent) actors must be explicitly killed; ephemeral
    actors are reaped when their handles go out of scope. We expose a
    single call so cleanup logic doesn't need to know the difference.
    """
    import ray

    if kill or handle.persistent:
        ray.kill(handle.actor, no_restart=True)


def _editable_install_path(dist_name: str) -> Optional[Path]:
    """Return the local source directory if ``dist_name`` is installed
    editable in the current Python; None otherwise.

    Uses :mod:`importlib.metadata` to find the dist's ``direct_url.json``
    marker that ``pip install -e`` writes alongside the egg-info. That
    file's ``dir_info.editable == True`` is the canonical "installed
    in editable mode" signal.
    """
    import json as _json
    from importlib import metadata as importlib_metadata

    try:
        dist = importlib_metadata.distribution(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None

    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text is None:
        return None
    try:
        info = _json.loads(direct_url_text)
    except ValueError:
        return None
    dir_info = info.get("dir_info") or {}
    if not dir_info.get("editable"):
        return None
    url = info.get("url", "")
    if url.startswith("file://"):
        return Path(url[len("file://"):])
    return None


def _rewrite_pip_for_dev(runtime_pip: tuple[str, ...]) -> list[str]:
    """Replace ``emboviz`` / ``emboviz-<x>`` deps with their editable
    local paths if they happen to be editable in the CURRENT venv.

    This is what makes "dev path = user path" hold for the runtime
    venv install. A developer running from a git checkout has
    ``emboviz`` and ``emboviz-openvla`` installed editable; the
    runtime venv should pick those up rather than the (probably
    nonexistent or stale) PyPI release. End-users with both shims
    installed from PyPI hit None on both lookups and the spec's
    runtime_pip is used verbatim.
    """
    rewritten: list[str] = []
    for req in runtime_pip:
        # Only consider plain ``name`` or ``name==X`` style specs; PEP
        # 508 markers / direct URLs / -e refs already specify exactly
        # what they want.
        if req.startswith("-") or "@" in req or "/" in req:
            rewritten.append(req)
            continue
        # Grab the bare dist name.
        bare = req.split(";")[0].split("[")[0]
        for op in ("==", ">=", "<=", "!=", "~=", ">", "<"):
            bare = bare.split(op)[0]
        bare = bare.strip()
        if bare in ("emboviz",) or bare.startswith("emboviz-"):
            local = _editable_install_path(bare)
            if local is not None:
                rewritten.append("-e")
                rewritten.append(str(local))
                continue
        rewritten.append(req)
    return rewritten


def install_venv(spec: AdapterSpec, *, force: bool = False) -> Path:
    """Create the adapter's runtime venv and install its heavy deps.

    Called by the ``emboviz install-<name>`` CLI subcommand. The user-
    facing flow:

        uv pip install emboviz-<name>     # adapter shim
        emboviz install-<name>            # this function (heavy deps)
        emboviz analyze --model <name>    # works

    The implementation shells out to ``uv venv`` and ``uv pip install``
    — same commands the README documents — so dev and user paths are
    byte-identical (CLAUDE.md "dev path is the user path" rule). The
    one place we diverge: ``emboviz`` / ``emboviz-<name>`` entries in
    ``spec.runtime_pip`` are rewritten to ``-e <local_path>`` if those
    shims are installed editable in the current venv (i.e. running
    from a git checkout).
    """
    import subprocess

    path = venv_path(spec.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if force:
            shutil.rmtree(path)
        else:
            # Idempotent: re-running install-<name> on an existing venv
            # just re-runs pip install (lets users add a missing dep
            # without nuking the GPU-side wheels they already have).
            pass

    if not path.exists():
        subprocess.run(
            ["uv", "venv", str(path), "--python", spec.requires_python],
            check=True,
        )

    env = dict(os.environ)
    env.update(spec.runtime_env_vars)

    py = path / "bin" / "python"
    requirements = _rewrite_pip_for_dev(spec.runtime_pip)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(py), *requirements],
        check=True,
        env=env,
    )

    return path
