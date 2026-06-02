"""Worker process lifecycle helpers — install, locate, optional spawn.

The architecture is **Pattern Y**: each adapter runs as an
independent long-lived ZMQ worker. Production / cloud deployments
manage these workers externally (systemd unit, docker compose,
Kubernetes Deployment) and core just connects to the known endpoint.
For local-development convenience we also support **opportunistic
auto-spawn**: if the user invokes ``emboviz analyze --config <file>``
whose ``model.adapter`` names an adapter with no worker already
running, we ``subprocess.Popen`` the
adapter's ``server`` entry-point in its runtime venv and wait until
it answers ``ping``. The spawned worker stays running between CLI
invocations, so the model only cold-loads once per session.

This module is the **only** place that knows about subprocesses,
venv paths, and PID files. Everything else (client.py, the
diagnostics, CLI commands) just sees endpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from emboviz.adapters.client import (
    RpcClient,
    ZMQAdapterClient,
    ZMQReaderClient,
    default_endpoint,
)
from emboviz.adapters.protocol import AdapterSpec
from emboviz.adapters.registry import find_adapter
from emboviz.adapters.reader_registry import find_reader


# ─────────────────────────────────────────────────────────────────────
# Filesystem layout
# ─────────────────────────────────────────────────────────────────────


def venv_root() -> Path:
    """Where the per-adapter isolated runtime venvs live.

    Override with the ``EMBOVIZ_VENVS_DIR`` env var (useful on shared
    machines where ``~/.emboviz/venvs`` is on a slow filesystem).
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
        # Windows layout fallback.
        alt = venv_path(name) / "Scripts" / "python.exe"
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Adapter '{name}' runtime venv not found at {venv_path(name)}. "
            f"Run `emboviz install-{name}` first to create it."
        )
    return p


def _run_dir() -> Path:
    """Directory holding one JSON run-record per live worker."""
    p = Path.home() / ".emboviz" / "run"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _endpoint_stem(endpoint: str) -> str:
    """Filesystem-safe identity for an endpoint, shared by its run-record and
    its ipc socket. For ``ipc://…/<stem>.sock`` this is ``<stem>`` — so the
    per-dataset reader sockets (which carry a path hash) each map to a distinct
    record, and ``stop`` can pair a socket with its pid exactly."""
    if endpoint.startswith("ipc://"):
        return Path(endpoint[len("ipc://"):]).stem
    return endpoint.replace("://", "-").replace(":", "-").replace("/", "_")


def _run_record_path(endpoint: str) -> Path:
    return _run_dir() / f"{_endpoint_stem(endpoint)}.json"


def record_worker(name: str, endpoint: str, pid: int) -> None:
    """Register a spawned worker so ``emboviz stop`` can find it: one JSON
    record per endpoint holding the adapter name, endpoint and pid. Best-effort
    — if the write fails, ``stop`` still finds the worker by its socket and can
    shut it down gracefully; only ``--force`` (which needs the pid) is lost."""
    try:
        _run_record_path(endpoint).write_text(
            json.dumps({"name": name, "endpoint": endpoint, "pid": int(pid)})
        )
    except OSError:
        pass


def forget_worker(endpoint: str) -> None:
    """Remove a worker's run-record (called once it has been stopped)."""
    try:
        _run_record_path(endpoint).unlink()
    except (FileNotFoundError, OSError):
        pass


def _read_record(path: Path) -> Optional[dict]:
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and "endpoint" in rec else None


def log_file(name: str) -> Path:
    p = Path.home() / ".emboviz" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.log"


# ─────────────────────────────────────────────────────────────────────
# Editable-install detection (so dev installs don't pull stale wheels
# from PyPI inside the runtime venv).
# ─────────────────────────────────────────────────────────────────────


def _editable_install_path(dist_name: str) -> Optional[Path]:
    """Return the local source directory if ``dist_name`` is installed
    editable in the current Python; None otherwise.

    Uses ``importlib.metadata``'s ``direct_url.json`` marker that
    ``pip install -e`` writes alongside the egg-info.
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


def repos_root() -> Path:
    """Where adapter-side git checkouts live. Override with
    ``EMBOVIZ_REPOS_DIR`` for shared / fast-filesystem setups."""
    override = os.environ.get("EMBOVIZ_REPOS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".emboviz" / "repos"


def _materialise_git_clones(
    runtime_pip: tuple[str, ...] | list[str],
) -> list[str]:
    """Replace ``<name> @ git+https://...`` entries with ``-e <local>``
    after cloning the repo to ``repos_root()/<repo-name>``. Other
    requirement strings pass through unchanged.

    Idempotent: an existing clone is reused (``git fetch + checkout``);
    a missing one is cloned from scratch. We pin to whatever the URL
    fragment specifies (``@<ref>``) so reruns get the same code.
    """
    out: list[str] = []
    for req in runtime_pip:
        parsed = _parse_named_git_requirement(req)
        if parsed is None:
            out.append(req)
            continue
        name, url, ref = parsed
        repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        local = repos_root() / repo_name
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            subprocess.run(["git", "clone", url, str(local)], check=True)
        if ref:
            subprocess.run(["git", "-C", str(local), "fetch", "--quiet"], check=True)
            subprocess.run(["git", "-C", str(local), "checkout", ref], check=True)
        out.append("-e")
        out.append(str(local))
    return out


def _parse_named_git_requirement(req: str) -> Optional[tuple[str, str, str]]:
    """Decompose ``<name> @ git+https://<host>/<owner>/<repo>(.git)?@<ref>?``
    into ``(name, url, ref)``. Returns None for anything that doesn't
    match that exact shape so other requirement strings pass through
    untouched."""
    if " @ git+" not in req:
        return None
    name, _, url = req.partition(" @ git+")
    name = name.strip()
    url = url.strip()
    ref = ""
    if "@" in url and not url.startswith("git@"):
        # split on the LAST @ — leaves room for URLs that contain @ in
        # the auth part (rare for our case).
        url, _, ref = url.rpartition("@")
    return (name, url, ref)


def _rewrite_pip_for_dev(runtime_pip: tuple[str, ...] | list[str]) -> list[str]:
    """Replace ``emboviz`` / ``emboviz-<x>`` deps with their local
    editable paths if installed editable in the current venv.

    Makes "dev path = user path": developers running from a git
    checkout pick up their working tree inside the runtime venv;
    end-users with both shims installed from PyPI hit None on both
    lookups and the spec's runtime_pip is used verbatim.
    """
    rewritten: list[str] = []
    for req in runtime_pip:
        # Anything already pinned via direct URL / editable / path → leave alone.
        if req.startswith("-") or "@" in req or "/" in req:
            rewritten.append(req)
            continue
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


# ─────────────────────────────────────────────────────────────────────
# Install: create the venv + install the heavy deps.
# ─────────────────────────────────────────────────────────────────────


def install_venv(spec: AdapterSpec, *, force: bool = False) -> Path:
    """Create the adapter's runtime venv and install its heavy deps.

    The user-facing flow:

        uv pip install emboviz-<name>     # adapter shim
        emboviz install-<name>            # this function (heavy deps)
        emboviz-<name> serve              # start the worker
        emboviz analyze --config <file>   # connect (config's model.adapter = <name>)

    Shells out to ``uv venv`` and ``uv pip install`` — exact same
    commands the README documents — so dev and user paths are
    byte-identical (CLAUDE.md "dev path is the user path" rule). The
    one place we diverge: ``emboviz`` / ``emboviz-<name>`` entries in
    ``spec.runtime_pip`` are rewritten to ``-e <local_path>`` if those
    shims are installed editable in the current venv (i.e. running
    from a git checkout).
    """
    path = venv_path(spec.name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and force:
        shutil.rmtree(path)
    if not path.exists():
        subprocess.run(
            ["uv", "venv", str(path), "--python", spec.requires_python],
            check=True,
        )

    env = dict(os.environ)
    env.update(spec.runtime_env_vars)

    py = path / "bin" / "python"

    # Materialise any ``<name> @ git+https://<host>/<owner>/<repo>...``
    # entry to a local ``-e <clone>`` (clone once to ``~/.emboviz/repos/
    # <repo>``, reused on rerun). Providers reference local wheels/paths
    # *inside their own pyproject* — e.g. NVIDIA Isaac-GR00T sources its
    # ``torchcodec`` / ``flash-attn`` wheels via ``{ path = ... }`` and
    # ``file://`` refs. uv accepts those as local-to-local paths when the
    # package is installed editable from a clone, but refuses the same
    # refs reached transitively through a ``git+`` URL. Cloning also
    # gives every adapter a stable, inspectable checkout we reuse for the
    # adapter's ``demo_data`` etc.
    requirements = _materialise_git_clones(spec.runtime_pip)
    requirements = _rewrite_pip_for_dev(requirements)

    cmd = ["uv", "pip", "install", "--python", str(py)]
    override_path: Optional[Path] = None
    if spec.runtime_pip_exclude:
        # Drop provider-declared deps we deliberately don't install
        # (e.g. gr00t's flash-attn). uv only honours an override when the
        # package is actually requested, and an always-false marker makes
        # it unsatisfiable everywhere — the documented way to exclude a
        # dependency. This keeps the adapter provider-driven: install the
        # upstream package WITH its own dependency closure, minus the few
        # we subtract here. No hand-mirrored, drift-prone dep list.
        override_path = _write_exclude_override(spec)
        cmd += ["--override", str(override_path)]
    cmd += requirements

    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        if override_path is not None:
            override_path.unlink(missing_ok=True)

    return path


def _write_exclude_override(spec: AdapterSpec) -> Path:
    """Write a uv ``--override`` file that drops every package in
    ``spec.runtime_pip_exclude`` from resolution.

    Each line names a package guarded by ``sys_platform == 'never'`` — a
    marker that is false on every real platform, so when the package is
    requested (directly or transitively) uv resolves it to "no version
    applicable" and omits it. See uv's resolution docs (overrides):
    https://docs.astral.sh/uv/concepts/resolution/.
    """
    body = "".join(f"{pkg} ; sys_platform == 'never'\n"
                   for pkg in spec.runtime_pip_exclude)
    fd, name = tempfile.mkstemp(
        prefix=f"emboviz-{spec.name}-exclude-", suffix=".txt",
    )
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    return Path(name)


# ─────────────────────────────────────────────────────────────────────
# Connect: get a client to the adapter, spawning the worker if needed.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class WorkerHandle:
    """Returned by :func:`connect`. Tracks the spawned subprocess (if
    we spawned it) plus the live :class:`ZMQAdapterClient`. Both
    ``Popen.terminate()`` and ``ZMQAdapterClient.close()`` are
    idempotent so callers can ``handle.close()`` unconditionally."""

    name: str
    endpoint: str
    client: RpcClient            # ZMQAdapterClient (model) or ZMQReaderClient (dataset)
    process: Optional[subprocess.Popen] = None
    spawned: bool = False

    def close(self, *, terminate: bool = False) -> None:
        """Close the client. If we spawned the worker AND ``terminate``
        is True, also stop it. Default leaves the worker running so
        subsequent CLI invocations stay warm — exactly the same pattern
        ``vllm serve`` / ``ollama`` use."""
        self.client.close()
        if terminate and self.process is not None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            except Exception:
                pass
            forget_worker(self.endpoint)


# Workers THIS process spawned, for auto-teardown by ``analyze`` on exit and on
# Ctrl-C. A worker we ATTACHED to (already running / started via
# ``emboviz-<x> serve``) is never added — we only ever stop what we started.
_SPAWNED_THIS_PROCESS: list[WorkerHandle] = []


def teardown_spawned_workers() -> None:
    """Stop every worker THIS process spawned.

    Called by ``emboviz analyze`` on normal exit and on Ctrl-C (unless
    ``--keep-warm``). Graceful: each worker gets SIGTERM, runs its close
    handlers — closing the model and freeing the GPU — and exits. Idempotent:
    handles are drained as they close, so a second call is a no-op.
    """
    if not _SPAWNED_THIS_PROCESS:
        return
    print(
        f"[emboviz] stopping {len(_SPAWNED_THIS_PROCESS)} worker(s) spawned this "
        "run (use --keep-warm to keep them loaded) ...",
        file=sys.stderr, flush=True,
    )
    while _SPAWNED_THIS_PROCESS:
        handle = _SPAWNED_THIS_PROCESS.pop()
        try:
            handle.close(terminate=True)
        except Exception:
            pass


def _is_alive(endpoint: str, *, timeout_ms: int = 1500) -> bool:
    """Cheap reachability check. True if a worker responds to ``ping``
    on ``endpoint`` within ``timeout_ms`` milliseconds."""
    try:
        c = ZMQAdapterClient(name="_probe", endpoint=endpoint, timeout_ms=timeout_ms)
    except Exception:
        return False
    try:
        return c.ping(timeout_ms=timeout_ms)
    finally:
        c.close()


def _spawn_worker(
    spec: AdapterSpec, endpoint: str,
    actor_kwargs: Optional[dict] = None,
) -> subprocess.Popen:
    """Start the adapter's worker process in its isolated runtime venv.

    Tries the ``[project.scripts]`` console entry-point first (e.g.
    ``emboviz-openvla`` on the venv's PATH). Falls back to ``python -m
    <server_module>``. The output is appended to
    ``~/.emboviz/logs/<name>.log`` so users can ``tail -f`` it.

    ``actor_kwargs`` are per-run constructor overrides (e.g. a user's
    fine-tuned ``checkpoint``). They are merged ON TOP of the spec's
    declared ``default_actor_kwargs`` and forwarded to the worker via
    ``serve --kwargs <json>``, which hands them to the model
    constructor. This is the single mechanism by which a user points an
    adapter at THEIR checkpoint instead of the spec default.
    """
    venv_bin = venv_python(spec.name).parent

    console = venv_bin / spec.console_script
    if console.exists():
        cmd = [str(console)]
    else:
        cmd = [str(venv_python(spec.name)), "-m", spec.server_module]

    # The worker's CLI is a Click group whose ``serve`` subcommand binds
    # the socket — same shape as ``vllm serve`` / ``ollama serve``.
    cmd.append("serve")

    # ipc:// endpoints carry a path we hand to --sock; tcp:// endpoints
    # take --tcp host:port.
    if endpoint.startswith("ipc://"):
        cmd += ["--sock", endpoint[len("ipc://"):]]
    elif endpoint.startswith("tcp://"):
        cmd += ["--tcp", endpoint[len("tcp://"):]]
    else:
        raise ValueError(f"unsupported endpoint scheme: {endpoint!r}")

    # Construction kwargs: the spec's declared defaults overlaid with any
    # per-run overrides (a user's fine-tuned checkpoint, alternate
    # unnorm_key, etc.). Forwarded to the worker, which merges them over
    # its own serve() defaults and hands them to the model constructor.
    merged_kwargs = {
        **(getattr(spec, "default_actor_kwargs", None) or {}),
        **(actor_kwargs or {}),
    }
    if merged_kwargs:
        cmd += ["--kwargs", json.dumps(merged_kwargs)]

    env = dict(os.environ)
    env.update(spec.runtime_env_vars)

    log_path = log_file(spec.name)
    log_fh = open(log_path, "ab")
    log_fh.write(f"\n=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] starting {' '.join(cmd)}\n".encode())
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,         # detach from our process group
    )
    # Register the worker (name + endpoint + pid) so ``emboviz stop`` can
    # find it and shut it down — gracefully or, with --force, by pid.
    record_worker(spec.name, endpoint, proc.pid)
    return proc


def _wait_for_ready(endpoint: str, proc: Optional[subprocess.Popen],
                    timeout_s: int) -> None:
    """Poll the endpoint until the worker answers ``ping``, or the
    spawned process dies, or the deadline expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_alive(endpoint, timeout_ms=500):
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"adapter worker exited before becoming ready "
                f"(exit_code={proc.returncode}); check the log."
            )
        time.sleep(0.5)
    raise TimeoutError(
        f"adapter worker did not become ready on {endpoint} within {timeout_s}s"
    )


def _runtime_venv_ready(name: str) -> bool:
    """True iff the runtime venv at ``~/.emboviz/venvs/<name>`` exists
    and its Python interpreter is callable. Cheap stat-only check."""
    path = venv_path(name)
    if not path.exists():
        return False
    py = path / "bin" / "python"
    if py.exists():
        return True
    return (path / "Scripts" / "python.exe").exists()


def _ensure_runtime_venv(
    spec: AdapterSpec, *, quiet: bool = False,
) -> Path:
    """Create the adapter's runtime venv on demand if missing.

    Visible progress: prints a one-line "[emboviz] setting up …" notice
    BEFORE the slow install starts (~MB-to-GB pip download) so the
    user knows what's happening. After the install completes the
    function returns the venv path; the caller proceeds to spawn the
    worker.

    Idempotent: if the venv already looks ready, returns immediately
    without re-running pip.
    """
    if _runtime_venv_ready(spec.name):
        return venv_path(spec.name)

    if not quiet:
        import sys
        py_size_hint = {
            "openvla": "~6 GB",
            "oft":     "~6 GB",
            "pi0":     "~8 GB (downloads checkpoint + Triton autotune ~5-10 min on first inference)",
            "gr00t":   "~7 GB",
            "sam3":    "~5 GB (downloads facebook/sam3 ~3.4 GB; gated, needs HF_TOKEN)",
            "lerobot": "~3 GB (lerobot 0.5.x / v3.0 + torch + video-decode stack)",
            "reader-gr00t": "~3 GB (lerobot 0.3.x / v2.1 + torch + video-decode stack)",
        }.get(spec.name, "")
        print(
            f"[emboviz] first run for '{spec.name}' — materialising the "
            f"runtime venv at {venv_path(spec.name)} "
            f"{('(' + py_size_hint + ') ') if py_size_hint else ''}...",
            file=sys.stderr, flush=True,
        )
    install_venv(spec, force=False)
    return venv_path(spec.name)


def _connect_with_spec(
    spec: AdapterSpec,
    *,
    client_cls,
    actor_kwargs: Optional[dict] = None,
    auto_spawn: bool = True,
    auto_install: bool = True,
    timeout_s: int = 600,
    endpoint: Optional[str] = None,
) -> WorkerHandle:
    """Shared connect lifecycle for ANY worker — model adapter OR dataset
    reader. They use one mechanism (isolated venv + ZMQ worker); only the
    typed client facade differs.

    States: (1) attach to a warm worker; (2) spawn into an existing
    runtime venv; (3) materialise the venv (``auto_install``) then spawn.
    ``client_cls`` is :class:`ZMQAdapterClient` (VLA) or
    :class:`ZMQReaderClient` (dataset). ``endpoint`` overrides the default
    socket (the reader uses a per-dataset socket so different datasets get
    distinct warm workers) — the runtime venv is still keyed by ``spec.name``.
    """
    name = spec.name
    endpoint = endpoint or default_endpoint(name)

    # ── 1. Already alive ────────────────────────────────────────────
    if _is_alive(endpoint):
        # A warm worker carries whatever construction kwargs it was
        # spawned with. If the caller requests specific per-run kwargs (a
        # model checkpoint, a dataset path), we must NOT silently attach
        # to a worker that may hold a DIFFERENT model/dataset — that would
        # diagnose the wrong thing. Refuse with a clear remediation. (No
        # kwargs → attaching to the warm worker is the intended fast path.)
        if actor_kwargs:
            raise RuntimeError(
                f"a '{name}' worker is already running at {endpoint}, "
                f"loaded with its own configuration. Refusing to route "
                f"this run's kwargs {actor_kwargs!r} to it — it may hold a "
                f"different model/dataset, and sending work to the wrong "
                f"one would be a silent-wrong-answer bug. Stop the running "
                f"worker first (kill the `emboviz-{name} serve` process, or "
                f"remove {endpoint[len('ipc://'):] if endpoint.startswith('ipc://') else endpoint}), "
                f"then re-run so a fresh worker loads with your kwargs."
            )
        client = client_cls(name=name, endpoint=endpoint)
        return WorkerHandle(
            name=name, endpoint=endpoint, client=client, spawned=False,
        )

    # ── 2. Ensure the runtime venv exists ───────────────────────────
    if not _runtime_venv_ready(name):
        if not auto_install:
            raise RuntimeError(
                f"runtime venv for '{name}' is missing at {venv_path(name)}. "
                f"Run `emboviz install-{name}` to create it, or pass "
                "--auto-install (default) to let emboviz set it up "
                "automatically."
            )
        _ensure_runtime_venv(spec)

    # ── 3. Spawn the worker if not already running ──────────────────
    if not auto_spawn:
        raise RuntimeError(
            f"no worker reachable at {endpoint} for '{name}'. Start one "
            f"with:\n    emboviz-{name} serve\nor pass --auto-spawn "
            "(default) to let emboviz launch one."
        )

    proc = _spawn_worker(spec, endpoint, actor_kwargs=actor_kwargs)
    try:
        _wait_for_ready(endpoint, proc, timeout_s=timeout_s)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
        raise

    client = client_cls(name=name, endpoint=endpoint)
    handle = WorkerHandle(
        name=name, endpoint=endpoint, client=client, process=proc, spawned=True,
    )
    _SPAWNED_THIS_PROCESS.append(handle)
    return handle


def connect(
    name: str,
    *,
    actor_kwargs: Optional[dict] = None,
    auto_spawn: bool = True,
    auto_install: bool = True,
    timeout_s: int = 600,
) -> WorkerHandle:
    """Return a live :class:`WorkerHandle` for the named VLA model adapter.

    Three states: (1) attach to a warm worker; (2) spawn into an existing
    runtime venv; (3) ``auto_install`` materialises the venv then spawn.
    Pass ``auto_install=False`` / ``auto_spawn=False`` to require the user
    to have run ``emboviz install-<name>`` / ``emboviz-<name> serve``.
    ``timeout_s`` bounds cold-load on first spawn (π0's Triton autotune
    cold can take minutes).
    """
    return _connect_with_spec(
        find_adapter(name), client_cls=ZMQAdapterClient,
        actor_kwargs=actor_kwargs, auto_spawn=auto_spawn,
        auto_install=auto_install, timeout_s=timeout_s,
    )


def connect_reader(
    name: str,
    *,
    reader_kwargs: Optional[dict] = None,
    auto_spawn: bool = True,
    auto_install: bool = True,
    timeout_s: int = 600,
) -> ZMQReaderClient:
    """Return a live :class:`ZMQReaderClient` (an EpisodeSource) for the
    named dataset reader (e.g. ``"lerobot"``).

    Identical lifecycle to :func:`connect` — same isolated venv, same
    spawn-and-wait over the wire — but resolves the reader spec from the
    ``emboviz.readers`` entry-point group and returns the EpisodeSource
    client directly (the worker stays warm, detached, exactly like a
    model worker). ``reader_kwargs`` is the run config's ``dataset``
    section, forwarded to the reader's source builder.

    A reader worker is bound to ONE dataset at spawn (``build_*_source``
    runs from these kwargs), so we give each dataset its OWN socket
    (keyed by a hash of the path). Different datasets therefore get
    distinct warm readers that coexist, and a request is never routed to
    a reader holding a different dataset — which would be a silent-wrong-
    answer bug. The runtime venv is shared (one ``lerobot`` venv).
    """
    path = str((reader_kwargs or {}).get("path", ""))
    tag = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12] if path else "default"
    endpoint = default_endpoint(f"{name}-{tag}")
    handle = _connect_with_spec(
        find_reader(name), client_cls=ZMQReaderClient,
        actor_kwargs=reader_kwargs, auto_spawn=auto_spawn,
        auto_install=auto_install, timeout_s=timeout_s, endpoint=endpoint,
    )
    return handle.client  # type: ignore[return-value]


def shutdown(handle: WorkerHandle, *, terminate: bool = False) -> None:
    """Compatibility shim: close the client (and optionally the worker)."""
    handle.close(terminate=terminate)


# ─────────────────────────────────────────────────────────────────────
# Stop: discover running workers and shut them down (graceful or forced).
# ─────────────────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """True iff a process with ``pid`` currently exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, owned by another user
    return True


def _wait_pid_gone(pid: int, timeout_s: float) -> bool:
    """Poll until ``pid`` exits or the deadline passes. Returns True if gone."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return not _pid_alive(pid)


def _wait_endpoint_gone(endpoint: str, timeout_s: float) -> bool:
    """Poll until ``endpoint`` stops answering ``ping``. Used to confirm a
    worker exited when its pid is unknown (record missing / write failed)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_alive(endpoint, timeout_ms=500):
            return True
        time.sleep(0.2)
    return not _is_alive(endpoint, timeout_ms=500)


@dataclass
class WorkerInfo:
    """A discovered worker: adapter name, endpoint, pid (if recorded), its ipc
    socket file, its run-record file, and whether it answers ``ping`` now."""
    name: str
    endpoint: str
    pid: Optional[int]
    socket_path: Optional[Path]
    record_path: Optional[Path]
    alive: bool


def list_running_workers() -> list[WorkerInfo]:
    """Every emboviz worker this host has a run-record for.

    The run-record — written on spawn, removed on stop — is the single
    registry. Liveness is confirmed by pinging the endpoint, so a stale record
    (left by a crash or ``kill -9``) is still listed and gets cleaned up by
    ``stop``. The socket path is derived from the endpoint for that cleanup.
    """
    out: list[WorkerInfo] = []
    for rp in sorted(_run_dir().glob("*.json")):
        rec = _read_record(rp)
        if rec is None:
            continue
        endpoint = rec["endpoint"]
        sp = Path(endpoint[len("ipc://"):]) if endpoint.startswith("ipc://") else None
        out.append(WorkerInfo(
            name=rec.get("name", rp.stem),
            endpoint=endpoint,
            pid=rec.get("pid"),
            socket_path=sp,
            record_path=rp,
            alive=_is_alive(endpoint, timeout_ms=1000),
        ))
    return out


def _cleanup_worker(w: WorkerInfo) -> None:
    """Remove a stopped worker's socket file and run-record."""
    if w.socket_path is not None:
        try:
            w.socket_path.unlink()
        except (FileNotFoundError, OSError):
            pass
    forget_worker(w.endpoint)


def _stop_worker(w: WorkerInfo, *, force: bool, timeout_s: float) -> dict:
    """Stop one worker and report exactly what happened.

    Graceful (default): send the transport-level ``shutdown`` RPC — the worker
    closes its model (freeing the GPU) and exits — then verify the process is
    gone. ``force``: SIGKILL its pid. Never a silent best-effort: the returned
    ``action`` always reflects the verified outcome, and an unresponsive worker
    is reported (so the user can choose ``--force``), never killed implicitly.
    """
    base = {"name": w.name, "endpoint": w.endpoint, "pid": w.pid}

    # Not answering ``ping``: it is either already gone, or wedged with an
    # unreachable socket. Decide from the pid — do not guess.
    if not w.alive:
        if w.pid is not None and _pid_alive(w.pid):
            if not force:
                return {**base, "action": "unreachable",
                        "detail": "process alive but not answering; re-run with --force"}
            try:
                os.kill(w.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            killed = _wait_pid_gone(w.pid, timeout_s=5.0)
            _cleanup_worker(w)
            return {**base, "action": "killed" if killed else "kill-failed"}
        _cleanup_worker(w)
        return {**base, "action": "already-stopped"}

    if force:
        if w.pid is None:
            return {**base, "action": "no-pid",
                    "detail": "no run-record for this socket; cannot SIGKILL. "
                              "Graceful `emboviz stop` (no --force) still works."}
        try:
            os.kill(w.pid, signal.SIGKILL)
        except ProcessLookupError:
            _cleanup_worker(w)
            return {**base, "action": "already-stopped"}
        killed = _wait_pid_gone(w.pid, timeout_s=5.0)
        _cleanup_worker(w)
        return {**base, "action": "killed" if killed else "kill-failed"}

    # Graceful: ask the worker to shut itself down, then verify it exited.
    client = ZMQAdapterClient(name="_stop", endpoint=w.endpoint, timeout_ms=5000)
    try:
        client.shutdown()
    except Exception:
        # The worker may drop the connection mid-shutdown before replying;
        # that is not an error — the truth is whether it exits, checked next.
        pass
    finally:
        client.close()

    gone = (_wait_pid_gone(w.pid, timeout_s) if w.pid is not None
            else _wait_endpoint_gone(w.endpoint, timeout_s))
    if gone:
        _cleanup_worker(w)
        return {**base, "action": "stopped"}
    return {**base, "action": "still-running",
            "detail": f"graceful shutdown did not exit within {timeout_s:.0f}s; "
                      "re-run with --force"}


def stop_workers(
    names: Optional[list[str]] = None, *, force: bool = False,
    timeout_s: float = 20.0,
) -> list[dict]:
    """Stop emboviz workers and free their GPUs.

    With no ``names``, stops every running worker; otherwise only those whose
    adapter name is in ``names``. Graceful by default (the worker frees its
    model and exits); ``force`` SIGKILLs. Returns one result dict per worker
    acted on, each with a verified ``action``.
    """
    name_filter = set(names) if names else None
    results: list[dict] = []
    for w in list_running_workers():
        if name_filter is not None and w.name not in name_filter:
            continue
        results.append(_stop_worker(w, force=force, timeout_s=timeout_s))
    return results
