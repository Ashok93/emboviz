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

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from emboviz.adapters.client import ZMQAdapterClient, default_endpoint
from emboviz.adapters.protocol import AdapterSpec
from emboviz.adapters.registry import find_adapter


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


def pid_file(name: str) -> Path:
    p = Path.home() / ".emboviz" / "run"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.pid"


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
    requirements = _rewrite_pip_for_dev(spec.runtime_pip)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(py), *requirements],
        check=True,
        env=env,
    )

    # Second pass: any --no-deps installs. For each requirement that
    # looks like ``<name> @ git+https://<host>/<owner>/<repo>...``, we
    # clone the repo to ``~/.emboviz/repos/<repo>`` first and replace
    # the requirement with ``-e <clone_path>``. This works around a uv
    # constraint: uv strictly parses transitive git deps, and NVIDIA's
    # Isaac-GR00T pyproject pins ``torchcodec`` via a ``file://...wheel``
    # URL inside its own repo. When we install via ``-e <clone>`` that
    # file:// ref is just a local-to-local path which uv accepts; when
    # we install via ``git+`` it's a transitive git ref pointing at a
    # file, which uv refuses. Cloning once gives every adapter a stable
    # checkout under ``~/.emboviz/repos/`` we can also reuse for the
    # adapter's ``demo_data`` etc.
    if spec.runtime_pip_no_deps:
        no_deps_requirements = _materialise_git_clones(spec.runtime_pip_no_deps)
        no_deps_requirements = _rewrite_pip_for_dev(no_deps_requirements)
        subprocess.run(
            [
                "uv", "pip", "install", "--python", str(py),
                "--no-deps", *no_deps_requirements,
            ],
            check=True,
            env=env,
        )

    return path


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
    client: ZMQAdapterClient
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
    # Best-effort PID file so external tools (and future
    # ``emboviz status`` / ``emboviz stop``) can find the worker.
    try:
        pid_file(spec.name).write_text(str(proc.pid))
    except OSError:
        pass
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
        }.get(spec.name, "")
        print(
            f"[emboviz] first run for '{spec.name}' — materialising the "
            f"runtime venv at {venv_path(spec.name)} "
            f"{('(' + py_size_hint + ') ') if py_size_hint else ''}...",
            file=sys.stderr, flush=True,
        )
    install_venv(spec, force=False)
    return venv_path(spec.name)


def connect(
    name: str,
    *,
    actor_kwargs: Optional[dict] = None,
    auto_spawn: bool = True,
    auto_install: bool = True,
    timeout_s: int = 600,
) -> WorkerHandle:
    """Return a live :class:`WorkerHandle` for the named adapter.

    Three states the caller can be in:

      1. Worker already running and responsive at the resolved endpoint
         (because a previous CLI invocation left it warm, or the user
         started it manually) — we attach and return.
      2. Worker not running but the adapter's runtime venv exists —
         spawn the worker via ``subprocess.Popen`` and wait for it to
         answer ``ping``.
      3. Runtime venv doesn't exist yet — ``auto_install`` (default
         True) creates it via :func:`install_venv`, then proceeds to
         state 2. Visible progress is printed to stderr.

    Pass ``auto_install=False`` or ``auto_spawn=False`` to require the
    user to have already run ``emboviz install-<name>`` /
    ``emboviz-<name> serve`` themselves.

    ``timeout_s`` bounds how long we wait for cold-load on first spawn —
    larger models (π0 with its Triton autotune cache cold) can take
    minutes.
    """
    spec = find_adapter(name)
    endpoint = default_endpoint(name)

    # ── 1. Already alive ────────────────────────────────────────────
    if _is_alive(endpoint):
        # A warm worker carries whatever construction kwargs it was
        # spawned with. If the caller is requesting specific per-run
        # kwargs (e.g. their own --model-kwargs checkpoint), we must NOT
        # silently attach to a worker that may hold a different model —
        # that would diagnose the wrong checkpoint. Refuse with a clear
        # remediation. (No actor_kwargs → attaching to the warm worker
        # is exactly the intended fast path.)
        if actor_kwargs:
            raise RuntimeError(
                f"a '{name}' worker is already running at {endpoint}, "
                f"loaded with its own configuration. Refusing to route "
                f"this run's model-kwargs {actor_kwargs!r} to it — it may "
                f"hold a different checkpoint, and sending your data to "
                f"the wrong model would be a silent-wrong-answer bug. "
                f"Stop the running worker first (kill the "
                f"`emboviz-{name} serve` process, or remove "
                f"{endpoint[len('ipc://'):] if endpoint.startswith('ipc://') else endpoint}), "
                f"then re-run so a fresh worker loads with your kwargs."
            )
        client = ZMQAdapterClient(name=name, endpoint=endpoint)
        return WorkerHandle(
            name=name, endpoint=endpoint, client=client, spawned=False,
        )

    # ── 2. Ensure the runtime venv exists ───────────────────────────
    if not _runtime_venv_ready(name):
        if not auto_install:
            raise RuntimeError(
                f"runtime venv for adapter '{name}' is missing at "
                f"{venv_path(name)}. Run `emboviz install-{name}` to "
                "create it, or pass --auto-install (default) to let "
                "emboviz set it up automatically."
            )
        _ensure_runtime_venv(spec)

    # ── 3. Spawn the worker if not already running ──────────────────
    if not auto_spawn:
        raise RuntimeError(
            f"no worker reachable at {endpoint} for adapter '{name}'. "
            f"Start one with:\n    emboviz-{name} serve\n"
            "or pass --auto-spawn (default) to let emboviz launch one."
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

    client = ZMQAdapterClient(name=name, endpoint=endpoint)
    return WorkerHandle(
        name=name, endpoint=endpoint, client=client, process=proc, spawned=True,
    )


def shutdown(handle: WorkerHandle, *, terminate: bool = False) -> None:
    """Compatibility shim: close the client (and optionally the worker)."""
    handle.close(terminate=terminate)
