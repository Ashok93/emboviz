"""Worker process lifecycle helpers — install, locate, optional spawn.

The architecture is **Pattern Y**: each adapter runs as an
independent long-lived ZMQ worker. Production / cloud deployments
manage these workers externally (systemd unit, docker compose,
Kubernetes Deployment) and core just connects to the known endpoint.
For local-development convenience we also support **opportunistic
auto-spawn**: if the user invokes ``emboviz analyze --model openvla``
and no worker is already running, we ``subprocess.Popen`` the
adapter's ``server`` entry-point in its runtime venv and wait until
it answers ``ping``. The spawned worker stays running between CLI
invocations, so the model only cold-loads once per session.

This module is the **only** place that knows about subprocesses,
venv paths, and PID files. Everything else (client.py, the
diagnostics, CLI commands) just sees endpoints.
"""

from __future__ import annotations

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


def _rewrite_pip_for_dev(runtime_pip: tuple[str, ...]) -> list[str]:
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
        emboviz analyze --model <name>    # connect

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

    if spec.runtime_pip_no_deps:
        no_deps_requirements = _rewrite_pip_for_dev(spec.runtime_pip_no_deps)
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


def _spawn_worker(spec: AdapterSpec, endpoint: str) -> subprocess.Popen:
    """Start the adapter's worker process in its isolated runtime venv.

    Tries the ``[project.scripts]`` console entry-point first (e.g.
    ``emboviz-openvla`` on the venv's PATH). Falls back to ``python -m
    <server_module>``. The output is appended to
    ``~/.emboviz/logs/<name>.log`` so users can ``tail -f`` it.
    """
    venv_bin = venv_python(spec.name).parent

    console = venv_bin / spec.console_script
    if console.exists():
        cmd = [str(console)]
    else:
        cmd = [str(venv_python(spec.name)), "-m", spec.server_module]

    # ipc:// endpoints carry a path we hand to --sock; tcp:// endpoints
    # take --tcp host:port.
    if endpoint.startswith("ipc://"):
        cmd += ["--sock", endpoint[len("ipc://"):]]
    elif endpoint.startswith("tcp://"):
        cmd += ["--tcp", endpoint[len("tcp://"):]]
    else:
        raise ValueError(f"unsupported endpoint scheme: {endpoint!r}")

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


def connect(
    name: str,
    *,
    auto_spawn: bool = True,
    timeout_s: int = 600,
) -> WorkerHandle:
    """Return a :class:`WorkerHandle` for the named adapter.

    If a worker is already responding on the resolved endpoint (because
    the user started it in another shell or a previous CLI run left it
    running), we reuse it. Otherwise, if ``auto_spawn`` is True, we
    ``subprocess.Popen`` the adapter's ``server`` entry-point in its
    runtime venv and wait until it answers ``ping``. If ``auto_spawn``
    is False, we raise with a friendly remediation.

    ``timeout_s`` bounds how long we wait for cold-load on first spawn —
    larger models (π0 with its Triton autotune cache cold) can take
    minutes.
    """
    spec = find_adapter(name)
    endpoint = default_endpoint(name)

    if _is_alive(endpoint):
        client = ZMQAdapterClient(name=name, endpoint=endpoint)
        return WorkerHandle(name=name, endpoint=endpoint, client=client, spawned=False)

    if not auto_spawn:
        raise RuntimeError(
            f"no worker reachable at {endpoint} for adapter '{name}'. "
            f"Start one with:\n    emboviz-{name} serve\n"
            "or pass --auto-spawn (default) to let emboviz launch one."
        )

    proc = _spawn_worker(spec, endpoint)
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
        name=name, endpoint=endpoint, client=client, process=proc, spawned=True
    )


def shutdown(handle: WorkerHandle, *, terminate: bool = False) -> None:
    """Compatibility shim: close the client (and optionally the worker)."""
    handle.close(terminate=terminate)
