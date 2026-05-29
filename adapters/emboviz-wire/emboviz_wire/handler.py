"""AdapterSpec — the contract every adapter package declares.

Each ``emboviz-<name>`` package ships ONE :class:`AdapterSpec` and
registers it via ``[project.entry-points."emboviz.adapters"]`` in its
``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."emboviz.adapters"]
    openvla = "emboviz_openvla.spec:SPEC"

The spec carries everything emboviz core needs to:

  • Build the isolated runtime venv on first ``emboviz install-<name>``.
  • Know which command launches the ZeroMQ worker for that adapter.
  • Pass the right env vars into the worker process.

It deliberately keeps the *adapter source code* out of core's import
path — core only imports a small spec module from the adapter package
(``emboviz_openvla.spec``), never the model code that pulls in torch
and transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdapterSpec:
    """Static metadata for one VLA family.

    Attributes
    ----------
    name
        CLI alias the user types — ``"openvla"``, ``"pi0"`` etc. Must
        match the entry-point key.
    server_module
        Python ``-m`` target that launches the adapter's ZMQ worker
        inside the runtime venv (e.g. ``"emboviz_openvla.server"``).
        Invoked as ``<runtime_venv_python> -m <server_module> --sock
        <path>``. The adapter package's ``[project.scripts]`` may
        additionally expose a console entry-point named ``emboviz-<name>
        = "emboviz_<name>.server:main"`` so the user can also start it
        from the runtime venv's PATH.
    runtime_pip
        The pip requirement specs that must exist in the runtime venv,
        resolved by ``install-<name>`` at first use. Always ends with
        ``emboviz-wire`` + the adapter shim (rewritten to ``-e <local>``
        in dev). The model deps follow ONE of two legitimate shapes —
        never a third:

          • Provider-driven — list the upstream package (``<pkg> @
            git+…`` or a PyPI name) and let ITS metadata pull the
            dependency closure. The provider's pyproject is the single
            source of truth; we add no version pins beyond pod-
            compatibility constraints. Used by oft / pi0 / gr00t /
            lerobot. Subtract un-takeable deps via ``runtime_pip_exclude``.

          • Runtime-spec — for models that ship as HF-hub *remote code*
            (loaded through ``trust_remote_code``) there is NO installable
            provider package, so list the inference runtime the hub code
            needs (torch + the right transformers window + vision libs).
            Used by openvla / sam3.

        What is FORBIDDEN is the third shape: hand-copying an installable
        package's transitive dependency list into ``runtime_pip``. That
        mirror drifts the instant the provider changes its deps and
        silently breaks the worker at import (this is exactly how gr00t's
        old ``--no-deps`` mirror lost ``tyro``). If the provider ships an
        installable package, install the package — don't retype its deps.
    runtime_env_vars
        Process-environment variables that must be set in the worker
        process — e.g. ``GIT_LFS_SKIP_SMUDGE=1`` for π0's openpi
        install. These are forwarded to the spawn AND to the install
        ``uv pip install`` subprocess (so transitive git+ deps build).
    default_actor_kwargs
        Keyword arguments forwarded to the underlying ``VLAModel``
        constructor on first server start when the user hasn't passed
        overrides on the command line. Used for HF model_id, dtype,
        attention impl, etc.
    description
        One-line human-facing summary used by ``emboviz list-adapters``.
    requires_python
        Python version constraint for the runtime venv (PEP 508-ish,
        passed to ``uv venv --python``). e.g. ``"3.11"`` for OpenVLA,
        ``"3.12"`` for SAM 3. Adapter venvs are independently versioned
        because the wire protocol (ZMQ + msgpack) is bytes — no
        cross-Python-version pickle constraint.
    needs_gpu
        If True, ``emboviz install-<name>`` will warn if no CUDA device
        is visible. Does not block — some users only test on CPU.
    runtime_pip_exclude
        Packages the provider *declares* as dependencies that we
        deliberately do NOT install. Passed to ``uv pip install`` as a
        ``--override`` whose entries carry an always-false environment
        marker (``sys_platform == 'never'``) — the documented uv idiom
        for dropping a dependency from resolution (an override only takes
        effect if the package is actually requested, and the false marker
        makes it unsatisfiable on every real platform).

        This is the escape hatch that lets an adapter stay *provider-
        driven*: install the upstream package WITH its own declared
        dependency closure (its pyproject is the single source of truth)
        and subtract only the few packages we can't or won't take. The
        sole user today is ``gr00t``: NVIDIA hard-pins ``flash-attn``,
        whose sdist build imports torch before pip has installed it
        (unbuildable under isolation) and whose prebuilt wheels exist
        only for cp310/cp312 (our venv is cp311). The adapter runs on
        SDPA / eager attention, so flash-attn is never invoked — we drop
        it and let the rest of gr00t's deps resolve normally, instead of
        hand-mirroring the whole list (which silently drifts the moment
        the provider adds a dependency).
    """

    name: str
    server_module: str
    runtime_pip: tuple[str, ...]
    description: str = ""
    runtime_env_vars: dict[str, str] = field(default_factory=dict)
    default_actor_kwargs: dict = field(default_factory=dict)
    requires_python: str = "3.11"
    needs_gpu: bool = True
    runtime_pip_exclude: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.server_module or "." not in self.server_module:
            raise ValueError(
                f"AdapterSpec({self.name!r}): server_module must be a "
                f"dotted module path, got {self.server_module!r}."
            )
        if not self.runtime_pip:
            raise ValueError(
                f"AdapterSpec({self.name!r}): runtime_pip must list at "
                "least one package — the adapter itself."
            )

    @property
    def console_script(self) -> str:
        """Conventional name of the ``[project.scripts]`` console entry
        the adapter ships (``emboviz-<name>``). The :func:`lifecycle.
        spawn_if_needed` helper uses this when it can find the console
        script on the runtime venv's PATH; otherwise it falls back to
        ``python -m <server_module>``."""
        return f"emboviz-{self.name}"
