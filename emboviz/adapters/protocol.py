"""AdapterSpec — the contract every adapter package declares.

Each ``emboviz-<name>`` package ships ONE :class:`AdapterSpec` and
registers it via ``[project.entry-points."emboviz.adapters"]`` in its
``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."emboviz.adapters"]
    openvla = "emboviz_openvla.spec:SPEC"

The spec carries everything emboviz core needs to:

  • Build the isolated runtime venv on first ``emboviz install-<name>``.
  • Locate that venv's Python and hand it to Ray as ``py_executable``.
  • Look up the actor class to instantiate inside that venv.

It deliberately keeps the *adapter source code* out of core's import
path — core only imports a small spec module from the adapter package
(``emboviz_openvla.spec``), never the model code that pulls in torch
and transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AdapterSpec:
    """Static metadata for one VLA family.

    Attributes
    ----------
    name
        CLI alias the user types — ``"openvla"``, ``"pi0"`` etc. Must
        match the entry-point key.
    actor_import_path
        ``"<module>:<class>"`` of the Ray actor class living in the
        adapter package, e.g. ``"emboviz_openvla.actor:OpenVLAActor"``.
        This is imported INSIDE the runtime venv, never in core.
    runtime_pip
        The pip requirement specs that must exist in the runtime venv.
        The ``install-<name>`` CLI subcommand resolves these into the
        venv at first use. Include the adapter package itself last so
        editable installs work cleanly during dev.
    runtime_env_vars
        Process-environment variables that must be set in the actor
        process — e.g. ``GIT_LFS_SKIP_SMUDGE=1`` for π0's openpi
        install. These are forwarded to ``ray.actor.options(
        runtime_env={"env_vars": ...})``.
    default_actor_kwargs
        Keyword arguments the lifecycle layer passes to the actor's
        ``__init__`` if the user didn't supply overrides via CLI. Used
        for HF model_id, dtype, attention impl, etc.
    description
        One-line human-facing summary used by ``emboviz list-adapters``.
    requires_python
        Python version constraint for the runtime venv (PEP 508-ish,
        passed to ``uv venv --python``). e.g. ``"3.10"`` for OpenVLA,
        ``"3.12"`` for SAM 3.
    needs_gpu
        If True, ``emboviz install-<name>`` will warn if no CUDA device
        is visible. Does not block — some users only test on CPU.
    """

    name: str
    actor_import_path: str
    runtime_pip: tuple[str, ...]
    description: str = ""
    runtime_env_vars: dict[str, str] = field(default_factory=dict)
    default_actor_kwargs: dict = field(default_factory=dict)
    requires_python: str = "3.10"
    needs_gpu: bool = True

    def __post_init__(self):
        if ":" not in self.actor_import_path:
            raise ValueError(
                f"AdapterSpec({self.name!r}): actor_import_path must be "
                f"'<module>:<class>', got {self.actor_import_path!r}."
            )
        if not self.runtime_pip:
            raise ValueError(
                f"AdapterSpec({self.name!r}): runtime_pip must list at "
                "least one package — the adapter itself."
            )
