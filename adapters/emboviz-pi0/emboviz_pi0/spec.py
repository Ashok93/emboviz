"""AdapterSpec for Physical Intelligence π0 / π0.5 via openpi.

π0's install has TWO upstream-imposed quirks we have to honor:

  1. ``GIT_LFS_SKIP_SMUDGE=1`` is REQUIRED when installing openpi —
     it pins an old lerobot commit whose git-lfs test fixtures are no
     longer fetchable, and that env var is openpi's own documented
     workaround. We forward it via ``runtime_env_vars`` so install_venv's
     ``uv pip install`` subprocess sees it.

  2. ``openpi`` is git-only (not on PyPI). It comes in as a direct
     PEP 508 reference inside ``runtime_pip``.

The PyTorch backend (used for attention extraction) needs a converted
checkpoint produced by ``emboviz convert-pi0 <config>``. That's a
one-off the user runs after install — kept as a separate command so
people who only want JAX inference don't pay the convert cost.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="pi0",
    server_module="emboviz_pi0.server",
    # PROVIDER-DRIVEN adapter (cf. AdapterSpec.runtime_pip): ``openpi``
    # drives its own dependency closure. The few explicit pins below are
    # not a mirror of openpi's deps — they are pod-compatibility
    # CONSTRAINTS layered on top (cap torch <2.10 off the cu13-only wheel;
    # hold transformers in openpi's gemma window) that narrow, not
    # restate, what openpi resolves.
    runtime_pip=(
        # openpi supports Python 3.11 and 3.12; we pin the venv to 3.11
        # so the RLDS tensorflow-cpu wheels (3.11-only on PyPI) install
        # alongside it without complaint. See openpi pyproject.toml.
        "torch>=2.2,<2.10",
        # openpi's gemma wrapper targets the 4.53.x series.
        "transformers>=4.50,<4.55",
        "einops>=0.8",
        "safetensors>=0.5",
        # openpi itself — direct git ref. The install_venv subprocess
        # carries GIT_LFS_SKIP_SMUDGE=1 in env so the transitive
        # lerobot checkout doesn't try to smudge missing fixtures.
        "openpi @ git+https://github.com/Physical-Intelligence/openpi.git",
        "emboviz-wire",
        "emboviz-pi0",
    ),
    runtime_env_vars={"GIT_LFS_SKIP_SMUDGE": "1"},
    # MUST match Pi0Adapter.__init__ exactly — these are now forwarded to
    # the worker at spawn (lifecycle passes default_actor_kwargs as
    # --kwargs). The old {checkpoint, device} keys were NOT real params
    # (the constructor takes ``checkpoint_uri``, and has no ``device``
    # arg) and would crash the worker. config_name selects checkpoint +
    # platform; checkpoint_uri stays None → openpi's published GCS
    # checkpoint for the config. A user's fine-tune overrides via the run
    # config's model.kwargs.checkpoint_uri.
    default_actor_kwargs={
        "config_name": "pi0_libero",
    },
    description="Physical Intelligence π0 / π0.5 (Pi family) via openpi.",
    requires_python="3.11",
    needs_gpu=True,
)
