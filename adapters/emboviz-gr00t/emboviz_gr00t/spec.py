"""AdapterSpec for NVIDIA GR00T-N1.7.

Provider-driven install: the runtime venv installs NVIDIA's ``gr00t``
package WITH its own declared dependency closure — *gr00t's* pyproject
is the single source of truth for what gr00t needs. We add only emboviz
core + this shim on top. We do NOT hand-mirror gr00t's dependency list:
that mirror drifts the instant NVIDIA adds a dependency (it did —
``tyro`` / ``omegaconf`` / ``dm-tree`` appeared and a hand-typed
``--no-deps`` list silently fell out of sync, killing the worker at
``import gr00t``).

There is exactly one dependency we subtract: ``flash-attn``. NVIDIA hard-
pins it, but (a) its sdist build imports torch before pip has installed
it, so it can't build under build isolation, and (b) the prebuilt wheels
gr00t sources via ``[tool.uv.sources]`` exist only for cp310/cp312 while
this venv is cp311. The adapter runs the Qwen3-VL backbone on SDPA /
eager attention, so flash-attn is never invoked — it is pure dead weight.
``runtime_pip_exclude`` drops it (uv ``--override`` with a false marker)
while the rest of gr00t's deps install normally.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="gr00t",
    server_module="emboviz_gr00t.server",
    # gr00t drives its entire ML stack (torch, transformers, diffusers,
    # peft, tyro, omegaconf, dm-tree, ...) from its own metadata. The
    # lifecycle layer clones the git ref to a local checkout and installs
    # it editable (``-e``) so gr00t's in-repo wheel/path sources resolve;
    # emboviz-wire / emboviz-gr00t are rewritten to ``-e <local>`` in dev.
    runtime_pip=(
        "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git",
        "emboviz-wire",
        "emboviz-gr00t",
    ),
    # The one dep we deliberately drop — see the module docstring.
    runtime_pip_exclude=("flash-attn",),
    default_actor_kwargs={
        "model_path":       "nvidia/GR00T-N1.7-3B",
        "embodiment_tag":   "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        "device":           "cuda",
    },
    description="NVIDIA GR00T-N1.7 (3B). Qwen3-VL backbone + diffusion action expert.",
    requires_python="3.11",
    needs_gpu=True,
)
