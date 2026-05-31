"""AdapterSpec for the SmolVLA worker.

Provider-driven: the runtime installs ``lerobot[smolvla]`` (the extra adds
transformers, num2words, accelerate) and lets its metadata pull the
dependency closure. lerobot 0.5.2 requires Python >= 3.12 and torch >= 2.7.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="smolvla",
    server_module="emboviz_smolvla.server",
    runtime_pip=(
        "lerobot[smolvla]==0.5.2",
        "emboviz-wire",
        "emboviz-smolvla",
    ),
    default_actor_kwargs={"device": "auto"},
    description="SmolVLA — lerobot SmolVLAPolicy, vision + language + state, flow-matching action chunks.",
    requires_python="3.12",
    needs_gpu=True,
)
