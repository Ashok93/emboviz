"""AdapterSpec for the ACT worker.

Provider-driven: the runtime installs ``lerobot`` and lets its metadata
pull the dependency closure (torch, torchvision, the ACT policy code).
ACT ships in base lerobot, so no extra is needed. lerobot 0.5.1 requires
Python >= 3.12 and torch >= 2.7.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="act",
    server_module="emboviz_act.server",
    runtime_pip=(
        "lerobot==0.5.1",
        "emboviz-wire",
        "emboviz-act",
    ),
    default_actor_kwargs={"device": "auto"},
    description="ACT (Action Chunking Transformer) — lerobot ACTPolicy, vision + state, action chunks.",
    requires_python="3.12",
    needs_gpu=True,
)
