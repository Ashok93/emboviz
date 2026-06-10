"""emboviz-ctrlworld — Ctrl-World world-model adapter for emboviz.

Wraps the Ctrl-World action-conditioned video world model (Guo et al., ICLR
2026, arXiv:2510.10125) behind the :class:`emboviz_wire.world_model_protocol.
WorldModel` contract: multi-view joint prediction of the three DROID cameras
with pose-anchored sparse-history conditioning, which holds closed-loop
rollouts coherent over tens of seconds.

Import-light by design: the heavy modules (torch, diffusers, the vendored
reference implementation in ``_ctrl_world``) are imported lazily inside the
worker; ``spec`` is the only module emboviz core reads.
"""

from emboviz_ctrlworld.stack_view import (
    STACK_VIEW_ORDER,
    VIEW_HW,
    StackView,
    build_stack_view,
    split_stack_view,
)

__all__ = [
    "STACK_VIEW_ORDER",
    "StackView",
    "VIEW_HW",
    "build_stack_view",
    "split_stack_view",
]
