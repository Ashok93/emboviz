"""emboviz-ctrlworld — Ctrl-World world-model adapter for emboviz.

Wraps the Ctrl-World action-conditioned video world model (Guo et al., ICLR
2026, arXiv:2510.10125) behind the :class:`emboviz_wire.world_model_protocol.
WorldModel` contract: multi-view joint prediction with pose-anchored
sparse-history conditioning, which holds closed-loop rollouts coherent over
tens of seconds.

A checkpoint's contract — views, sizes, rates, history schedule, action
bounds, weight locations — is a :class:`emboviz_ctrlworld.profiles.
CtrlWorldProfile`: ``"droid"`` ships (the released DROID checkpoint); a
fine-tune on another rig is a profile JSON, not a code change.

Import-light by design: the heavy modules (torch, diffusers, the vendored
reference implementation in ``_ctrl_world``) are imported lazily inside the
worker; ``spec`` is the only module emboviz core reads at discovery time.
"""

from emboviz_ctrlworld.profiles import (
    ACTION_DIM,
    CtrlWorldProfile,
    check_stress_compat,
    get_profile,
    load_profile,
    resolve_profile,
)
from emboviz_ctrlworld.stack_view import build_stack_view, split_stack_view

__all__ = [
    "ACTION_DIM",
    "CtrlWorldProfile",
    "build_stack_view",
    "check_stress_compat",
    "get_profile",
    "load_profile",
    "resolve_profile",
    "split_stack_view",
]
