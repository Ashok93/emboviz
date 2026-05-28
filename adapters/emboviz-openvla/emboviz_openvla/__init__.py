"""emboviz-openvla — OpenVLA-7B adapter for emboviz.

Importing this package is cheap: no torch, no transformers, no model
load. The heavy machinery is materialised inside the isolated runtime
venv (``~/.emboviz/venvs/openvla``) when ``emboviz analyze --model
openvla ...`` spawns the Ray actor.

The entry point ``emboviz.adapters:openvla`` resolves to
:data:`emboviz_openvla.spec.SPEC`; emboviz core uses that to look up
where the actor lives, what to install, and what env vars it needs.
"""

__version__ = "0.3.0"
