"""Compatibility shim — moved to `emboviz_wire.model_protocol`.

The implementation now lives in the standalone `emboviz-wire` package
(the minimal ZMQ connector + shared wire contracts that model worker
venvs install). This module re-exports it so existing
`emboviz.models.protocol` imports keep working in the emboviz host venv. New code should
import from `emboviz_wire.model_protocol` directly.
"""
from emboviz_wire.model_protocol import *  # noqa: F401,F403
