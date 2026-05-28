"""Compatibility shim — moved to `emboviz_wire.profile`.

The implementation now lives in the standalone `emboviz-wire` package
(the minimal ZMQ connector + shared wire contracts that model worker
venvs install). This module re-exports it so existing
`emboviz.core.profile` imports keep working in the emboviz host venv. New code should
import from `emboviz_wire.profile` directly.
"""
from emboviz_wire.profile import *  # noqa: F401,F403
