"""Compatibility shim — moved to `emboviz_wire.distances`.

The implementation now lives in the standalone `emboviz-wire` package
(the minimal ZMQ connector + shared wire contracts that model worker
venvs install). This module re-exports it so existing
`emboviz.core.distances` imports keep working in the emboviz host venv. New code should
import from `emboviz_wire.distances` directly.
"""
from emboviz_wire.distances import *  # noqa: F401,F403
