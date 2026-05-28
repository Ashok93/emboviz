"""Compatibility shim — moved to `emboviz_wire.client`.

The ZMQ client end of the wire (RpcClient / ZMQAdapterClient) now lives
in the standalone `emboviz-wire` package. Re-exported so existing
`emboviz.adapters.client` imports keep working in the host venv.
"""
from emboviz_wire.client import *  # noqa: F401,F403
from emboviz_wire.client import (  # explicit (names used by emboviz.adapters.__init__)
    AdapterRpcError,
    RpcClient,
    ZMQAdapterClient,
    default_endpoint,
)
