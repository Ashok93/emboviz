# emboviz-wire

The minimal ZeroMQ connector + shared wire contracts for emboviz.

This is the **only** emboviz package a model worker venv installs. It carries
the ZMQ ROUTER/serve loop, the msgpack codec, the `AdapterSpec`/handler
contract, and the `Scene`/`Observations`/`ActionResult`/`VLAModel` types that
cross the socket — and nothing else (no datasets, diagnostics, rerun, viz).

Its dependency floor is deliberately tiny and loosely pinned so it never
fights a model's own numpy/pyzmq/pydantic versions: the worker and the
emboviz host are separate processes that talk only through standardized
bytes (msgpack over ZMQ), which are version-stable across both sides.
