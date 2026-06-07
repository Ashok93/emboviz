"""ZeroMQ worker entry-point for the Cosmos 3 world model.

Run from the runtime venv as one of::

    emboviz-cosmos3 serve --sock /tmp/emboviz/cosmos3.sock
    python -m emboviz_cosmos3.server --sock /tmp/emboviz/cosmos3.sock

The worker is a thin client; it does not load a model, so startup is cheap.
It does require a reachable vLLM-Omni Cosmos 3 server — pass its address and
the embodiment via ``--kwargs``, e.g.::

    --kwargs '{"server_url": "http://localhost:8000",
               "domain_name": "agibotworld", "action_dim": 29}'

``domain_name`` and ``action_dim`` are required; the worker refuses to start
without them rather than guess an embodiment.
"""

from __future__ import annotations


def _build_handler(**kwargs):
    from emboviz_wire import WorldModelHandler
    from emboviz_cosmos3.model import Cosmos3WorldModel

    return WorldModelHandler(Cosmos3WorldModel(**kwargs))


def main() -> None:
    from emboviz_wire import serve

    serve(
        _build_handler,
        name="cosmos3",
        default_kwargs={"server_url": "http://localhost:8000"},
    )


if __name__ == "__main__":
    main()
