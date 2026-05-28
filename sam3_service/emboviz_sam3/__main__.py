"""CLI entry point for the SAM 3 sidecar service.

Usage::

    emboviz-sam3 serve [--host 127.0.0.1] [--port 8311] [--preload]

The ``serve`` subcommand starts the uvicorn server bound to localhost
by default. Pass ``--preload`` to fully load SAM 3 before accepting
the first request — useful when invoked from a script that gates
on ``GET /health`` returning ``model_loaded=True``.

The server listens on ``localhost`` by default so we don't accidentally
expose the model to the host network. Bind to ``0.0.0.0`` only if the
client lives on a different host.
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(
        prog="emboviz-sam3",
        description="SAM 3 sidecar service for emboviz",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start the HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8311)
    serve.add_argument(
        "--preload", action="store_true",
        help="Load SAM 3 BEFORE accepting requests (otherwise model loads "
             "lazily on the first /detect call).",
    )
    serve.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error", "critical"],
    )

    args = p.parse_args()

    if args.cmd == "serve":
        _serve(args)
    else:
        p.error(f"unknown command: {args.cmd}")


def _serve(args) -> None:
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    if args.preload:
        # Trigger the load synchronously so /detect doesn't have a
        # 30-second first-request stall and any model-load error is
        # surfaced before uvicorn binds the port.
        from emboviz_sam3.server import STATE
        STATE.ensure_loaded()
    uvicorn.run(
        "emboviz_sam3.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
