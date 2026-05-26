"""CLI entry point for the ``emboviz`` console command.

Subcommands:

  emboviz analyze       — run diagnostics on a model + episode
  emboviz list-models   — show installed model adapters
  emboviz list-datasets — show installed dataset / recording adapters
  emboviz version       — print version + Python info

Each subcommand lives in its own module and is registered here. The
top-level group is intentionally cheap to load (no torch, no transformers)
so ``emboviz --help`` works in any install.
"""

from __future__ import annotations

import click

from emboviz.cli.analyze import analyze_cmd
from emboviz.cli.info import list_datasets_cmd, list_models_cmd, version_cmd


@click.group(
    help=(
        "Emboviz — the X-ray for deployed VLA policies.\n\n"
        "Run diagnostics on your trained model + recorded episodes to find "
        "out WHY your robot did what it did."
    )
)
def main() -> None:
    """Root command group."""


main.add_command(analyze_cmd)
main.add_command(list_models_cmd)
main.add_command(list_datasets_cmd)
main.add_command(version_cmd)


if __name__ == "__main__":  # pragma: no cover
    main()
