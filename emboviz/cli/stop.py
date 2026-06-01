"""``emboviz stop [NAMES...] [--force]`` — stop running adapter workers.

Each model / dataset-reader / detector runs as a long-lived isolated worker
(its own venv, holding torch and, for the model, the GPU). They stay warm
between ``emboviz analyze`` runs on purpose, so this command is how a user
frees those resources when done.

Graceful by default: the worker is asked to shut itself down over its own
socket — it closes the model (freeing the GPU) and exits. ``--force`` SIGKILLs
a worker that will not stop gracefully.
"""

from __future__ import annotations

import click

from emboviz.adapters.lifecycle import list_running_workers, stop_workers

# Outcomes that mean a worker did NOT cleanly stop on this invocation.
_STRAGGLER_ACTIONS = frozenset({"still-running", "unreachable", "kill-failed", "no-pid"})


@click.command("stop")
@click.argument("names", nargs=-1)
@click.option(
    "--force", is_flag=True, default=False,
    help="SIGKILL workers instead of asking them to shut down gracefully. "
         "Use only for a wedged worker that won't stop on its own.",
)
@click.option(
    "--timeout", "timeout_s", type=float, default=20.0, show_default=True,
    help="Seconds to wait for a graceful shutdown before reporting the "
         "worker as still-running.",
)
def stop_cmd(names: tuple[str, ...], force: bool, timeout_s: float) -> None:
    """Stop emboviz workers and free their GPUs.

    With no NAMES, stops every running worker (model, dataset reader, SAM 3,
    LaMa). Pass one or more adapter names (e.g. ``gr00t sam3``) to stop only
    those. Graceful by default; ``--force`` SIGKILLs.
    """
    running = list_running_workers()
    if not running:
        click.echo("No emboviz workers are running.")
        return

    results = stop_workers(list(names) or None, force=force, timeout_s=timeout_s)
    if not results:
        click.echo(
            f"No running workers matched {list(names)}. "
            f"Running: {sorted({w.name for w in running})}."
        )
        return

    stragglers = 0
    for r in results:
        pid = f"pid {r['pid']}" if r.get("pid") is not None else "no pid"
        line = f"  {r['name']:<18} {r['action']:<16} ({pid})"
        if r.get("detail"):
            line += f"\n      → {r['detail']}"
        click.echo(line)
        if r["action"] in _STRAGGLER_ACTIONS:
            stragglers += 1

    verb = "force-stopped" if force else "stopped"
    ok = len(results) - stragglers
    click.echo(f"{ok}/{len(results)} worker(s) {verb}.")
    if stragglers:
        if not force:
            click.echo("Some workers did not stop gracefully — re-run "
                       "`emboviz stop --force`.")
        raise SystemExit(1)
