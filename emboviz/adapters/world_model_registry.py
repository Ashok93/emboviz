"""Discover installed world-model packages via entry points.

The third sibling of :mod:`emboviz.adapters.registry` (policies) and
:mod:`emboviz.adapters.reader_registry` (datasets). A user's main venv lists
which world-model packages they've installed in
``importlib.metadata.entry_points(group="emboviz.world_models")``. Each entry
resolves to an :class:`AdapterSpec` declared by that package
(``emboviz-cosmos3`` ships one). Core never imports the world model's heavy
code — only the small spec module.

World models, policies, and readers share the :class:`AdapterSpec` shape and
the same venv / spawn machinery; they differ only in the entry-point group
(``emboviz.world_models`` vs ``emboviz.adapters`` vs ``emboviz.readers``) and
the wire methods their worker exposes (the WorldModel contract vs the VLAModel
vs the EpisodeSource contract).
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata

from emboviz.adapters.protocol import AdapterSpec


_GROUP = "emboviz.world_models"


def list_world_models() -> dict[str, AdapterSpec]:
    """Return every installed world model's spec, keyed by CLI name."""
    specs: dict[str, AdapterSpec] = {}
    try:
        eps = importlib_metadata.entry_points(group=_GROUP)
    except TypeError:
        eps = importlib_metadata.entry_points().get(_GROUP, [])  # type: ignore[assignment]
    for ep in eps:
        spec = ep.load()
        if not isinstance(spec, AdapterSpec):
            raise TypeError(
                f"emboviz.world_models entry point {ep.name!r} -> {ep.value!r} "
                f"resolved to {type(spec).__name__}, expected AdapterSpec. "
                "World-model packages must export an AdapterSpec instance."
            )
        if spec.name != ep.name:
            raise ValueError(
                f"AdapterSpec.name {spec.name!r} must match the entry-"
                f"point key {ep.name!r} (declared in {ep.value!r}'s "
                "pyproject.toml)."
            )
        specs[spec.name] = spec
    return specs


def find_world_model(name: str) -> AdapterSpec:
    """Look up one world-model spec by name (e.g. ``"cosmos3"``).

    Raises a friendly error listing what IS installed if the requested
    world model isn't registered.
    """
    specs = list_world_models()
    if name in specs:
        return specs[name]

    installed = sorted(specs)
    pkg = f"emboviz-{name}"
    if installed:
        hint = (
            f"Installed world models: {', '.join(installed)}. If you meant "
            f"'{name}', install it with `uv pip install {pkg}` and then "
            f"`emboviz install-{name}`."
        )
    else:
        hint = (
            "No world models are installed in this environment. To use "
            f"'{name}', run `uv pip install {pkg}` and then "
            f"`emboviz install-{name}` (creates the isolated worker venv)."
        )
    raise KeyError(f"Unknown world model '{name}'. {hint}")
