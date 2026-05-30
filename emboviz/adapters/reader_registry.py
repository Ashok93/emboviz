"""Discover installed dataset-reader packages via entry points.

The dataset-side analogue of :mod:`emboviz.adapters.registry`. A user's
main venv lists which reader packages they've installed in
``importlib.metadata.entry_points(group="emboviz.readers")``. Each entry
resolves to an :class:`AdapterSpec` declared by that reader package
(``emboviz-lerobot`` ships one). Core never imports the reader's heavy
code — only the small spec module.

Readers and model adapters share the :class:`AdapterSpec` shape and the
same venv / spawn machinery; they differ only in the entry-point group
(``emboviz.readers`` vs ``emboviz.adapters``) and the wire methods their
worker exposes (the EpisodeSource contract vs the VLAModel contract).
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata

from emboviz.adapters.protocol import AdapterSpec


_GROUP = "emboviz.readers"


def list_readers() -> dict[str, AdapterSpec]:
    """Return every installed dataset reader's spec, keyed by format name."""
    specs: dict[str, AdapterSpec] = {}
    try:
        eps = importlib_metadata.entry_points(group=_GROUP)
    except TypeError:
        eps = importlib_metadata.entry_points().get(_GROUP, [])  # type: ignore[assignment]
    for ep in eps:
        spec = ep.load()
        if not isinstance(spec, AdapterSpec):
            raise TypeError(
                f"emboviz.readers entry point {ep.name!r} -> {ep.value!r} "
                f"resolved to {type(spec).__name__}, expected AdapterSpec. "
                "Reader packages must export an AdapterSpec instance."
            )
        if spec.name != ep.name:
            raise ValueError(
                f"AdapterSpec.name {spec.name!r} must match the entry-"
                f"point key {ep.name!r} (declared in {ep.value!r}'s "
                "pyproject.toml)."
            )
        specs[spec.name] = spec
    return specs


def find_reader(name: str) -> AdapterSpec:
    """Look up one dataset-reader spec by format name (e.g. ``"lerobot"``).

    Raises a friendly error listing what IS installed if the requested
    reader isn't registered.
    """
    specs = list_readers()
    if name in specs:
        return specs[name]

    installed = sorted(specs)
    pkg = f"emboviz-{name}"
    if installed:
        hint = (
            f"Installed readers: {', '.join(installed)}. If you meant "
            f"'{name}', install it with `uv pip install {pkg}` and then "
            f"`emboviz install-{name}`."
        )
    else:
        hint = (
            "No dataset readers are installed in this environment. To read "
            f"'{name}' datasets, run `uv pip install {pkg}` and then "
            f"`emboviz install-{name}` (creates the isolated reader venv)."
        )
    raise KeyError(f"Unknown dataset reader '{name}'. {hint}")
