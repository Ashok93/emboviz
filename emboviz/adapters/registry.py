"""Discover installed adapter packages via entry points.

A user's main venv lists which adapter packages they've installed in
``importlib.metadata.entry_points(group="emboviz.adapters")``. Each
entry resolves to an :class:`AdapterSpec` declared by that adapter
package. Core never imports the adapter's model code — only the
small spec module.
"""

from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata
from typing import Optional

from emboviz.adapters.protocol import AdapterSpec


_GROUP = "emboviz.adapters"


def list_adapters() -> dict[str, AdapterSpec]:
    """Return every installed adapter's spec, keyed by CLI alias.

    Each adapter package in the user's main venv contributes one entry
    via its ``[project.entry-points."emboviz.adapters"]`` table. The
    entry value is ``"<module>:<attr>"`` where ``<attr>`` is the
    :class:`AdapterSpec` instance.
    """
    specs: dict[str, AdapterSpec] = {}
    try:
        eps = importlib_metadata.entry_points(group=_GROUP)
    except TypeError:
        # importlib.metadata <3.10 returned a dict
        eps = importlib_metadata.entry_points().get(_GROUP, [])  # type: ignore[assignment]
    for ep in eps:
        spec = ep.load()
        if not isinstance(spec, AdapterSpec):
            raise TypeError(
                f"emboviz.adapters entry point {ep.name!r} -> {ep.value!r} "
                f"resolved to {type(spec).__name__}, expected AdapterSpec. "
                "Adapter packages must export an AdapterSpec instance."
            )
        if spec.name != ep.name:
            raise ValueError(
                f"AdapterSpec.name {spec.name!r} must match the entry-"
                f"point key {ep.name!r} (declared in "
                f"{ep.value!r}'s pyproject.toml)."
            )
        specs[spec.name] = spec
    return specs


def find_adapter(name: str) -> AdapterSpec:
    """Look up one adapter spec by CLI alias.

    Raises a friendly error listing what IS installed if the requested
    alias isn't registered. Diagnoses both the "you forgot to install
    the adapter package" and the "you typo'd the model name" cases.
    """
    specs = list_adapters()
    if name in specs:
        return specs[name]

    installed = sorted(specs)
    pkg = f"emboviz-{name}"
    if installed:
        installed_str = ", ".join(installed)
        hint = (
            f"Installed adapters: {installed_str}. "
            f"If you meant '{name}', install it with "
            f"`uv pip install {pkg}` and then `emboviz install-{name}`."
        )
    else:
        hint = (
            "No adapters are installed in this environment. "
            f"To use '{name}', run `uv pip install {pkg}` and then "
            f"`emboviz install-{name}` (creates the isolated runtime "
            "venv with the heavy model deps)."
        )
    raise KeyError(f"Unknown adapter '{name}'. {hint}")
