"""Internal orchestration code — not part of the public Python API.

These modules implement the orchestration that ``emboviz analyze`` runs
on the user's behalf. They are kept in a ``_internal`` package so users
don't import them directly; the supported interface is the ``emboviz``
console command.
"""
