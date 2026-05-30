"""Compatibility shim — moved to ``emboviz_wire.reader_protocol``.

The dataset-reader contract (:class:`EpisodeSource`) now lives in the
standalone ``emboviz-wire`` package, alongside the model-side contract
(:class:`emboviz_wire.model_protocol.VLAModel`) — so an isolated reader
worker (which has the wire package but not emboviz core) implements the
same interface the host consumes. Re-exported here so existing
``emboviz.datasets.base`` imports keep working in the host venv. New
code should import from ``emboviz_wire.reader_protocol`` directly.
"""

from emboviz_wire.reader_protocol import EpisodeSource  # noqa: F401

__all__ = ["EpisodeSource"]
