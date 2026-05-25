"""Foxglove `.mcap` exporter.

Same data as the Rerun export, written as mcap topics that Foxglove
Studio can subscribe to. Lets teams running Foxglove for rollout
playback see Emboviz overlays without learning a new tool.

Topics emitted:
  • /emboviz/cameras/<cam>      — JPEG-encoded images (CompressedImage)
  • /emboviz/diagnostics/<axis>/score — scalar Float32 message
  • /emboviz/diagnostics/<axis>/severity — string message
  • /emboviz/predictions/action  — Float32MultiArray with dim names in metadata

Lazy imports `mcap`. Install with: uv add mcap
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Trajectory
from emboviz.diagnostics.trajectory import TrajectoryDiagnosticResult


def export_foxglove(
    trajectory: Trajectory,
    per_axis_results: dict[str, TrajectoryDiagnosticResult],
    out_path: Path,
) -> Path:
    """Emit an .mcap file with Foxglove-compatible diagnostic topics."""
    try:
        from mcap.writer import Writer
    except ImportError as e:
        raise ImportError(
            "Foxglove export requires the `mcap` package. "
            "Install with: uv add mcap"
        ) from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps = trajectory.fps if trajectory.fps > 0 else 5.0

    with open(out_path, "wb") as f:
        writer = Writer(f)
        writer.start()

        # Schemas (JSON-encoded for Foxglove Studio's auto-schema detection)
        image_schema = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {
                    "timestamp": {"type": "object",
                                  "properties": {"sec": {"type": "integer"},
                                                 "nsec": {"type": "integer"}}},
                    "frame_id": {"type": "string"},
                    "data": {"type": "string", "contentEncoding": "base64"},
                    "format": {"type": "string"},
                },
            }).encode("utf-8"),
        )
        scalar_schema = writer.register_schema(
            name="emboviz.Scalar",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {"value": {"type": "number"}},
            }).encode("utf-8"),
        )
        string_schema = writer.register_schema(
            name="emboviz.String",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }).encode("utf-8"),
        )

        # Channels — one per (topic, modality)
        camera_channels: dict[str, int] = {}
        if trajectory.frames:
            for cam_name in trajectory.frames[0].observations.images.keys():
                camera_channels[cam_name] = writer.register_channel(
                    topic=f"/emboviz/cameras/{cam_name}",
                    message_encoding="json",
                    schema_id=image_schema,
                )

        diag_score_channels: dict[str, int] = {}
        diag_severity_channels: dict[str, int] = {}
        for axis in per_axis_results.keys():
            diag_score_channels[axis] = writer.register_channel(
                topic=f"/emboviz/diagnostics/{axis}/score",
                message_encoding="json",
                schema_id=scalar_schema,
            )
            diag_severity_channels[axis] = writer.register_channel(
                topic=f"/emboviz/diagnostics/{axis}/severity",
                message_encoding="json",
                schema_id=string_schema,
            )

        # Messages
        for i, scene in enumerate(trajectory.frames):
            log_time_ns = int(i * (1_000_000_000 / fps))

            for cam_name, rgb in scene.observations.images.items():
                if cam_name not in camera_channels:
                    continue
                buf = io.BytesIO()
                pil = rgb.data if isinstance(rgb.data, Image.Image) else Image.fromarray(np.asarray(rgb.data))
                pil.save(buf, format="JPEG", quality=85)
                import base64
                payload = json.dumps({
                    "timestamp": {
                        "sec": log_time_ns // 1_000_000_000,
                        "nsec": log_time_ns % 1_000_000_000,
                    },
                    "frame_id": cam_name,
                    "data": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "format": "jpeg",
                }).encode("utf-8")
                writer.add_message(
                    channel_id=camera_channels[cam_name],
                    log_time=log_time_ns,
                    publish_time=log_time_ns,
                    data=payload,
                )

            for axis, traj_result in per_axis_results.items():
                if i >= len(traj_result.per_frame):
                    continue
                r = traj_result.per_frame[i]
                score = r.scalar_score
                if score == score:
                    writer.add_message(
                        channel_id=diag_score_channels[axis],
                        log_time=log_time_ns,
                        publish_time=log_time_ns,
                        data=json.dumps({"value": float(score)}).encode("utf-8"),
                    )
                writer.add_message(
                    channel_id=diag_severity_channels[axis],
                    log_time=log_time_ns,
                    publish_time=log_time_ns,
                    data=json.dumps({"value": r.severity.value}).encode("utf-8"),
                )

        writer.finish()
    return out_path
