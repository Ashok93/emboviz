"""Cross-frame attention drift diagnostic.

Across a trajectory, extract the model's attention map at each frame
and compute the attention centroid (where on the image the model is
looking). If the centroid moves a lot across frames where the target
should be roughly stationary, the policy is "drifting" — attention
isn't anchored. Drift correlates with brittle policies that may grasp
adjacent to the target.

Requires Capability.ATTENTION. Returns Severity.UNKNOWN if the model
doesn't expose attention.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Finding, Severity
from emboviz.core.types import Scene, TokenSelector, Trajectory
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, NotSupported, VLAModel


class AttentionDriftDiagnostic(Diagnostic):
    """Attention centroid stability across trajectory frames.

    This diagnostic operates on whichever image stream the model's
    ``extract_attention`` reports attention over (the standard VLA case
    is one image fed to the LLM). The pixel-space conversion of the
    attention centroid is done against the named ``camera`` — defaults
    to "primary". Pass a different ``camera`` if your model attends over
    a non-primary stream; the diagnostic will raise rather than silently
    pretend that primary is what the model attends to when it isn't.
    """

    required_capabilities = Capability.INFERENCE | Capability.ATTENTION

    def __init__(
        self,
        drift_warn_px: float = 30.0,
        drift_critical_px: float = 70.0,
        query: Optional[TokenSelector] = None,
        camera: str = "primary",
    ):
        self.drift_warn_px = drift_warn_px
        self.drift_critical_px = drift_critical_px
        self.query = query or TokenSelector(relative="before_action")
        self.camera = camera
        self.name = "attention_drift"
        self.axis = "internal.attention_drift"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        return self._not_applicable(
            model, scene,
            "attention_drift requires a Trajectory; use run_trajectory()",
        )

    def run_trajectory(self, model: VLAModel, trajectory: Trajectory) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(
                model, trajectory.frames[0] if trajectory.frames else None,
                "model lacks ATTENTION capability",
            )
        if len(trajectory.frames) < 2:
            return self._not_applicable(
                model, trajectory.frames[0] if trajectory.frames else None,
                "need ≥2 frames for drift measurement",
            )

        # Strict camera check: the named camera must exist in the scene.
        first_scene = trajectory.frames[0]
        if self.camera not in first_scene.observations.images:
            raise ValueError(
                f"AttentionDriftDiagnostic configured for camera='{self.camera}' "
                f"but scene only has {sorted(first_scene.observations.images)}. "
                "Pass a different `camera` to the constructor or load the missing "
                "camera in the dataset adapter — never falling back silently."
            )

        centroids: list[tuple[float, float]] = []   # (cy_norm, cx_norm) in [0,1]
        for scene in trajectory.frames:
            try:
                attn = model.extract_attention(scene, self.query)
            except NotSupported as e:
                return self._not_applicable(
                    model, scene, f"attention extraction failed: {e}",
                )
            # Clean per-camera attention heatmap: mid-layer head filter
            # + sink masking. The literature is unambiguous that raw
            # all-layers-all-heads averaging is dominated by softmax
            # routing artifacts (RoPE sinks, BOS sinks, early-layer
            # token-grouping). See LITERATURE.md §4. For
            # power-users wanting raw, call ``attn.image_weights(cam)``
            # directly. The diagnostic always uses ``_clean``.
            img_attn, _debug = attn.image_weights_clean(self.camera)   # (side, side)
            side = img_attn.shape[0]
            total = img_attn.sum()
            if total <= 0:
                raise RuntimeError(
                    f"AttentionDriftDiagnostic: image-attention sums to "
                    f"{total:.3e} on scene '{scene.scene_id}'. Attention "
                    "from softmax should be strictly positive; zero / "
                    "negative attention indicates a model-adapter bug "
                    "(wrong image_token_range slice, attention extracted "
                    "from a layer with masked heads, etc.). Refusing to "
                    "fabricate a (0.5, 0.5) centroid — fix the adapter."
                )
            img_norm = img_attn / total
            yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
            cy = float((img_norm * yy).sum()) / max(side - 1, 1)
            cx = float((img_norm * xx).sum()) / max(side - 1, 1)
            centroids.append((cy, cx))

        # Pixel-space conversion uses the configured camera's image size.
        h, w = np.asarray(first_scene.observations.images[self.camera].data).shape[:2]
        centroids_px = [(cy * h, cx * w) for cy, cx in centroids]

        # Compute frame-to-frame centroid displacement in pixels.
        displacements: list[float] = []
        for i in range(1, len(centroids_px)):
            cy0, cx0 = centroids_px[i - 1]
            cy1, cx1 = centroids_px[i]
            displacements.append(float(np.hypot(cy1 - cy0, cx1 - cx0)))
        mean_drift = float(np.mean(displacements)) if displacements else 0.0
        max_drift = float(np.max(displacements)) if displacements else 0.0

        raw_numbers = {
            "mean_drift_px":     mean_drift,
            "max_drift_px":      max_drift,
            "warn_threshold_px": self.drift_warn_px,
            "critical_threshold_px": self.drift_critical_px,
            "image_size_hw":     [h, w],
            "camera":            self.camera,
            "n_frame_pairs":     len(displacements),
        }
        if mean_drift >= self.drift_critical_px:
            sev = Severity.CRITICAL
            finding = Finding(
                observed=(
                    f"On camera '{self.camera}', the model's attention "
                    f"centroid moves an average of {mean_drift:.1f} "
                    f"pixels between consecutive frames (image is "
                    f"{w}×{h} px). Largest jump was {max_drift:.1f} px."
                ),
                meaning=(
                    "Attention is wandering frame-to-frame instead of "
                    "tracking the manipulated region. In deployment "
                    "recordings, this often appears in the few frames "
                    "before a failure — the policy lost its visual anchor."
                ),
                next_step=(
                    "Open the Rerun rollout, scrub to the frames with "
                    "the largest drifts (raw_numbers['n_frame_pairs']), "
                    "and check what the attention overlay is pointing "
                    "at versus what the gripper is doing."
                ),
                raw_numbers=raw_numbers,
            )
            verdict = (
                f"Attention centroid drifts an average of {mean_drift:.1f} px "
                f"frame-to-frame (≥ critical {self.drift_critical_px} px). "
                f"The model isn't anchoring its visual focus."
            )
        elif mean_drift >= self.drift_warn_px:
            sev = Severity.MODERATE
            finding = Finding(
                observed=(
                    f"On camera '{self.camera}', the model's attention "
                    f"centroid moves {mean_drift:.1f} pixels per frame "
                    f"(image is {w}×{h} px). Some movement, but not "
                    f"severe."
                ),
                meaning=(
                    "The policy's visual focus shifts noticeably "
                    "frame-to-frame. May indicate it's tracking the "
                    "gripper or following a moving target — normal in "
                    "active manipulation; concerning in static phases."
                ),
                next_step=(
                    "Use the Rerun overlay to confirm the centroid is "
                    "following something task-relevant (the target, the "
                    "gripper). If it's jumping to background regions, "
                    "the model is losing focus."
                ),
                raw_numbers=raw_numbers,
            )
            verdict = (
                f"Attention centroid drifts {mean_drift:.1f} px frame-to-frame "
                f"(≥ warning {self.drift_warn_px} px). Some focus instability."
            )
        else:
            sev = Severity.PASS
            finding = Finding(
                observed=(
                    f"On camera '{self.camera}', the model's attention "
                    f"centroid is stable: only {mean_drift:.1f} px of "
                    f"drift per frame on a {w}×{h} px image."
                ),
                meaning=(
                    "The model holds a consistent visual focus across "
                    "the window — healthy for a policy that's actively "
                    "tracking a task-relevant region."
                ),
                next_step="No action needed.",
                raw_numbers=raw_numbers,
            )
            verdict = (
                f"Attention centroid is stable (drift {mean_drift:.1f} px < "
                f"warning {self.drift_warn_px} px). Model is visually anchored."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=trajectory.episode_id or trajectory.source or "trajectory",
            scalar_score=mean_drift,
            severity=sev,
            direction="higher_is_worse",
            explanation=verdict,
            finding=finding,
            per_variant={f"drift_{i}_to_{i+1}": d for i, d in enumerate(displacements)},
            raw={
                "centroids_normalized": centroids,
                "centroids_pixel": centroids_px,
                "displacements_pixel": displacements,
                "image_size_hw": [h, w],
                "camera": self.camera,
                "drift_warn_px": self.drift_warn_px,
                "drift_critical_px": self.drift_critical_px,
            },
        )
