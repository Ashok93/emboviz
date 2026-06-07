"""Masked counterfactual object swap for the closed-loop dream seed.

The dream stress test seeds Cosmos from a real frame. This module produces a
*counterfactual* seed from that frame by locating an object (SAM 3) and either
removing it (LaMa fills the masked region with plausible background) or replacing
it with a different object (Stable Diffusion text-guided inpainting paints the
described object into the masked region). The dream is then run from both the
original seed and the edited seed and shown side by side, so the policy's
behaviour under the counterfactual is judged against reality.

Mode is chosen by ``replace_query``: empty -> remove (needs an ``Inpainter``,
e.g. LaMa); non-empty -> insert that object (needs an ``ObjectInserter``, e.g.
the SD inpaint adapter).

Why per-camera, and why "keep the original when not detected":

  The seed Cosmos conditions on is a concat of all three DROID cameras, and the
  policy under test consumes its individual cameras — so every camera must carry
  a valid image or the dream cannot run. SAM 3 is run independently per camera
  (the object's size/visibility differs per viewpoint; the distant exteriors
  often cannot resolve a small object the close wrist sees clearly). A camera
  with a confident detection is edited; a camera without one keeps its ORIGINAL
  image. That is the defined behaviour, not a silent fallback: the per-camera
  outcome (edited / kept-original, with the reason) is recorded on
  :class:`SwapResult` so the caller can surface a partial edit honestly and never
  present "wrist-only swap" as a full swap across every view.

This module is torch-free. It composes the SAM 3 detector (a ``TargetDetector``)
and either an inpainter (``Inpainter``, removal) or an object inserter
(``ObjectInserter``, replacement) — all injected by the caller, which owns
bringing up the corresponding workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from emboviz.core.types import Scene
from emboviz.perturb._target_detection import TargetDetector
from emboviz.perturb.image._image_utils import to_array, to_pil
from emboviz.perturb.image._inpaint import Inpainter, ObjectInserter


@dataclass(frozen=True)
class CameraSwap:
    """Per-camera outcome of a scene edit."""

    region: str                     # concat region: wrist | exterior_left | exterior_right
    role: str                       # the episode camera role this region maps to
    detected: bool                  # did SAM locate the target on this camera?
    edited: bool                    # was the image actually changed?
    operation: str                  # "insert" | "remove" | "none"
    reason: str                     # human-readable status (esp. why nothing changed)
    label: Optional[str] = None     # detector's label for the located object
    confidence: Optional[float] = None


@dataclass(frozen=True)
class SwapResult:
    """Result of editing a seed frame across its concat cameras."""

    images_by_region: dict[str, np.ndarray]   # region -> (possibly edited) uint8 RGB image
    per_camera: list[CameraSwap]              # one entry per concat region, in input order
    mask_query: str
    replace_query: str

    @property
    def any_edited(self) -> bool:
        """True if at least one camera was actually edited."""
        return any(c.edited for c in self.per_camera)

    @property
    def edited_regions(self) -> list[str]:
        return [c.region for c in self.per_camera if c.edited]

    def summary(self) -> str:
        """One-line per-camera status, for logs and the clip's context card."""
        op = f"insert→{self.replace_query!r}" if self.replace_query else "remove"
        parts = []
        for c in self.per_camera:
            if c.edited:
                conf = f" {c.confidence:.2f}" if c.confidence is not None else ""
                parts.append(f"{c.region}: {c.operation}{conf}")
            else:
                parts.append(f"{c.region}: original ({c.reason})")
        return f"swap[{self.mask_query!r} {op}] — " + "; ".join(parts)


class SceneSwapper:
    """Build a counterfactual seed by editing a detected object per camera.

    Parameters
    ----------
    mask_query
        SAM 3 phrase naming the object to locate (e.g. ``"the marker"``).
    detector
        A ``TargetDetector`` (e.g. ``SAM3Detector``) that, given a Scene whose
        ``"primary"`` camera is the image to inspect, returns a detection with a
        pixel mask or ``None``.
    replace_query
        What to put in the object's place. Empty -> remove the object.
    inpainter
        An ``Inpainter`` (e.g. ``LamaInpainter``) — required when
        ``replace_query`` is empty (removal).
    inserter
        An ``ObjectInserter`` (e.g. ``SDInpaintInserter``) — required when
        ``replace_query`` is non-empty (insertion).
    """

    def __init__(
        self,
        *,
        mask_query: str,
        detector: TargetDetector,
        replace_query: str = "",
        inpainter: Optional[Inpainter] = None,
        inserter: Optional[ObjectInserter] = None,
    ):
        if not mask_query or not mask_query.strip():
            raise ValueError("SceneSwapper: mask_query must be a non-empty phrase.")
        self.mask_query = mask_query.strip()
        self.replace_query = (replace_query or "").strip()
        self.detector = detector
        if self.replace_query:
            if inserter is None:
                raise ValueError(
                    "SceneSwapper: replace_query is set, so an ObjectInserter "
                    "(`inserter=`) is required to paint the replacement object."
                )
        else:
            if inpainter is None:
                raise ValueError(
                    "SceneSwapper: replace_query is empty (removal mode), so an "
                    "Inpainter (`inpainter=`) is required to fill the removed region."
                )
        self.inpainter = inpainter
        self.inserter = inserter

    def swap(self, frame: Scene, concat_cameras: dict[str, str]) -> SwapResult:
        """Edit the target across each concat camera of ``frame``.

        ``concat_cameras`` maps each concat region (wrist / exterior_left /
        exterior_right) to the episode camera role that supplies it. Returns a
        :class:`SwapResult` whose ``images_by_region`` always covers every region
        (edited where detected, original where not) so the caller can build a
        complete concat seed.
        """
        images: dict[str, np.ndarray] = {}
        records: list[CameraSwap] = []
        for region, role in concat_cameras.items():
            if role not in frame.observations.images:
                raise KeyError(
                    f"SceneSwapper: concat region {region!r} maps to camera role "
                    f"{role!r}, which is not in the frame "
                    f"(available: {sorted(frame.observations.images)})."
                )
            arr = to_array(frame.observations.images[role].data).astype(np.uint8)
            # Probe scene: a fresh single-camera Scene exposing THIS camera's image
            # under "primary" (the key detectors read). Built standalone rather than
            # aliasing the frame so detection does not depend on the frame already
            # having a "primary" camera.
            probe = Scene.from_image(to_pil(arr), instruction=frame.instruction)
            detection = self.detector(probe)

            if detection is None or detection.mask is None:
                images[region] = arr
                reason = (
                    "no mask" if detection is not None
                    else f"{self.mask_query!r} not detected"
                )
                records.append(CameraSwap(
                    region=region, role=role, detected=detection is not None,
                    edited=False, operation="none", reason=reason,
                ))
                continue

            mask = np.asarray(detection.mask).astype(bool)
            if mask.shape != arr.shape[:2]:
                raise ValueError(
                    f"SceneSwapper: detector mask {mask.shape} does not match the "
                    f"{region!r} image {arr.shape[:2]}. The detector must return a "
                    "mask at the queried image's resolution."
                )
            if not mask.any():
                images[region] = arr
                records.append(CameraSwap(
                    region=region, role=role, detected=True, edited=False,
                    operation="none", reason="empty mask",
                    label=detection.label, confidence=detection.confidence,
                ))
                continue

            if self.replace_query:
                assert self.inserter is not None
                images[region] = np.asarray(
                    self.inserter.insert(arr, mask, self.replace_query), dtype=np.uint8
                )
                operation = "insert"
            else:
                assert self.inpainter is not None
                images[region] = np.asarray(
                    self.inpainter.inpaint(arr, mask), dtype=np.uint8
                )
                operation = "remove"
            records.append(CameraSwap(
                region=region, role=role, detected=True, edited=True,
                operation=operation, reason="ok",
                label=detection.label, confidence=detection.confidence,
            ))

        return SwapResult(
            images_by_region=images, per_camera=records,
            mask_query=self.mask_query, replace_query=self.replace_query,
        )


__all__ = ["CameraSwap", "SwapResult", "SceneSwapper"]
