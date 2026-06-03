"""Memorization-sniff diagnostic — does the policy USE its visual input?

If we mask out the target and the model still produces an action close to
the unmasked baseline, it's running on memorized trajectories conditioned
on non-visual signals (proprio, instruction, action history), not on
visual feedback. This is the LIBERO-Pro signature (Geng et al. 2025,
arXiv:2510.03827) and the BYOVLA visual-robustness probe inverted
(Hancock et al. 2024, arXiv:2410.01971).

Implementation principles (LITERATURE.md §1):

  1. **Open-vocabulary target localization.** Default detector is SAM 3
     (single model, text → mask, native concept-aware prompting). The
     legacy GroundingDINO + SAM combo is kept as a maintained fallback
     for environments where SAM 3 isn't available. Users with their own
     tracker / motion-capture / hand-labelled bboxes plug in via the
     ``CallableConnector`` / ``JSONAnnotationConnector`` /
     ``CocoAnnotationConnector`` connectors.

  2. **Fill ensemble.** We mask with TWO independent fills (channel-mean
     and Gaussian blur) and require AGREEMENT across both fills before
     calling memorization. Single-fill is susceptible to "baseline
     blindness" (Sturmfels, Lundberg & Lee 2020): if the masked region
     matches the fill color the model treats it as informative noise.

  3. **Intervention magnitude reporting.** For every fill we report
     ``mask_contrast`` — the pixel-L2 difference between the masked
     region's original content and the fill, normalized to [0, 1].
     This is the "did the image actually change?" sanity gate from
     Vig et al. 2020 causal-mediation framing. If contrast is too low,
     the intervention didn't happen (e.g. fill ≈ target color); we
     refuse to emit a verdict.

  4. **Per-camera detection and masking.** Each camera is queried
     independently via a probe scene that aliases the camera as ``primary``
     and sets ``scene.metadata['_emboviz_probe_camera']`` so annotation
     connectors can resolve the right per-camera entry. The target is masked
     on every camera where it is found, in one perturbed scene. A frame is
     scored only when it is located on the required cameras (``require_cameras``
     — the primary scene view by default); otherwise it is reported as
     "couldn't test" and excluded from the trajectory verdict rather than
     scored on a partial mask. Any in-scope camera left unmasked is disclosed.

  5. **N-sample averaging for stochastic policies.** π0, GR00T,
     diffusion policies need K samples per prediction averaged before
     comparison (handled via ``averaged_predict`` and the
     ``ModelCalibration.n_samples`` derivation). The runner can compute
     the per-frame baseline ONCE and pass it via the ``baseline=``
     kwarg so the diagnostic does not duplicate that work across the
     other diagnostics.

  6. **Required masks (no bbox-only fallback).** The diagnostic needs a
     pixel-accurate mask to make the intervention interpretable.
     Detectors that return only a bbox raise rather than silently
     degrading.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from emboviz.calibration import ModelCalibration, averaged_predict, averaged_predict_batch
from emboviz.core.results import DiagnosticResult, Finding, Severity
from emboviz.core.types import ActionResult, Scene
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, VLAModel
from emboviz.perturb._target_detection import (
    BBoxDetector,
    TargetDetector,
)
from emboviz.perturb.image._image_utils import to_array, to_pil
from emboviz.perturb.image._inpaint import Inpainter


# Minimum normalized contrast (in [0,1]) between fill colour and the
# original target region for the intervention to count as "real." Below
# this the mask is visually indistinguishable from the target and the
# Δaction is uninterpretable. Default 0.05 (≈5% of pixel range) is a
# conservative threshold — for a 256-grey image, 13 steps of difference.
DEFAULT_MIN_MASK_CONTRAST = 0.05


# Removal-mask dilation. A pixel-tight detection silhouette excludes the
# object's anti-aliased boundary (and any thin contact shadow): the OOD
# fills then leave that 1–2 px rim at its original colour, and LaMa
# reconstructs it straight back into the hole — a faint ghost of the
# removed object. Growing the mask by a small margin before the fill
# ensemble runs makes every fill cover the whole object incl. its edge.
# This is the standard erase-mask preprocessing (Suvorov et al. 2022, the
# LaMa object-removal recipe; iopaint / lama-cleaner dilate erase masks
# for the same reason — LITERATURE.md §1). The same dilated mask feeds
# ALL fills and the contrast gate, so the OOD↔on-manifold agreement is
# still measured over one identical region. Scaled to the frame's shorter
# side so it is resolution-independent.
DEFAULT_MASK_DILATION_FRAC = 0.03


def _dilation_radius(image_hw: tuple[int, int], frac: float) -> int:
    """Removal-mask dilation radius in pixels for a frame of shape
    ``image_hw`` (H, W). ``max(2, round(min(H, W) * frac))`` — at least
    2 px so even a tiny frame covers the anti-aliased rim."""
    return max(2, int(round(min(image_hw) * float(frac))))


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate boolean ``mask`` by ``radius`` px (≈ square structuring
    element, chebyshev growth). Grows a tight detection silhouette into a
    clean removal mask before the fill ensemble runs. ``radius <= 0`` is a
    no-op (returns the mask as bool)."""
    m = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return m
    from scipy.ndimage import binary_dilation, generate_binary_structure
    selem = generate_binary_structure(2, 2)   # 3×3 full → grow in all dirs
    return binary_dilation(m, structure=selem, iterations=int(radius))


# ── Fill modes (LITERATURE.md §1) ─────────────────────────────────────
#
# The mask-fill ensemble. Single-fill baselines suffer "baseline
# blindness" and create OOD inputs that bias the verdict (Sturmfels et al.
# 2020; ROAR/ROAD). We require AGREEMENT across fills spanning the
# OOD/on-manifold axis before calling memorization.
CHANNEL_MEAN_FILL = "channel_mean"     # aggressive, OOD (Zeiler-Fergus 2014)
GAUSSIAN_BLUR_FILL = "gaussian_blur"   # OOD-leaning, on-image colour
LAMA_INPAINT_FILL = "lama_inpaint"     # on-manifold (LaMa, Suvorov et al. 2022)

# The two pure-numpy fills, shippable with core (no worker). Default
# ensemble: lama_inpaint is opt-in because it needs the emboviz-lama
# worker (torch). Adding it lifts the "2-of-3" honesty caveat below.
DEFAULT_FILL_MODES = [CHANNEL_MEAN_FILL, GAUSSIAN_BLUR_FILL]
# The full ensemble the literature prescribes (OOD ∪ on-manifold).
LITERATURE_FILL_MODES = [CHANNEL_MEAN_FILL, GAUSSIAN_BLUR_FILL, LAMA_INPAINT_FILL]
# Fills that keep the filled hole on the data manifold (plausible
# background rather than an alien colour / blur).
ON_MANIFOLD_FILLS = frozenset({LAMA_INPAINT_FILL})
KNOWN_FILL_MODES = frozenset(LITERATURE_FILL_MODES)


def _apply_fill(arr: np.ndarray, mask: np.ndarray, fill_mode: str) -> np.ndarray:
    """Apply one of the supported fills to the mask region.

    ``arr`` is HxWxC uint8. ``mask`` is HxW bool. Returns a new uint8
    array with the masked region replaced.

    Fills:
      • ``"channel_mean"``  — fill with the per-channel mean of the
        whole image (the Zeiler-Fergus 2014 baseline). Strong intervention
        but can suffer baseline-blindness if target is itself near-mean.
      • ``"gaussian_blur"`` — replace the masked region with a heavily-
        blurred version of ITSELF. Stays on-manifold (doesn't introduce
        an alien colour) while still destroying high-frequency target
        content. Sigma scaled with the mask's bbox diameter so the blur
        magnitude matches target size.
    """
    out = arr.copy()
    if fill_mode == "channel_mean":
        chan_mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.uint8)
        out[mask] = chan_mean
        return out
    if fill_mode == "gaussian_blur":
        # Sigma proportional to mask diameter (bbox-based). For a 50x50
        # mask in a 480x640 image, sigma ≈ 50/3 ≈ 17 pixels — strong
        # blur that removes object identity but preserves rough colour.
        ys, xs = np.where(mask)
        if ys.size == 0:
            return out
        bbox_diag = float(
            np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        )
        sigma = max(2.0, bbox_diag / 6.0)
        # Lazy import scipy only when needed.
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError as e:
            raise ImportError(
                "memorization gaussian_blur fill requires scipy. "
                "Install scipy or pass fill_modes=['channel_mean'] only."
            ) from e
        # Blur the WHOLE image, then paste blurred pixels into masked area.
        blurred = np.stack([
            gaussian_filter(arr[..., c].astype(np.float32), sigma=sigma)
            for c in range(arr.shape[-1])
        ], axis=-1).astype(np.uint8)
        out[mask] = blurred[mask]
        return out
    if fill_mode == LAMA_INPAINT_FILL:
        # lama_inpaint is model-backed (the emboviz-lama worker), not a
        # pure function of (arr, mask). It must be routed through
        # ``apply_fill`` with an inpainter; reaching the pure helper means
        # a caller bypassed that dispatch.
        raise ValueError(
            "fill_mode 'lama_inpaint' is model-backed and cannot be applied "
            "by the pure _apply_fill helper. Route it through apply_fill() "
            "with an inpainter."
        )
    raise ValueError(
        f"Unknown fill_mode={fill_mode!r}. Supported: {sorted(KNOWN_FILL_MODES)}."
    )


def apply_fill(
    arr: np.ndarray,
    mask: np.ndarray,
    fill_mode: str,
    *,
    inpainter: Optional[Inpainter] = None,
    cache_key: Optional[tuple] = None,
) -> np.ndarray:
    """Apply any fill mode, dispatching the model-backed one to ``inpainter``.

    The single fill entry-point shared by the diagnostic and the runner's
    Rerun-overlay reconstruction, so both produce byte-identical masked
    images. Pure fills (``channel_mean``, ``gaussian_blur``) are handled
    by :func:`_apply_fill`; ``lama_inpaint`` is delegated to ``inpainter``
    (the LaMa worker), keyed by ``cache_key`` so the two call sites share
    one forward pass per (frame, camera).

    Raises if ``lama_inpaint`` is requested without an ``inpainter`` — no
    silent fallback to a pure fill (that would mislabel which fills the
    agreement gate ran over).
    """
    if fill_mode == LAMA_INPAINT_FILL:
        if inpainter is None:
            raise ValueError(
                "fill mode 'lama_inpaint' requires the LaMa inpainter "
                "worker, but no inpainter was provided. LaMa ships with emboviz "
                "and its worker builds automatically when a config requests the "
                "lama_inpaint fill — request it via analysis.fills, or drop "
                "'lama_inpaint' from analysis.fills."
            )
        return inpainter.inpaint(arr, mask, key=cache_key)
    return _apply_fill(arr, mask, fill_mode)


def _mask_contrast(
    original: np.ndarray, masked: np.ndarray, mask: np.ndarray,
) -> float:
    """Pixel-L2 difference between original and masked image, restricted
    to the masked region, normalized to [0, 1].

    Approximates LPIPS (which would require a learned perceptual model)
    at a fraction of the cost. A value of 0 means the mask is identical
    to the original; 1 means maximum possible pixel difference.
    """
    if not mask.any():
        return 0.0
    diff = (original.astype(np.float32) - masked.astype(np.float32))
    if diff.ndim == 3:
        per_pixel = np.linalg.norm(diff, axis=-1)
    else:
        per_pixel = np.abs(diff)
    masked_pixels = per_pixel[mask]
    # Max possible per-pixel L2 difference for 3-channel uint8 = sqrt(3) * 255
    max_per_pixel = float(np.sqrt(diff.shape[-1] if diff.ndim == 3 else 1) * 255.0)
    return float(masked_pixels.mean() / max_per_pixel)


class MemorizationDiagnostic(Diagnostic):
    """Mask the target on the cameras that show it; check whether the model
    still executes a coherent action.

    Args:
        target_detector: how to locate the target. REQUIRED — there is NO
            default: the constructor raises unless one of ``target_detector``,
            ``bbox``, or a runner-supplied target_text is given (the
            diagnostic refuses to guess what to mask). Pass e.g.
            ``SAM3Detector(target_text="the pipe")`` to scope to a specific
            referent in multi-object instructions.
        bbox: shortcut — fixed bbox in every camera (only valid when
            cameras share resolution/intrinsics).
        fill_modes: which fills to ensemble. Default =
            ``["channel_mean", "gaussian_blur"]`` (agreement across fills is
            required for a CRITICAL verdict). The on-manifold ``lama_inpaint``
            fill prescribed by LITERATURE.md §1 is available: add it (it needs
            the emboviz-lama worker + an ``inpainter``) to span the full
            OOD↔on-manifold axis. Which fills the agreement gate actually ran
            over is surfaced in the result's raw output (``fill_ensemble``).
            Pass ``["channel_mean"]`` to skip the blur fill if scipy is missing.
        mask_dilation_frac: grow each detected mask by
            ``round(min(H, W) * frac)`` px before filling, so the fills cover
            the object's anti-aliased boundary instead of leaving a rim (which
            LaMa reconstructs back into the hole). The same dilated mask feeds
            ALL fills and the contrast gate. Default
            ``DEFAULT_MASK_DILATION_FRAC`` (0.03).
        min_mask_contrast: refuse to emit CRITICAL when ALL fills
            produce a normalized contrast below this on a frame. Default
            ``DEFAULT_MIN_MASK_CONTRAST`` (0.05).
        noise_floor_score: anchored 0-1 score below which the model is
            "ignoring the intervention" (memorization signature).
        grounded_threshold: anchored 0-1 score above which the model is
            "genuinely reading the scene."
        cameras: which cameras to operate on. None = every camera in the scene.
        require_cameras: views that must carry a detection for a frame to be
            scored. ``"primary"`` (default) gates on the main scene view;
            ``"all"`` requires every camera; a list names explicit roles. The
            target is masked on every camera where it is found, and any
            in-scope view left unmasked is disclosed in the result.
        calibration: per-model anchors from
            ``emboviz.calibration.calibrate_model``. Required for
            anchored thresholds to mean the same thing across models;
            without it the diagnostic reports raw L2 scores.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        target_detector: Optional[TargetDetector] = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
        fill_modes: Optional[list[str]] = None,
        min_mask_contrast: float = DEFAULT_MIN_MASK_CONTRAST,
        noise_floor_score: float = 0.05,
        grounded_threshold: float = 0.30,
        cameras: Optional[list[str]] = None,
        require_cameras: str | list[str] = "primary",
        calibration: Optional["ModelCalibration"] = None,
        inpainter: Optional[Inpainter] = None,
        mask_dilation_frac: float = DEFAULT_MASK_DILATION_FRAC,
    ):
        if bbox is not None:
            self.detector: TargetDetector = BBoxDetector(bbox)
        elif target_detector is not None:
            self.detector = target_detector
        else:
            # Default detector — text-required. The diagnostic is never
            # constructed without either a bbox, a custom detector, or a
            # target_text being passed through by the runner. We pick
            # SAM 3 over the legacy GD+SAM combo (single model, faster,
            # native concept prompting). Adapters that want the old
            # pipeline pass it explicitly.
            raise ValueError(
                "MemorizationDiagnostic needs one of: ``bbox=(x0,y0,x1,y1)`` "
                "for a fixed user-supplied box, or ``target_detector=...`` "
                "for any TargetDetector (SAM3Detector / "
                "GroundingDINOSAMDetector / JSONAnnotationConnector / "
                "CocoAnnotationConnector / CallableConnector). The "
                "diagnostic refuses to guess what to mask."
            )
        self.fill_modes = list(fill_modes) if fill_modes else list(DEFAULT_FILL_MODES)
        unknown = [f for f in self.fill_modes if f not in KNOWN_FILL_MODES]
        if unknown:
            raise ValueError(
                f"unknown fill mode(s) {unknown}; supported: "
                f"{sorted(KNOWN_FILL_MODES)}."
            )
        if LAMA_INPAINT_FILL in self.fill_modes and inpainter is None:
            # No silent fallback to the pure fills: a CRITICAL verdict's
            # honesty metadata names which fills the agreement gate ran
            # over, so we must not quietly drop the on-manifold fill.
            raise ValueError(
                "fill_modes includes 'lama_inpaint' but no ``inpainter`` was "
                "given. The on-manifold fill needs the emboviz-lama worker. "
                "Pass inpainter=LamaInpainter() (the runner wires this up "
                "automatically), or remove 'lama_inpaint' from the fills."
            )
        self.inpainter = inpainter
        self.min_mask_contrast = float(min_mask_contrast)
        self.mask_dilation_frac = float(mask_dilation_frac)
        self.noise_floor_score = noise_floor_score
        self.grounded_threshold = grounded_threshold
        self.cameras = cameras
        self.require_cameras = require_cameras
        self.calibration = calibration
        self.name = "memorization_test"
        self.axis = "vision.memorization"

    def run(
        self, model: VLAModel, scene: Scene,
        *, baseline: Optional[ActionResult] = None,
    ) -> DiagnosticResult:
        """Evaluate memorization for one scene.

        ``baseline`` is an optional precomputed unperturbed prediction.
        The runner computes one baseline per frame and shares it across
        every diagnostic; without that, each diagnostic would re-run
        ``averaged_predict`` and we'd pay n_samples × num-diagnostics
        worth of forward passes per frame.
        """
        from emboviz.core.types import resolve_cameras
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        cameras = resolve_cameras(scene, self.cameras)
        n_samples = self.calibration.n_samples if self.calibration else 1
        if baseline is None:
            baseline = averaged_predict(model, scene, n_samples)

        # 1. Per-camera target detection. Each camera is probed via a
        # scene whose primary alias points at that camera, with the
        # ``_emboviz_probe_camera`` metadata key set so user-annotation
        # connectors (JSON / COCO / Callable) can resolve the right
        # per-camera entry.
        per_cam_detection: dict = {}
        per_cam_original: dict = {}
        per_cam_mask: dict = {}        # dilated removal mask (shared by all fills)
        per_cam_dilation: dict = {}    # px radius applied, for transparency
        for cam in cameras:
            cam_image = scene.observations.images[cam].data
            if "primary" in scene.observations.images:
                probe_scene = scene.with_image(cam_image, camera="primary")
            else:
                probe_scene = scene.with_image(cam_image, camera=cam)
            # Tag the probe so connectors see which camera we're querying.
            # ``Scene`` is frozen, so we re-create with augmented metadata.
            tagged_meta = dict(probe_scene.metadata)
            tagged_meta["_emboviz_probe_camera"] = cam
            probe_scene = replace(probe_scene, metadata=tagged_meta)
            detection = self.detector(probe_scene)
            per_cam_detection[cam] = detection
            if detection is None:
                continue
            arr = to_array(cam_image)
            if detection.mask is None or detection.mask.shape != arr.shape[:2]:
                # Detector contract: mask must be HxW matching image.
                # GroundingDINOSAMDetector enforces this; a custom
                # detector that violates it must be caught.
                raise ValueError(
                    f"Target detector for camera '{cam}' returned a "
                    f"detection without a pixel-accurate mask (got "
                    f"mask={detection.mask!r}). Memorization requires a "
                    "pixel mask; bbox-only is too coarse. Use a detector "
                    "that produces a mask (SAM3Detector / GroundingDINOSAMDetector)."
                )
            per_cam_original[cam] = arr
            # Grow the tight detection silhouette into a clean removal mask
            # ONCE per camera (shared by every fill + the contrast gate).
            radius = _dilation_radius(arr.shape[:2], self.mask_dilation_frac)
            per_cam_mask[cam] = _dilate_mask(detection.mask, radius)
            per_cam_dilation[cam] = radius

        # Score a frame only when the target is located on the required
        # cameras (``require_cameras``); mask it wherever it is found. The
        # default "primary" falls back to strict-all if the scene exposes no
        # camera by that role; an explicit value that names no scene camera is
        # a configuration error, not a silent relaxation.
        if self.require_cameras == "all":
            required = list(cameras)
        elif self.require_cameras == "primary":
            required = ["primary"] if "primary" in cameras else list(cameras)
        else:
            names = ([self.require_cameras] if isinstance(self.require_cameras, str)
                     else list(self.require_cameras))
            required = [c for c in names if c in cameras]
            if not required:
                raise ValueError(
                    f"memorization_require_cameras={self.require_cameras!r} names no "
                    f"camera in the scene ({sorted(cameras)}); use 'all', 'primary', "
                    "or a camera role the scene exposes."
                )

        missing_required = [c for c in required if c not in per_cam_original]
        if missing_required:
            located = sorted(per_cam_original)
            if not located:
                return self._not_applicable(
                    model, scene,
                    "could not confidently locate the manipulated target on ANY "
                    f"camera ({sorted(cameras)}). Lower "
                    "analysis.detector_score_threshold, give the detector a more "
                    "specific target_text, or supply per-frame annotations. We "
                    "never fabricate a centred rectangle.",
                )
            return self._not_applicable(
                model, scene,
                f"located the target on {located} but NOT on required "
                f"camera(s) {sorted(missing_required)}: with the target still "
                "visible there, an unchanged action cannot be attributed to "
                "memorization. Lower analysis.detector_score_threshold to catch "
                "it, set analysis.memorization_require_cameras to the view(s) "
                "that actually show the target, or this frame does not show it "
                "on the required view(s).",
            )

        detected_cams = sorted(per_cam_original)
        # In-scope cameras without a detection — left unmasked, disclosed below.
        unmasked_cams = sorted(c for c in cameras if c not in per_cam_original)
        labels = sorted({per_cam_detection[c].label for c in detected_cams})
        confs = {c: round(per_cam_detection[c].confidence, 3) for c in detected_cams}

        # 2. Fill ensemble: for each fill mode, build masked scene,
        # measure intervention magnitude (mask_contrast) and response
        # magnitude (Δaction vs baseline).
        per_fill_results: dict[str, dict] = {}
        # Phase 1 — build one masked scene per fill mode and record its
        # contrast. The fill modes are independent interventions on the same
        # frame, so we predict them as ONE batch (Phase 2) instead of a
        # forward per fill.
        fill_scenes: list[Scene] = []
        fill_meta: list[tuple[str, float, dict[str, float]]] = []
        for fill_mode in self.fill_modes:
            masked_arrays: dict[str, np.ndarray] = {}
            contrasts: dict[str, float] = {}
            for cam in detected_cams:
                arr = per_cam_original[cam]
                mask = per_cam_mask[cam]   # dilated removal mask (shared)
                masked = apply_fill(
                    arr, mask, fill_mode,
                    inpainter=self.inpainter,
                    cache_key=(scene.scene_id, cam),
                )
                masked_arrays[cam] = masked
                contrasts[cam] = _mask_contrast(arr, masked, mask)
            mean_contrast = float(np.mean(list(contrasts.values())))
            masked_pils = {cam: to_pil(a) for cam, a in masked_arrays.items()}
            fill_scenes.append(scene.with_images(masked_pils))
            fill_meta.append((fill_mode, mean_contrast, contrasts))

        # Phase 2 — one batched prediction across all fill modes.
        fill_preds = averaged_predict_batch(model, fill_scenes, n_samples)

        # Phase 3 — Δaction vs baseline per fill (numerics unchanged).
        for (fill_mode, mean_contrast, contrasts), action_masked in zip(
            fill_meta, fill_preds,
        ):
            raw_delta = float(np.linalg.norm(action_masked.action - baseline.action))
            if self.calibration is not None:
                norm_delta = self.calibration.normalize(raw_delta)
            else:
                norm_delta = raw_delta

            per_fill_results[fill_mode] = {
                "mean_contrast":   mean_contrast,
                "per_cam_contrast": contrasts,
                "raw_delta":       raw_delta,
                "normalized_delta": norm_delta,
                "action_masked":   action_masked.action.tolist(),
            }

        # 3. Intervention validity gate. If EVERY fill produced contrast
        # below ``min_mask_contrast`` the intervention was effectively
        # invisible — refuse a verdict.
        max_contrast = max(r["mean_contrast"] for r in per_fill_results.values())
        if max_contrast < self.min_mask_contrast:
            return self._not_applicable(
                model, scene,
                f"intervention too weak to test memorization on "
                f"{detected_cams}: max fill contrast {max_contrast:.3f} < "
                f"{self.min_mask_contrast} on every fill mode "
                f"({list(per_fill_results)}). The target's pixels are "
                "themselves close to every available fill colour, so the "
                "masked image is visually indistinguishable from the "
                "original. Either supply a higher-contrast fill mode or "
                "skip this frame.",
            )

        # 4. Verdict — require agreement across all fills with sufficient
        # contrast. Strongest fill = highest contrast (most aggressive
        # intervention); we trust its Δaction as the headline signal,
        # but bin severity based on the worst-case (max) fill Δaction
        # to be conservative about "ignored" verdicts.
        deltas = [r["normalized_delta"] for r in per_fill_results.values()]
        min_delta = float(min(deltas))   # most conservative — worst signal across fills
        max_delta = float(max(deltas))
        mean_delta = float(np.mean(deltas))

        # Did the ensemble include the on-manifold fill? Drives the
        # honesty caveat below — a CRITICAL verdict means something
        # stronger when the agreement spans the OOD/on-manifold axis.
        _on_manifold_present = any(
            f in ON_MANIFOLD_FILLS for f in per_fill_results
        )

        # The principle (LITERATURE.md §1 step 6):
        #   MEMORIZATION_SIGNATURE iff max(δ) < threshold  (all fills agree → small)
        #   VISUALLY_GROUNDED iff min(δ) > threshold        (all fills agree → large)
        #   else MIXED
        #
        # NEW (signal-vs-noise gate, added 2026-05-26): before declaring
        # memorization we must check that the response is statistically
        # distinguishable from the model's own per-call sampling noise.
        # For a stochastic model (e.g. flow-matching DiT), per-call jitter
        # can dwarf the response to target masking — calling that
        # "memorization" would conflate signal with noise.
        # Noise gate: is the response distinguishable from the model's own
        # per-call jitter? The statistic compared below is max(deltas) over
        # the fills — a MAX, not a mean — and the fills are DIFFERENT
        # interventions, not repeated draws of one input. So we must NOT
        # shrink the floor by sqrt(num_fills): that was a max-vs-mean mismatch
        # that made the threshold ~sqrt(2)x too small and leaked decoding
        # jitter through as a "memorization" verdict. calibration.noise_floor
        # is already the POST-averaging floor (the Δ between two
        # n_samples-averaged predictions), and each fill's delta is exactly
        # one such averaged-prediction Δ — so the correct floor for a single
        # delta is k_samples=1. (A max-of-K Bonferroni bump would be
        # marginally more conservative; k=1 is the single-comparison floor and
        # removes the erroneous shrinkage.)
        signal_threshold = (
            self.calibration.signal_threshold_normalized(k_samples=1)
            if self.calibration is not None
            else self.noise_floor_score
        )
        target_label = labels[0] if labels else "the target"
        cam_phrase = (
            f"camera '{detected_cams[0]}'" if len(detected_cams) == 1
            else f"cameras {detected_cams}"
        )
        # Per-camera detection + masking breakdown, for debugging the blended
        # verdict (which masks every detected camera and reads one Δaction).
        per_camera: dict[str, dict] = {}
        for cam in cameras:
            det = per_cam_detection.get(cam)
            entry: dict = {"masked": cam in per_cam_original}
            if det is not None:
                entry["confidence"] = round(float(det.confidence), 3)
                entry["label"] = det.label
            if cam in per_cam_original:
                entry["dilation_px"] = per_cam_dilation[cam]
                entry["mean_contrast"] = round(float(np.mean(
                    [r["per_cam_contrast"][cam] for r in per_fill_results.values()]
                )), 4)
            per_camera[cam] = entry
        raw_numbers = {
            "min_delta_normalized":   min_delta,
            "max_delta_normalized":   max_delta,
            "mean_delta_normalized":  mean_delta,
            "max_mask_contrast":      max_contrast,
            "signal_threshold":       signal_threshold,
            "ignored_threshold":      self.noise_floor_score,
            "grounded_threshold":     self.grounded_threshold,
            "required_cameras":       list(required),
            "per_camera":             per_camera,
            "fills":                  list(per_fill_results),
            "fill_ensemble": {
                "fills_used":       list(per_fill_results),
                "literature_fills": list(LITERATURE_FILL_MODES),
                "on_manifold_fill_present": _on_manifold_present,
                "note": (
                    "Full literature ensemble: the agreement gate spans both "
                    "the OOD-leaning fills (channel_mean, gaussian_blur) and "
                    "the on-manifold lama_inpaint fill, so a CRITICAL "
                    "'memorization' verdict means every fill — across the "
                    "OOD/on-manifold axis — agrees the action barely moved."
                    if _on_manifold_present else
                    "LITERATURE.md §1 prescribes 3 fills including the "
                    "on-manifold lama_inpaint; it was NOT enabled for this "
                    "run, so the agreement gate ran over OOD-leaning fills "
                    "only (in 'fills_used'). Add 'lama_inpaint' to "
                    "analysis.fills (needs the emboviz-lama worker) to span "
                    "the on-manifold/OOD axis. Until then, read a CRITICAL "
                    "'memorization' verdict as agreement across non-on-"
                    "manifold fills."
                ),
            },
            "detected_cameras":       detected_cams,
            "detection_confidences":  confs,
            "mask_dilation_px":       {c: per_cam_dilation[c] for c in detected_cams},
        }

        if max_delta < signal_threshold:
            sev = Severity.UNKNOWN
            finding = Finding(
                observed=(
                    f"We masked '{target_label}' on {cam_phrase} with two "
                    f"fills (channel-mean + Gaussian blur). The model's "
                    f"action moved by at most {max_delta:.4f} of typical "
                    f"action magnitude — smaller than its own per-call "
                    f"sampling noise threshold ({signal_threshold:.4f})."
                ),
                meaning=(
                    "We can't tell from this single frame whether the "
                    "model is ignoring the target or whether the response "
                    "is just lost in its sampling jitter. This is common "
                    "on quiescent / low-action frames where the policy "
                    "would output a similar action no matter what."
                ),
                next_step=(
                    "Pick a more dynamic frame (mid-episode, during "
                    "active manipulation), increase K samples per "
                    "intervention, or use a higher-contrast fill mode."
                ),
                raw_numbers=raw_numbers,
            )
        elif max_delta < self.noise_floor_score:
            sev = Severity.CRITICAL
            finding = Finding(
                observed=(
                    f"We masked '{target_label}' on {cam_phrase} with two "
                    f"independent fills. The model's action barely moved "
                    f"(at most {max_delta:.3f} of typical action magnitude; "
                    f"both fills agree). Mask actually changed the image "
                    f"(contrast {max_contrast:.2f})."
                ),
                meaning=(
                    "Strong memorized-trajectory signature: the policy "
                    "is predicting from state + instruction + history "
                    "without visually consuming the target. Common when "
                    "running on training-distribution data."
                ),
                next_step=(
                    "Run on an UNSEEN episode (or a held-out task). If "
                    "Δaction grows substantially, your model IS visually "
                    "grounded when forced outside its training "
                    "distribution — this frame just happened to be one "
                    "the policy had memorized."
                ),
                raw_numbers=raw_numbers,
            )
        elif min_delta > self.grounded_threshold:
            sev = Severity.PASS
            finding = Finding(
                observed=(
                    f"Masking '{target_label}' shifted the model's action "
                    f"by at least {min_delta:.2f} of typical magnitude "
                    f"across both fill modes ({cam_phrase})."
                ),
                meaning=(
                    "The model is actively reading the target from the "
                    "image — it does not have a memorized backup for "
                    "this scene."
                ),
                next_step=(
                    "Visually grounded on this frame; no action needed."
                ),
                raw_numbers=raw_numbers,
            )
        elif max_delta < self.grounded_threshold:
            sev = Severity.MODERATE
            finding = Finding(
                observed=(
                    f"Masking '{target_label}' on {cam_phrase} shifted "
                    f"the action by {min_delta:.2f}–{max_delta:.2f} of "
                    f"typical magnitude across fills — between our "
                    f"'memorized' (<{self.noise_floor_score:.0%}) and "
                    f"'grounded' (>{self.grounded_threshold:.0%}) bands."
                ),
                meaning=(
                    "The model uses some visual cues from the target "
                    "but not decisively. Mixed grounding."
                ),
                next_step=(
                    "Check more frames in the same episode. If the "
                    "verdict stays in this middle band, the policy is "
                    "partially grounding; if some frames flip to "
                    "'grounded' and others to 'memorized', the response "
                    "depends on the phase of the trajectory."
                ),
                raw_numbers=raw_numbers,
            )
        else:
            sev = Severity.MODERATE
            strongest = max(
                per_fill_results,
                key=lambda k: per_fill_results[k]["normalized_delta"],
            )
            finding = Finding(
                observed=(
                    f"The two fill modes disagree: "
                    f"Δaction ranges {min_delta:.2f}–{max_delta:.2f} of "
                    f"typical magnitude. '{strongest}' produced the "
                    f"larger response."
                ),
                meaning=(
                    "Suggests fill-specific artifacts rather than a "
                    "clean visual-grounding signal. The model may be "
                    "responding to colour statistics rather than to "
                    "object shape."
                ),
                next_step=(
                    "Look at the per-fill masked images in the Rerun "
                    "export — confirm both fills look like the same "
                    "intervention (similar contrast, similar coverage)."
                ),
                raw_numbers=raw_numbers,
            )
        # Disclose any in-scope camera left unmasked: if the target stays
        # visible to the model there, the verdict understates memorization.
        if unmasked_cams:
            finding = replace(finding, next_step=(
                f"{finding.next_step} Masked on {detected_cams}, not on "
                f"{unmasked_cams} (target not located there); set "
                "analysis.memorization_require_cameras='all' to require every view."
            ))
        # Legacy single-string explanation for backward compat with code
        # paths that still read DiagnosticResult.explanation.
        verdict = f"{finding.observed} {finding.meaning} {finding.next_step}"

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=mean_delta,
            severity=sev,
            direction="lower_is_worse",
            explanation=verdict,
            finding=finding,
            per_variant={
                "mean_delta_across_fills": mean_delta,
                "min_delta_across_fills":  min_delta,
                "max_delta_across_fills":  max_delta,
                "mean_contrast_across_fills": max_contrast,
                **{
                    f"fill:{fm}:normalized_delta": r["normalized_delta"]
                    for fm, r in per_fill_results.items()
                },
                **{
                    f"fill:{fm}:mean_contrast": r["mean_contrast"]
                    for fm, r in per_fill_results.items()
                },
                **{f"detected:{c}": 1.0 for c in detected_cams},
            },
            raw={
                "baseline_action":      baseline.action.tolist(),
                "per_fill":             per_fill_results,
                "calibration_used":     self.calibration.to_summary() if self.calibration else None,
                "min_mask_contrast":    self.min_mask_contrast,
                "noise_floor_score":    self.noise_floor_score,
                "grounded_threshold":   self.grounded_threshold,
                "detected_cameras":     detected_cams,
                "skipped_cameras":      sorted(
                    c for c in cameras if c not in per_cam_original
                ),
                "per_camera_detection": {
                    c: {
                        "label":      per_cam_detection[c].label,
                        "bbox":       list(per_cam_detection[c].bbox),
                        "confidence": per_cam_detection[c].confidence,
                    } for c in detected_cams
                },
            },
        )
