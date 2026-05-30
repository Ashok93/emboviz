"""Target detection — the shared "where is the manipulated object" interface.

Multiple diagnostics need to know where the target object is in the frame:
  • MemorizationDiagnostic — to mask the target and check if action still moves
  • ObjectRecolorPerturber — to recolor the target via segmentation mask
  • Future: per-target sensitivity maps, attention-target alignment, etc.

Detection lives behind a ``TargetDetector`` protocol so users can:
  1. Pass an explicit bbox or pre-computed annotation map
  2. Plug in their own fine-tuned detector (custom DINO / SAM / YOLO)
  3. Use the bundled SAM 3 zero-shot pipeline (text → mask)
  4. Use the legacy GroundingDINO + SAM combo (kept for backward compat)

Honest principle: if a detector cannot locate the target with confidence,
it returns ``None`` and the calling diagnostic skips that frame with a
clear reason. We never default to a "centered bbox" hack — that's silently
wrong (it might mask the gripper, the table, or empty space) and
contaminates downstream verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np

from emboviz.core.types import Scene


# ----------------------------------------------------------------------
# Result + protocol
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TargetDetection:
    """A target localization result.

    ``bbox`` is ``(x0, y0, x1, y1)`` in pixel coordinates of the image the
    detector was queried on. ``mask`` is an optional binary ``HxW`` array
    (``True`` = target pixels) — populated by detectors that produce
    segmentation (SAM, SAM 3); bbox-only detectors leave it ``None`` and
    callers that require a mask raise.

    ``label`` is what the detector thought it found (for debugging /
    logging). ``confidence`` is the detector's score in ``[0, 1]`` if
    available.

    When the detector localizes MULTIPLE instances of the phrase (e.g.
    every spoon in a drawer), ``mask`` is the UNION over instances and
    ``all_boxes`` / ``all_scores`` hold the per-instance metadata for
    display + transparency. ``bbox`` is then the union's bounding box.
    """

    bbox: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None
    label: str = ""
    confidence: float = 1.0
    all_boxes: Optional[list[tuple[int, int, int, int]]] = None
    all_scores: Optional[list[float]] = None


class TargetDetector(Protocol):
    """A callable that locates a target in a ``Scene``.

    Implementations must return ``None`` if they cannot find the target
    with acceptable confidence — diagnostics treat ``None`` as "skip with
    reason" rather than fabricating an answer.

    The contract: the detector inspects ``scene.observations.images["primary"]``
    (multi-camera diagnostics build per-camera probe scenes that alias the
    target camera under ``"primary"``). Implementations that need a
    per-camera annotation key off ``scene.metadata["_emboviz_probe_camera"]``
    (set by the runner when constructing the probe scene).
    """

    def __call__(self, scene: Scene) -> Optional[TargetDetection]: ...


# ----------------------------------------------------------------------
# Trivial detectors (no AI in the loop)
# ----------------------------------------------------------------------

class BBoxDetector:
    """Trivial detector that returns a fixed user-supplied bbox.

    Use when your tracking system already knows where the target is and
    it is roughly stationary across frames (motion capture with a static
    target, fiducials, single-frame debugging).
    """

    def __init__(self, bbox: tuple[int, int, int, int], label: str = "user_supplied"):
        self._bbox = bbox
        self._label = label

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        return TargetDetection(bbox=self._bbox, label=self._label, confidence=1.0)


class CachingTargetDetector:
    """Wraps any ``TargetDetector`` and memoizes its results.

    Memorization runs detection inside the diagnostic AND the runner
    re-uses the same masks for the Rerun overlay. Without caching we'd
    pay the GD+SAM (or SAM 3) cost twice per camera per frame.

    The cache key is ``(scene.scene_id, probe_camera)`` so per-camera
    probe scenes built by the runner each hit a distinct slot. Probe
    cameras are read from ``scene.metadata["_emboviz_probe_camera"]``
    when present; otherwise we key off ``"primary"``.

    All cache lookups are by IDENTITY of scene_id + camera, never by
    image content. Re-running on the same scene returns the cached
    detection — which is exactly what we want when the same physical
    frame is queried twice by different consumers.
    """

    def __init__(self, base: TargetDetector):
        self._base = base
        self._cache: dict[tuple[str, str], Optional[TargetDetection]] = {}

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        cam = scene.metadata.get("_emboviz_probe_camera", "primary")
        key = (scene.scene_id, str(cam))
        if key in self._cache:
            return self._cache[key]
        det = self._base(scene)
        self._cache[key] = det
        return det

    def lookup(self, scene_id: str, camera: str) -> Optional[TargetDetection]:
        """Read-only access for callers that already ran detection.

        Returns the cached detection if present, ``None`` if no entry
        exists (NOT the same as "we detected nothing" — that case stores
        ``None`` against the key, also returned here). The runner uses
        this to avoid building probe scenes when it only needs to read
        the cached result.
        """
        return self._cache.get((str(scene_id), str(camera)))

    def clear(self) -> None:
        """Drop all cached detections (e.g. between episodes)."""
        self._cache.clear()


# ----------------------------------------------------------------------
# User-supplied annotation connectors (task 3)
# ----------------------------------------------------------------------

def _ensure_bbox(value: Any) -> tuple[int, int, int, int]:
    """Coerce a 4-element annotation to ``(x0, y0, x1, y1)`` ints."""
    if isinstance(value, dict):
        # Permissive: accept {"x0":..,"y0":..,"x1":..,"y1":..} or
        # {"xmin":..,"ymin":..,"xmax":..,"ymax":..}.
        for keys in (("x0", "y0", "x1", "y1"),
                     ("xmin", "ymin", "xmax", "ymax")):
            if all(k in value for k in keys):
                return tuple(int(value[k]) for k in keys)  # type: ignore[return-value]
        raise ValueError(
            f"bbox dict {value!r} lacks (x0,y0,x1,y1) or (xmin,ymin,xmax,ymax)"
        )
    seq = list(value)
    if len(seq) != 4:
        raise ValueError(
            f"bbox must have 4 numbers (x0,y0,x1,y1); got {len(seq)}: {value!r}"
        )
    return tuple(int(round(float(v))) for v in seq)  # type: ignore[return-value]


def _rectangular_mask_from_bbox(
    scene: Scene, camera: str, bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Rasterize a bbox into a HxW rectangular mask at the scene's image size.

    Used when an annotation source only provides a bbox (mocap, fiducials,
    a tracker that outputs boxes). The memorization diagnostic needs a
    pixel mask to fill; a rectangular mask is the honest "we only know
    the bounding box" answer — coarser than a segmentation but never
    fabricated. The image shape is read from the per-camera RGB so the
    mask matches whatever the model actually consumes.
    """
    if camera not in scene.observations.images:
        raise ValueError(
            f"_rectangular_mask_from_bbox: scene has no camera "
            f"'{camera}' (has {sorted(scene.observations.images)}). "
            "The connector cannot rasterize the bbox without the image "
            "the annotation refers to."
        )
    img_data = scene.observations.images[camera].data
    arr = np.asarray(img_data)
    if arr.ndim < 2:
        raise ValueError(
            f"_rectangular_mask_from_bbox: image for camera '{camera}' "
            f"has unexpected shape {arr.shape}; need at least HxW."
        )
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(W, int(x0)))
    y0 = max(0, min(H, int(y0)))
    x1 = max(0, min(W, int(x1)))
    y1 = max(0, min(H, int(y1)))
    if x1 <= x0 or y1 <= y0:
        # Degenerate / inverted bbox — return an empty mask, the
        # caller's intervention-validity gate will then skip the frame.
        return np.zeros((H, W), dtype=bool)
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _ensure_mask(value: Any, image_hw: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Coerce a mask payload to a boolean ``HxW`` ndarray.

    Accepted:
      • numpy boolean array (HxW) — used as-is.
      • numpy non-bool array — non-zero is True.
      • nested Python list of 0/1 — converted.
      • dict with ``{"rle": "...", "size": [H, W]}`` for COCO-style
        run-length encoding (uncompressed; one of the two common
        COCO RLE formats). For compressed COCO RLE we require
        ``pycocotools``.
    """
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            return value
        return value.astype(bool)
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        if arr.dtype == object:
            raise ValueError(
                f"mask list could not be converted to a numeric array "
                f"(jagged rows?); got shape inference failure on {arr.shape}"
            )
        return arr.astype(bool)
    if isinstance(value, dict):
        if "rle" in value and "size" in value:
            try:
                from pycocotools import mask as _coco_mask
            except ImportError as e:
                raise ImportError(
                    "Decoding a COCO-format mask requires `pycocotools`. "
                    "Install via: uv pip install pycocotools"
                ) from e
            decoded = _coco_mask.decode(value)
            return decoded.astype(bool)
        if "counts" in value and "size" in value:
            # COCO compressed RLE.
            try:
                from pycocotools import mask as _coco_mask
            except ImportError as e:
                raise ImportError(
                    "Decoding a COCO compressed-RLE mask requires `pycocotools`."
                ) from e
            return _coco_mask.decode(value).astype(bool)
    raise ValueError(
        f"unrecognized mask payload type {type(value).__name__}: {value!r}"
    )


class CallableConnector:
    """Wraps a user-supplied callable ``(scene) -> TargetDetection | None``.

    The simplest possible connector — for users whose annotation source
    is something exotic (custom tracker, live ROS topic, hand-loaded
    pickle). They just provide the function.

    The wrapped callable must follow the ``TargetDetector`` contract:
    return ``None`` when the target cannot be located on this scene.
    """

    def __init__(self, fn: Callable[[Scene], Optional[TargetDetection]]):
        if not callable(fn):
            raise TypeError(
                f"CallableConnector expects a callable; got {type(fn).__name__}"
            )
        self._fn = fn

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        result = self._fn(scene)
        if result is not None and not isinstance(result, TargetDetection):
            raise TypeError(
                f"CallableConnector's wrapped callable must return a "
                f"TargetDetection or None; got {type(result).__name__}"
            )
        return result


class JSONAnnotationConnector:
    """Reads per-frame bboxes / masks from a JSON manifest.

    File schema (one of two equivalent layouts — the connector accepts both):

    A) Frame-major::

        {
          "frames": {
            "<scene_id_or_frame_idx>": {
              "<camera>": {
                "bbox": [x0, y0, x1, y1],
                "label": "the mug",       # optional
                "confidence": 0.95,        # optional
                "mask": [...]              # optional HxW int/bool array
              }
            }
          }
        }

    B) List form::

        {
          "frames": [
            {"frame_id": 42, "camera": "primary", "bbox": [...], "label": "..."},
            ...
          ]
        }

    Lookup keys: the connector matches by ``scene.scene_id`` first, then
    falls back to ``scene.metadata["frame_idx"]`` (the runner sets this
    when constructing trajectory frames). A missing key → ``None`` →
    diagnostic skips that frame with a reason; we never invent a bbox.

    Camera resolution: the per-camera key is the camera name the diagnostic
    is probing (read from ``scene.metadata["_emboviz_probe_camera"]``,
    default ``"primary"``).
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"annotations file not found: {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"could not parse {path} as JSON: {e}"
            ) from e
        self._path = path
        self._by_frame: dict[str, dict[str, dict]] = {}
        self._ingest(data)

    def _ingest(self, data: Any) -> None:
        if not isinstance(data, dict) or "frames" not in data:
            raise ValueError(
                f"JSON annotations expect a top-level dict with a "
                f"'frames' key; got top-level type {type(data).__name__}"
            )
        frames = data["frames"]
        if isinstance(frames, dict):
            # Frame-major: {frame_id_str: {camera: {...}}}.
            for fk, cam_map in frames.items():
                if not isinstance(cam_map, dict):
                    raise ValueError(
                        f"frame entry '{fk}' must be a dict {{camera: ann}}, "
                        f"got {type(cam_map).__name__}"
                    )
                self._by_frame[str(fk)] = cam_map
        elif isinstance(frames, list):
            for row in frames:
                if not isinstance(row, dict):
                    raise ValueError(
                        f"list-form annotation row must be a dict; "
                        f"got {type(row).__name__}"
                    )
                fk = str(row.get("frame_id", row.get("scene_id", row.get("id"))))
                if fk in (None, "None"):
                    raise ValueError(
                        f"list-form row is missing one of frame_id / "
                        f"scene_id / id: {row}"
                    )
                cam = str(row.get("camera", "primary"))
                self._by_frame.setdefault(fk, {})[cam] = row
        else:
            raise ValueError(
                f"'frames' must be a dict or a list; "
                f"got {type(frames).__name__}"
            )

    def _resolve_key(self, scene: Scene) -> Optional[str]:
        # Prefer scene_id (stable across runs); fall back to frame_idx in
        # metadata (set by EpisodeSource adapters). Both are stringified.
        if scene.scene_id and scene.scene_id in self._by_frame:
            return scene.scene_id
        fi = scene.metadata.get("frame_idx")
        if fi is not None and str(fi) in self._by_frame:
            return str(fi)
        return None

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        key = self._resolve_key(scene)
        if key is None:
            return None
        cam = scene.metadata.get("_emboviz_probe_camera", "primary")
        ann = self._by_frame[key].get(str(cam))
        if ann is None:
            return None
        bbox = _ensure_bbox(ann["bbox"]) if "bbox" in ann else None
        mask = _ensure_mask(ann["mask"]) if ann.get("mask") is not None else None
        if bbox is None and mask is not None:
            ys, xs = np.where(mask)
            if not ys.size:
                return None
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if bbox is None:
            raise ValueError(
                f"annotation for scene='{key}' camera='{cam}' has neither "
                f"a 'bbox' nor a 'mask' — at least one must be present"
            )
        if mask is None:
            # Bbox-only annotation: rasterize the rectangle into a mask
            # at the probe image's pixel grid. Memorization needs a mask
            # to know what to fill; a rectangular mask is the honest
            # "we only know the bounding box" answer — coarser than a
            # polygon but never fabricated.
            mask = _rectangular_mask_from_bbox(scene, cam, bbox)
        return TargetDetection(
            bbox=bbox,
            mask=mask,
            label=str(ann.get("label", "user_annotated")),
            confidence=float(ann.get("confidence", 1.0)),
        )


class CocoAnnotationConnector:
    """Reads per-frame bboxes / masks from a COCO-format JSON file.

    COCO schema: top-level dict with ``images`` (list of {id, file_name,
    width, height, ...}) and ``annotations`` (list of {image_id,
    category_id, bbox: [x, y, w, h], segmentation, score, ...}).

    Lookup matches by ``scene.scene_id`` against COCO ``file_name`` first,
    then by ``scene.metadata["frame_idx"]`` against ``image_id``. The
    camera is read from ``scene.metadata["_emboviz_probe_camera"]`` (default
    ``"primary"``) and matched against an optional per-annotation ``camera``
    extra; if no per-annotation camera is set, all annotations are
    considered to apply to every camera (typical for single-camera COCO
    exports).

    When multiple annotations match a frame+camera, the highest-``score``
    one wins. To union multiple instances (every spoon in a drawer), set
    each annotation's ``category_id`` to the same value and the connector
    will pick the strongest.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        category_filter: Optional[str | int] = None,
    ):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"COCO annotations file not found: {path}")
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"could not parse {path} as JSON: {e}") from e
        if not isinstance(doc, dict) or "images" not in doc or "annotations" not in doc:
            raise ValueError(
                f"COCO file must have top-level 'images' and 'annotations' lists; "
                f"got keys {sorted(doc) if isinstance(doc, dict) else type(doc).__name__}"
            )
        self._path = path
        self._categories: dict[int, str] = {
            int(c["id"]): str(c.get("name", c["id"]))
            for c in doc.get("categories", [])
        }
        # Map image_id → {"file_name": ..., "size": (H, W)}
        self._image_meta: dict[int, dict] = {}
        self._image_by_filename: dict[str, int] = {}
        for img in doc["images"]:
            iid = int(img["id"])
            self._image_meta[iid] = {
                "file_name": str(img.get("file_name", "")),
                "size": (int(img.get("height", 0)), int(img.get("width", 0))),
            }
            if img.get("file_name"):
                self._image_by_filename[str(img["file_name"])] = iid
        # Map image_id → list of annotations.
        self._anns_by_image: dict[int, list[dict]] = {}
        self._category_filter = category_filter
        for ann in doc["annotations"]:
            iid = int(ann["image_id"])
            if category_filter is not None:
                cat_id = ann.get("category_id")
                cat_name = self._categories.get(int(cat_id), "") if cat_id is not None else ""
                if isinstance(category_filter, int):
                    if cat_id != category_filter:
                        continue
                else:
                    if cat_name != category_filter:
                        continue
            self._anns_by_image.setdefault(iid, []).append(ann)

    def _resolve_image_id(self, scene: Scene) -> Optional[int]:
        # Try filename match against scene_id first.
        if scene.scene_id in self._image_by_filename:
            return self._image_by_filename[scene.scene_id]
        # Then frame_idx → image_id.
        fi = scene.metadata.get("frame_idx")
        if fi is not None and int(fi) in self._image_meta:
            return int(fi)
        return None

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        iid = self._resolve_image_id(scene)
        if iid is None:
            return None
        anns = self._anns_by_image.get(iid, [])
        if not anns:
            return None
        cam = str(scene.metadata.get("_emboviz_probe_camera", "primary"))
        # Per-annotation camera filter (optional COCO extension we honour
        # when present; absence = matches all cameras).
        matching = [
            a for a in anns
            if "camera" not in a or str(a["camera"]) == cam
        ]
        if not matching:
            return None
        # Highest-scoring annotation wins. COCO uses ``score`` for
        # detection results; predictions without scores default to 1.0.
        matching.sort(key=lambda a: float(a.get("score", 1.0)), reverse=True)
        top = matching[0]
        # COCO bbox is [x, y, w, h]; we want (x0, y0, x1, y1).
        if "bbox" not in top:
            raise ValueError(
                f"COCO annotation for image_id={iid} lacks 'bbox': {top}"
            )
        x, y, w, h = (float(v) for v in top["bbox"])
        bbox = (int(round(x)), int(round(y)),
                int(round(x + w)), int(round(y + h)))
        mask: Optional[np.ndarray] = None
        seg = top.get("segmentation")
        if seg is not None:
            try:
                if isinstance(seg, dict):
                    mask = _ensure_mask(seg)
                elif isinstance(seg, list) and seg and isinstance(seg[0], list):
                    # Polygon segmentation — rasterize via pycocotools.
                    try:
                        from pycocotools import mask as _coco_mask
                    except ImportError as e:
                        raise ImportError(
                            "Polygon COCO segmentation requires `pycocotools`. "
                            "Install via: uv pip install pycocotools"
                        ) from e
                    h_img, w_img = self._image_meta[iid]["size"]
                    if h_img == 0 or w_img == 0:
                        raise ValueError(
                            f"COCO image_id={iid} has zero-sized image metadata; "
                            "cannot rasterize polygon without H/W."
                        )
                    rles = _coco_mask.frPyObjects(seg, h_img, w_img)
                    rle = _coco_mask.merge(rles)
                    mask = _coco_mask.decode(rle).astype(bool)
            except (ValueError, ImportError) as e:
                raise ValueError(
                    f"could not decode COCO segmentation for image_id={iid}: {e}"
                ) from e
        cat_id = top.get("category_id")
        label = self._categories.get(int(cat_id), "coco_object") if cat_id is not None else "coco_object"
        if mask is None:
            # Bbox-only annotation: rasterize the rectangle. See
            # :func:`_rectangular_mask_from_bbox` for the rationale.
            cam = str(scene.metadata.get("_emboviz_probe_camera", "primary"))
            mask = _rectangular_mask_from_bbox(scene, cam, bbox)
        return TargetDetection(
            bbox=bbox,
            mask=mask,
            label=label,
            confidence=float(top.get("score", 1.0)),
        )


def load_annotation_connector(path: str | Path) -> TargetDetector:
    """Auto-detect JSON-vs-COCO and build the right connector.

    Heuristic: a file with top-level ``images`` + ``annotations`` is COCO;
    a file with top-level ``frames`` is our JSONAnnotationConnector schema;
    anything else raises with a clear schema description.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"annotations file not found: {path}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse {path} as JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ValueError(
            f"annotations file must be a JSON object; got "
            f"top-level type {type(doc).__name__}"
        )
    has_coco = {"images", "annotations"}.issubset(doc)
    has_frames = "frames" in doc
    if has_coco and not has_frames:
        return CocoAnnotationConnector(path)
    if has_frames and not has_coco:
        return JSONAnnotationConnector(path)
    if has_coco and has_frames:
        raise ValueError(
            f"{path} contains BOTH a COCO 'images'/'annotations' block AND "
            f"a 'frames' block. Cannot auto-detect format; instantiate "
            f"either CocoAnnotationConnector or JSONAnnotationConnector "
            f"directly."
        )
    raise ValueError(
        f"{path} is JSON but does not match either schema we know:\n"
        f"  • COCO: top-level {{'images': [...], 'annotations': [...]}}\n"
        f"  • Frame-major: top-level {{'frames': {{...}} or [...]}}\n"
        f"Top-level keys present: {sorted(doc)}"
    )


# ----------------------------------------------------------------------
# Zero-shot text-to-mask: SAM 3 (default) and GroundingDINO+SAM (fallback)
# ----------------------------------------------------------------------

class SAM3Detector:
    """Zero-shot open-vocabulary target detection via Meta's SAM 3 (ZMQ client).

    SAM 3 (Meta AI, [released Nov 2025](https://github.com/facebookresearch/sam3))
    is a single model that takes a text phrase and segments every
    instance of that concept in an image:

        text → boxes + masks (one forward pass).

    Architecture: this class is a thin wrapper around
    :class:`emboviz_sam3.client.Sam3Client`. The actual SAM 3 model
    runs in a SEPARATE Python 3.12 venv (the ``emboviz-sam3`` adapter
    worker) and answers ZMQ ``detect`` requests over a Unix socket.

    Why an isolated worker: SAM 3 requires Python 3.12+ and
    ``transformers >= 4.56``. None of the four VLA adapter venvs
    (OpenVLA on Python 3.10 + transformers 4.49, OFT on a vendored
    transformers fork, π0 on Python 3.11 + transformers 4.53, GR00T on
    Python 3.11 + transformers 4.57) can host those constraints
    alongside their pinned adapter deps. ZMQ's wire (bytes / msgpack)
    is Python-version-agnostic so SAM 3 can stay on 3.12 forever.

    Why this is the default over GD+SAM:
      • One model + one forward pass on the worker side.
      • Native concept-aware text prompting (no noun extraction).
      • Better discrimination on close concepts via the presence-token.
      • The worker is shared across multiple runs — model loads once,
        not per-runner-launch.

    Usage::

        # First, start the worker (one-time per session):
        #   ~/.emboviz/venvs/sam3/bin/emboviz-sam3 serve
        # Then in the calling code:
        det = SAM3Detector(target_text="the mug")
        detection = det(scene)
        if detection is not None:
            mask = detection.mask         # HxW bool
            box  = detection.bbox         # (x0,y0,x1,y1)
    """

    def __init__(
        self,
        target_text: str,
        endpoint: Optional[str] = None,
        score_threshold: float = 0.30,
        mask_threshold: float = 0.50,
        timeout: float = 120.0,
        device: str = "cuda",
    ):
        """Args:
            target_text: REQUIRED. The phrase to localize — e.g.
                ``"the mug"``, ``"the lid"``, ``"the welding torch"``,
                ``"the red pipe on the left"``. Memorization is a
                USER-SCOPED test (which object do you want to check
                the policy isn't ignoring?); we never guess.
            endpoint: ZMQ endpoint of the running ``emboviz-sam3``
                worker. Default: read from ``EMBOVIZ_SAM3_ENDPOINT``
                env var, else ``ipc://~/.emboviz/sockets/sam3.sock``.
            score_threshold: detections with the top instance's score
                below this are returned as ``None`` (the diagnostic
                then skips that frame with a clear reason). 0.30 is
                the "high-precision" cutoff from the SAM 3 release
                notes.
            mask_threshold: SAM 3 emits per-pixel mask logits; values
                above this become foreground. 0.50 is standard.
            timeout: per-request RPC timeout in seconds. SAM 3 inference
                is ~100-300 ms per image on H100/A6000; the first
                request to a freshly-started worker pays a ~30 s warmup
                unless the worker was started with ``--preload``.
            device: legacy kwarg kept for signature compatibility; the
                worker picks its own device. Ignored on this side.
        """
        if target_text is None or not str(target_text).strip():
            raise ValueError(
                "SAM3Detector requires a non-empty ``target_text`` — the "
                "phrase to localize. Memorization tests whether the policy "
                "is using vision for a specific object (\"the mug\", "
                "\"the lid\", \"the welding torch\"). We do not guess the "
                "target from the policy's instruction; only the user knows "
                "what their model is supposed to manipulate. Set "
                "target_text=\"<your object>\" when constructing the detector."
            )
        self.target_text = str(target_text).strip()
        self.endpoint = endpoint
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.timeout = float(timeout)
        # Kept for signature compatibility — the worker picks its own
        # device.
        self.device = device
        self._client = None
        self._health_checked = False

    # -- low-level ZMQ helpers -----------------------------------------

    def _zmq(self):
        if self._client is not None:
            return self._client
        try:
            from emboviz_sam3.client import Sam3Client
        except ImportError as e:
            raise ImportError(
                "SAM3Detector requires the ``emboviz-sam3`` adapter "
                "package to be installed (it ships the typed RPC client "
                "alongside the worker code). Install via:\n"
                "    uv pip install emboviz-sam3"
            ) from e
        self._client = Sam3Client(
            endpoint=self.endpoint,
            timeout_ms=int(self.timeout * 1000),
        )
        return self._client

    def _check_health(self) -> None:
        """First-call probe: confirm the worker is reachable and emit a
        clear, actionable error if it isn't. We do not auto-spawn the
        worker — it has its own venv and HF cache and the user should
        be in control of when the 850M-param model loads."""
        if self._health_checked:
            return
        client = self._zmq()
        if not client.ping(timeout_ms=2000):
            raise RuntimeError(
                f"SAM3Detector cannot reach the SAM 3 worker at "
                f"{client._endpoint}.\n\n"
                "Start the worker (in its own Python 3.12 venv):\n"
                "    ~/.emboviz/venvs/sam3/bin/emboviz-sam3 serve\n\n"
                "Or override the endpoint via "
                "EMBOVIZ_SAM3_ENDPOINT=ipc://... or tcp://...\n\n"
                "If the worker isn't installed, run:\n"
                "    uv pip install emboviz-sam3\n"
                "    emboviz install-sam3\n\n"
                "If you need to keep moving without SAM 3, pass\n"
                "    --detector gd-sam\n"
                "to fall back to GroundingDINO + SAM."
            )
        self._health_checked = True

    # -- public detector contract --------------------------------------

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        if "primary" not in scene.observations.images:
            raise ValueError(
                "SAM3Detector expects a 'primary' camera in the scene "
                f"(available: {sorted(scene.observations.images)}). Build "
                "a probe scene that aliases the camera you want to inspect "
                "under the name 'primary'."
            )
        self._check_health()
        pil = scene.observations.images["primary"].data

        # Encode image as PNG bytes once. PNG over JPEG because lossy
        # compression can shift detection slightly; PNG is lossless
        # and SAM 3 worker-side decoding cost is negligible compared
        # to the inference itself.
        import io as _io
        buf = _io.BytesIO()
        pil.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        body = self._zmq().detect(
            image_bytes=img_bytes,
            target_text=self.target_text,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
        )
        instances = body.get("instances") or []
        if not instances:
            return None
        return self._build_detection(instances, body)

    def _build_detection(
        self, instances: list[dict], body: dict,
    ) -> TargetDetection:
        """Union the per-instance masks (already raw uint8 ndarrays
        from the wire) into a single :class:`TargetDetection`.

        Instances arrive sorted highest-score-first (per the worker).
        We keep all above ``score_threshold`` and report each instance's
        bbox + score for transparency.
        """
        kept = [
            i for i in instances
            if float(i.get("score", 0.0)) >= self.score_threshold
        ]
        if not kept:
            return None  # type: ignore[return-value]
        inst_boxes: list[tuple[int, int, int, int]] = []
        inst_scores: list[float] = []
        union: Optional[np.ndarray] = None
        for inst in kept:
            m = inst.get("mask")
            if m is None:
                continue
            m = np.asarray(m).astype(bool)
            if m.ndim == 3 and m.shape[-1] == 1:
                m = m[..., 0]
            if not m.any():
                continue
            bx = inst.get("bbox") or []
            if len(bx) == 4:
                x0, y0, x1, y1 = (int(v) for v in bx)
            else:
                ys, xs = np.where(m)
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
            inst_boxes.append((x0, y0, x1, y1))
            inst_scores.append(float(inst.get("score", 0.0)))
            union = m if union is None else (union | m)
        if union is None or not union.any():
            return None  # type: ignore[return-value]
        ys, xs = np.where(union)
        union_bbox = (
            int(xs.min()), int(ys.min()),
            int(xs.max()), int(ys.max()),
        )
        n = len(inst_boxes)
        target_text = str(body.get("label", self.target_text))
        label = target_text if n == 1 else f"{target_text} ×{n}"
        return TargetDetection(
            bbox=union_bbox,
            mask=union,
            label=label,
            confidence=max(inst_scores),
            all_boxes=inst_boxes,
            all_scores=inst_scores,
        )


class GroundingDINOSAMDetector:
    """Two-stage open-vocabulary detection: GroundingDINO bbox + SAM mask.

    Kept as a maintained fallback for environments where SAM 3 isn't
    available yet (transformers < 4.50, gated checkpoint not accepted,
    user already has GD+SAM in their venv). Prefer :class:`SAM3Detector`
    for new code — single model, native concept prompting, better recall.

    Pipeline (per LITERATURE.md §1):
      1. ``target_text`` (required) is the GroundingDINO query phrase.
      2. GroundingDINO returns box(es) + scores.
      3. SAM refines the top boxes → pixel-accurate masks.
      4. We mask EVERY instance above ``min_confidence`` (e.g. every
         spoon in a drawer of spoons) and union the per-instance masks.
      5. Return ``TargetDetection`` or ``None`` (low confidence / no
         detection — diagnostics then skip that frame with a reason).

    No fallbacks to bbox-only when SAM fails — that produces too-coarse
    masks (target + background) and gives uninterpretable verdicts. We
    raise rather than degrade silently.
    """

    def __init__(
        self,
        target_text: str,
        gd_repo: str = "IDEA-Research/grounding-dino-tiny",
        sam_repo: str = "facebook/sam-vit-base",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        min_confidence: float = 0.25,
        device: str = "cuda",
    ):
        if target_text is None or not str(target_text).strip():
            raise ValueError(
                "GroundingDINOSAMDetector requires a non-empty "
                "``target_text`` at construction. This is the phrase to "
                "mask — the user must say what their policy is supposed "
                "to manipulate (e.g. \"the mug\")."
            )
        self.gd_repo = gd_repo
        self.sam_repo = sam_repo
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.min_confidence = float(min_confidence)
        self.device = device
        self.target_text = str(target_text).strip()
        self._gd = None  # (processor, model)
        self._sam = None

    def _ensure_loaded(self) -> None:
        if self._gd is None:
            try:
                import torch  # noqa: F401
                from transformers import (
                    AutoModelForZeroShotObjectDetection,
                    AutoProcessor,
                )
            except ImportError as e:
                raise ImportError(
                    "GroundingDINOSAMDetector requires `transformers`. "
                    "Install via your model adapter's optional deps."
                ) from e
            proc = AutoProcessor.from_pretrained(self.gd_repo)
            model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(self.gd_repo)
                .to(self.device).eval()
            )
            self._gd = (proc, model)
        if self._sam is None:
            try:
                from transformers import SamModel, SamProcessor
                sam_proc = SamProcessor.from_pretrained(self.sam_repo)
                sam_model = SamModel.from_pretrained(self.sam_repo).to(self.device).eval()
                self._sam = (sam_proc, sam_model)
            except (ImportError, OSError, RuntimeError) as e:
                raise RuntimeError(
                    f"GroundingDINOSAMDetector requires SAM "
                    f"({self.sam_repo}) but it failed to load: "
                    f"{type(e).__name__}: {e}. SAM provides the pixel-"
                    "accurate masks the memorization diagnostic needs — "
                    "bbox-only masking is too coarse and produces "
                    "uninterpretable verdicts. Install the SAM checkpoint, "
                    "switch to SAM3Detector, or pass a different "
                    "target_detector."
                ) from e

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        import inspect

        import torch

        if "primary" not in scene.observations.images:
            raise ValueError(
                "GroundingDINOSAMDetector expects a 'primary' camera in "
                f"the scene (available: {sorted(scene.observations.images)}). "
                "Build a probe scene that aliases the camera you want to "
                "inspect under the name 'primary'."
            )

        self._ensure_loaded()
        assert self._gd is not None and self._sam is not None
        proc, model = self._gd
        pil = scene.observations.images["primary"].data
        text = self.target_text if self.target_text.endswith(".") else f"{self.target_text}."
        inputs = proc(images=pil, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]]).to(self.device)
        sig_params = inspect.signature(
            proc.post_process_grounded_object_detection
        ).parameters
        if "box_threshold" in sig_params:
            thresh_kwargs = {
                "box_threshold":  self.box_threshold,
                "text_threshold": self.text_threshold,
            }
        elif "threshold" in sig_params:
            thresh_kwargs = {
                "threshold":      self.box_threshold,
                "text_threshold": self.text_threshold,
            }
        else:
            raise RuntimeError(
                "GroundingDinoProcessor.post_process_grounded_object_detection "
                f"signature does not accept either 'box_threshold' or "
                f"'threshold'. Got params: {list(sig_params)}. transformers "
                "may have changed the API again — update the call site."
            )
        results = proc.post_process_grounded_object_detection(
            outputs, input_ids=inputs["input_ids"],
            target_sizes=target_sizes,
            **thresh_kwargs,
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        if boxes.size == 0:
            return None
        keep = [
            i for i in range(len(scores))
            if float(scores[i]) >= self.min_confidence
        ]
        if not keep:
            return None
        inst_boxes = [tuple(int(v) for v in boxes[i].astype(int)) for i in keep]
        inst_scores = [float(scores[i]) for i in keep]

        sam_proc, sam_model = self._sam
        sam_inputs = sam_proc(
            pil, input_boxes=[[list(b) for b in inst_boxes]], return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            sam_out = sam_model(**sam_inputs, multimask_output=False)
        masks = sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )
        if not masks or len(masks) == 0 or masks[0].shape[0] == 0:
            raise RuntimeError(
                f"SAM returned no mask for {len(inst_boxes)} box(es) on a "
                f"phrase GroundingDINO scored up to {max(inst_scores):.3f}. "
                "Likely a SAM preprocessing edge case — investigate rather "
                "than fall back to bbox-only."
            )
        per_instance = masks[0]  # (N, 1, H, W)
        union: Optional[np.ndarray] = None
        for k in range(per_instance.shape[0]):
            m = per_instance[k][0].numpy().astype(bool)
            union = m if union is None else (union | m)
        if union is None or not union.any():
            return None
        ys, xs = np.where(union)
        union_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        n = len(inst_boxes)
        label = self.target_text if n == 1 else f"{self.target_text} ×{n}"
        return TargetDetection(
            bbox=union_bbox, mask=union, label=label,
            confidence=max(inst_scores),
            all_boxes=inst_boxes, all_scores=inst_scores,
        )


__all__ = [
    "TargetDetection",
    "TargetDetector",
    "BBoxDetector",
    "CachingTargetDetector",
    "CallableConnector",
    "JSONAnnotationConnector",
    "CocoAnnotationConnector",
    "load_annotation_connector",
    "SAM3Detector",
    "GroundingDINOSAMDetector",
]
