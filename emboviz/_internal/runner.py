"""Generalized multi-camera trajectory diagnostic runner.

Runs the full diagnostic suite on one (model, episode) pair, with the
post-audit strict contract: per-camera attention / sensitivity / target
overlays in Rerun; per-camera modality dropout; no silent primary-only
fallback.

Efficiency contract (refactor 2026-05):

  • One ``averaged_predict`` baseline per frame, shared across every
    diagnostic that needs an unperturbed prediction. Saves
    ``n_samples × num_diagnostics`` worth of model forward passes per
    frame on stochastic models.

  • One ``extract_attention`` call per frame, shared between the
    attention-drift trajectory diagnostic and the per-camera Rerun
    overlay. The full attention tensor is downsampled to per-camera
    ``image_weights_clean`` and the raw tensor freed immediately so the
    (layers × heads × seq²) allocation never accumulates.

  • One target detection per (scene, camera) via
    ``CachingTargetDetector``. The memorization diagnostic populates
    the cache; the Rerun-overlay collection reads back from it. SAM 3
    (or GroundingDINO + SAM, when SAM 3 isn't available) runs once per
    camera per frame, never twice.

  • Periodic ``gc.collect()`` + ``torch.cuda.empty_cache()`` between
    frames so the per-frame attention / detection tensors are reclaimed
    before the next frame's allocations push GPU memory toward OOM.

  • Optional pre-flight cost estimate via ``--dry-run``: prints the
    per-frame forward-pass count and total estimated wall time so users
    can size their run before committing GPU hours.

Invocation (normally via ``emboviz analyze --config``, which fills these
in from the run config; shown here for reference):

    uv run python -m emboviz._internal.runner \\
        --story-id     openvla:bridge:ep0 \\
        --model-builder adapter:openvla \\
        --dataset-builder emboviz.datasets.manifest:build_source \\
        --episode-idx 0 \\
        --frame-start 8 --n-frames 8 \\
        --out-dir /tmp/itest/openvla/bridge_ep0

The runner deliberately stays slim: it imports a builder, asks the
dataset for a trajectory, runs the diagnostic suite, and writes
``summary.json`` + ``rollout.rrd``. All Tier 1 + Tier 2 + post-audit
fixes are exercised.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from emboviz.calibration import averaged_predict, calibrate_model
from emboviz.core.results import Severity
from emboviz.core.types import ActionResult, TokenSelector
from emboviz.diagnostics.attention_drift import AttentionDriftDiagnostic
from emboviz.diagnostics.chunk_consistency import ChunkConsistencyDiagnostic
from emboviz.diagnostics.memorization import (
    DEFAULT_FILL_MODES,
    KNOWN_FILL_MODES,
    LAMA_INPAINT_FILL,
    MemorizationDiagnostic,
)
from emboviz.diagnostics.modality_dropout import ModalityDropoutDiagnostic
from emboviz.diagnostics.sensitivity_map import SensitivityMapDiagnostic
from emboviz.diagnostics.trajectory import TrajectoryDiagnostic
from emboviz.exporters.correlation import find_failure_moments, format_failure_moments
from emboviz.exporters.rerun import export_rerun
from emboviz.models.protocol import Capability
from emboviz.perturb._target_detection import (
    CachingTargetDetector,
    GroundingDINOSAMDetector,
    SAM3Detector,
    TargetDetector,
    load_annotation_connector,
)
from emboviz.perturb.image._inpaint import CachingInpainter, LamaInpainter
from emboviz.perturb.instruction import PromptParaphrasePerturber


# Frequency (in frames) at which we trigger garbage collection +
# ``torch.cuda.empty_cache``. Cheap relative to a model forward pass;
# critical to keep per-frame intermediate tensors from accumulating
# across long episodes.
_GC_EVERY_N_FRAMES = 5


def _resolve(spec: str, kwargs_json: str = ""):
    """Resolve a model / dataset spec into a callable instance.

    Three forms are supported:

      1. ``adapter:<name>`` — connect to (or auto-spawn) the ZeroMQ
         worker for the installed emboviz adapter package registered
         under ``<name>`` (e.g. ``adapter:openvla``). Returns the
         live :class:`ZMQAdapterClient` (a VLAModel) — the actual
         model lives in the adapter's isolated runtime venv on the
         other side of a Unix socket.

      2. ``module.path:attr`` — call ``attr(**kwargs)``. Used by
         dataset adapters and the built-in mock / lerobot model.

      3. ``module.path:attr:arg`` — call ``attr(arg)`` first; if it
         returns a class, then call ``cls(**kwargs)``. Used by the
         (now-deprecated) ``emboviz.models.registry:get_model:<name>``
         resolver path; kept for back-compat until the legacy in-
         process adapters are removed.

    The ``kwargs_json`` blob for ``adapter:<name>`` carries the user's
    per-run constructor overrides (their fine-tuned checkpoint, unnorm
    key, etc.). It is merged over the spec's ``default_actor_kwargs`` and
    forwarded to the worker at spawn time (``serve --kwargs``), so the
    worker loads exactly the model the config names.
    """
    kwargs = json.loads(kwargs_json) if kwargs_json else {}

    # ── Form 1: adapter:<name> ──────────────────────────────────────
    if spec.startswith("adapter:"):
        from emboviz.adapters import connect
        name = spec.split(":", 1)[1]
        # ``--model-kwargs`` (parsed into ``kwargs`` above) are the user's
        # per-run constructor overrides — most importantly THEIR
        # fine-tuned ``checkpoint``. They're merged over the spec's
        # ``default_actor_kwargs`` and forwarded into the worker at spawn
        # time, so the worker loads the user's model, not the demo default.
        handle = connect(name, actor_kwargs=kwargs or None)
        # We return the client only. The Popen handle (if we spawned)
        # is intentionally left attached to the running worker so the
        # next CLI invocation in the same session re-uses it. The OS
        # reaps it when the user kills the worker or the machine exits.
        return handle.client

    parts = spec.split(":")
    module = importlib.import_module(parts[0])
    obj = getattr(module, parts[1])
    if len(parts) == 2:
        return obj(**kwargs)
    intermediate = obj(parts[2])
    if isinstance(intermediate, type):
        return intermediate(**kwargs)
    return intermediate


def _build_target_detector(args) -> tuple[Optional[CachingTargetDetector], Optional[str]]:
    """Build the (cached) target detector based on CLI flags.

    Priority order — first non-empty source wins:

      1. ``--target-annotations <path>``: user-supplied per-frame
         bbox/mask manifest (JSON or COCO). No AI in the loop.
      2. ``--target-text <phrase>`` + ``--detector <sam3|gd-sam>``:
         text-to-mask via the chosen detector. SAM 3 by default.
      3. Neither supplied: returns ``(None, reason)`` — memorization
         is skipped at the runner level with a clear reason.

    Returns ``(detector, skip_reason)``. When ``detector`` is not None,
    ``skip_reason`` is None (and vice versa). The detector is always
    wrapped in a ``CachingTargetDetector`` so re-queries by the
    Rerun-overlay collection don't pay another forward pass.
    """
    ann_path = (args.target_annotations or "").strip() if args.target_annotations else ""
    if ann_path:
        connector = load_annotation_connector(ann_path)
        cached = CachingTargetDetector(connector)
        print(
            f"      memorization detector = "
            f"{type(connector).__name__}(path={ann_path!r})",
            flush=True,
        )
        return cached, None

    text = (args.target_text or "").strip()
    if not text:
        return None, (
            "no --target-text and no --target-annotations supplied. "
            "Memorization tests whether the policy uses vision for a "
            "specific object; the user must say which (\"the mug\", "
            "\"the lid\", \"the welding torch\", ...) or hand us per-frame "
            "annotations. Re-run with --target-text \"<phrase>\" or "
            "--target-annotations <path>."
        )
    detector_kind = (args.detector or "sam3").lower()
    if detector_kind == "sam3":
        # The detector is a client to the isolated SAM 3 worker (its own
        # py3.12 venv). Spawn it the SAME way the runner brings up the
        # model and dataset-reader workers (connect → auto-install the
        # runtime venv if missing, auto-spawn, wait for ping) so that
        # ``emboviz analyze --config`` is self-contained — no separate
        # ``emboviz-sam3 serve`` step. ``connect`` attaches to a warm
        # worker if one is already running (no kwargs → no conflict). We
        # only need the worker UP; SAM3Detector opens its own client.
        from emboviz.adapters import connect
        connect("sam3", auto_spawn=True, auto_install=True)
        base: TargetDetector = SAM3Detector(target_text=text, device="cuda")
        print(
            f"      memorization detector = "
            f"SAM3Detector(target_text={text!r})",
            flush=True,
        )
    elif detector_kind in ("gd-sam", "groundingdino", "gd_sam", "gd"):
        base = GroundingDINOSAMDetector(target_text=text, device="cuda")
        print(
            f"      memorization detector = "
            f"GroundingDINOSAMDetector(target_text={text!r}) "
            f"[fallback — prefer SAM 3]",
            flush=True,
        )
    else:
        raise ValueError(
            f"--detector must be 'sam3' or 'gd-sam'; got {detector_kind!r}"
        )
    return CachingTargetDetector(base), None


def _resolve_fills(args) -> list[str]:
    """Normalise ``args.fills`` (a list, or a comma string from the CLI)
    into a deduped, lowercased, validated fill-mode list.

    Defaults to the two pure fills (``channel_mean``, ``gaussian_blur``)
    when unset. ``lama_inpaint`` is accepted but pulls in the emboviz-lama
    worker (wired by :func:`_build_inpainter`)."""
    raw = getattr(args, "fills", None)
    if raw is None:
        return list(DEFAULT_FILL_MODES)
    if isinstance(raw, str):
        raw = [tok for tok in raw.split(",")]
    fills: list[str] = []
    for tok in raw:
        f = str(tok).strip().lower()
        if not f or f in fills:
            continue
        if f not in KNOWN_FILL_MODES:
            raise ValueError(
                f"unknown fill mode {f!r} in analysis.fills; supported: "
                f"{sorted(KNOWN_FILL_MODES)}."
            )
        fills.append(f)
    return fills or list(DEFAULT_FILL_MODES)


def _build_inpainter(fills: list[str]) -> Optional[CachingInpainter]:
    """Build the (cached) LaMa inpainter iff the on-manifold fill is
    requested, bringing up the ``emboviz-lama`` worker the SAME way the
    runner brings up SAM 3 (connect → auto-install the runtime venv if
    missing → auto-spawn → wait for ping). Returns ``None`` when
    ``lama_inpaint`` isn't in the fills (the pure fills need no worker)."""
    if LAMA_INPAINT_FILL not in fills:
        return None
    from emboviz.adapters import connect
    connect("lama", auto_spawn=True, auto_install=True)
    print(
        f"      memorization on-manifold fill = LaMa inpainting "
        f"(fills={fills})",
        flush=True,
    )
    return CachingInpainter(LamaInpainter())


def _maybe_collect(i: int) -> None:
    """Periodic GC + CUDA cache release.

    Called once every ``_GC_EVERY_N_FRAMES`` per-frame iterations. The
    cost is negligible (Python GC on a few hundred references + an
    empty_cache call) and it prevents the runner from accumulating
    per-frame attention / detection tensors on the GPU.
    """
    if (i + 1) % _GC_EVERY_N_FRAMES != 0:
        return
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _estimate_cost(args, model, trajectory) -> dict:
    """Estimate per-frame and per-episode model-forward-pass count.

    Used by ``--dry-run`` to give the user a budget number BEFORE they
    commit GPU hours. Uses the model's calibrated ``n_samples`` and the
    declared modality count (which dominates the cost on stochastic
    multi-camera models).
    """
    calibration = calibrate_model(model, trajectory, n_noise_probes=3)
    n_s = calibration.n_samples
    n_cam = len(model.required_inputs.cameras)
    n_mods_non_image = sum(
        bool(getattr(model.required_inputs, f))
        for f in ("state", "gripper", "action_history", "instruction")
    )
    grid = int(args.sensitivity_grid_side)
    k_md = int(args.modality_k_samples)
    has_target = bool(
        (args.target_text and args.target_text.strip())
        or (args.target_annotations and args.target_annotations.strip())
    )
    # Per frame, distinct model forward passes (NOT counting calibration):
    #   1 shared baseline (averaged over n_samples)
    #   memorization: N_FILLS perturbed predictions (only if target given);
    #                 N_FILLS = len(analysis.fills) (the LaMa inpaint that
    #                 builds a fill is a separate cheap worker call, not a
    #                 VLA forward, so it isn't counted here)
    #   modality_dropout: (n_mods_non_image + n_cam) * k_md perturbed preds
    #   sensitivity:     n_cam * grid² perturbed preds
    #   chunk_consistency: shares baseline → 0 extra
    #   attention_drift:  1 extract_attention call (shared with overlay)
    n_fills = len(_resolve_fills(args))
    per_frame_baseline = n_s
    per_frame_memo = n_fills * n_s if has_target else 0
    per_frame_modality = (n_mods_non_image + n_cam) * k_md * n_s
    per_frame_sensitivity = n_cam * grid * grid * n_s
    per_frame_attention = 1
    per_frame_total = (
        per_frame_baseline + per_frame_memo + per_frame_modality
        + per_frame_sensitivity + per_frame_attention
    )
    n_frames = len(trajectory.frames)
    total = per_frame_total * n_frames
    return {
        "n_samples":              n_s,
        "n_cameras":              n_cam,
        "n_modalities_non_image": n_mods_non_image,
        "sensitivity_grid":       grid,
        "modality_k_samples":     k_md,
        "has_target":             has_target,
        "n_frames":               n_frames,
        "per_frame": {
            "baseline":     per_frame_baseline,
            "memorization": per_frame_memo,
            "modality":     per_frame_modality,
            "sensitivity":  per_frame_sensitivity,
            "attention":    per_frame_attention,
            "total":        per_frame_total,
        },
        "episode_total_forward_passes": total,
    }


def run_story(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[runner] story={args.story_id}", flush=True)
    print(f"[runner] out_dir={out_dir}", flush=True)

    # Diagnostic gating. The CLI resolves --diagnostics + --skip-diagnostics
    # into ``args.enabled_diagnostics`` (a frozenset of canonical axis
    # names like ``"vision.memorization"``). Older callers that don't set
    # it default to "all enabled".
    enabled = frozenset(getattr(args, "enabled_diagnostics", None) or {
        "vision.memorization",
        "input.modality_dropout",
        "vision.scene_sensitivity",
        "internal.chunk_consistency",
        "internal.attention_drift",
    })

    # --- 1. model + dataset ----------------------------------------------
    print(f"[1/7] load model: {args.model_builder} kwargs={args.model_kwargs_json}", flush=True)
    model = _resolve(args.model_builder, args.model_kwargs_json)
    print(f"      model_id={model.model_id}  caps={model.capabilities}", flush=True)
    print(f"      required_inputs.cameras={sorted(model.required_inputs.cameras)}", flush=True)

    print(f"[2/7] load dataset: {args.dataset_builder} kwargs={args.dataset_kwargs_json}", flush=True)
    dataset = _resolve(args.dataset_builder, args.dataset_kwargs_json)
    full_traj = dataset.load_trajectory(int(args.episode_idx))

    # Window construction.
    n_total = len(full_traj.frames)
    start = max(0, int(args.frame_start))
    n_req = int(args.n_frames)
    stride = max(1, int(getattr(args, "frame_stride", 1)))
    end = n_total if n_req <= 0 else min(n_total, start + n_req * stride)
    sel = list(range(start, end, stride))
    if not sel:
        raise ValueError(
            f"empty trajectory window for episode {args.episode_idx}: "
            f"frame_start={start} n_frames={n_req} stride={stride} "
            f"out of {n_total} total frames."
        )
    window_frames  = [full_traj.frames[i] for i in sel]
    window_indices = [full_traj.frame_indices[i] for i in sel]
    trajectory = replace(full_traj, frames=window_frames, frame_indices=window_indices)
    print(f"      trajectory: {len(trajectory.frames)} frames "
          f"(stride={stride}, episode has {n_total} total) "
          f"source={trajectory.source}", flush=True)
    print(f"      cameras in scene: {sorted(trajectory.frames[0].observations.images)}", flush=True)
    print(f'      instruction: "{trajectory.frames[0].instruction}"', flush=True)

    # --- 1.5 dry-run cost estimate ---------------------------------------
    if getattr(args, "dry_run", False):
        est = _estimate_cost(args, model, trajectory)
        print("\n[dry-run] estimated cost:", flush=True)
        print(json.dumps(est, indent=2), flush=True)
        print(
            f"\n[dry-run] {est['episode_total_forward_passes']:,} model "
            f"forward passes for this episode "
            f"({est['per_frame']['total']:,} per frame × "
            f"{est['n_frames']} frames). Not committing to a real run.",
            flush=True,
        )
        return

    # --- 2. CALIBRATION: noise floor + typical action magnitude ----------
    print("[2b/7] calibrating model on this trajectory "
          "(noise-floor + typical action magnitude)...", flush=True)
    calibration = calibrate_model(model, trajectory, n_noise_probes=5)
    print(f"      noise_floor              = {calibration.noise_floor:.4f}", flush=True)
    print(f"      typical_action_magnitude = {calibration.typical_action_magnitude:.4f}", flush=True)
    print(f"      n_samples (averaging)    = {calibration.n_samples}", flush=True)
    if calibration.single_sample_noise_floor is not None:
        print(f"      (single-sample noise floor was "
              f"{calibration.single_sample_noise_floor:.4f}; "
              f"averaging reduces it to {calibration.noise_floor:.4f})", flush=True)
    print("      → diagnostic scores reported on a 0-1 anchored scale", flush=True)

    # --- 3. Build the target detector + caching wrapper ------------------
    print("[3/7] preparing target detector and per-frame artifact caches ...", flush=True)
    cached_detector, memorization_skip_reason = _build_target_detector(args)
    not_applicable: dict[str, str] = {}
    if memorization_skip_reason is not None:
        not_applicable["vision.memorization"] = memorization_skip_reason
        print(f"      memorization SKIPPED — {memorization_skip_reason}", flush=True)

    # Memorization fill ensemble. lama_inpaint (the on-manifold fill) needs
    # the emboviz-lama worker; only bring it up when memorization will
    # actually run (a target is present) AND it was requested.
    memo_fills = _resolve_fills(args)
    cached_inpainter = (
        _build_inpainter(memo_fills)
        if cached_detector is not None and "vision.memorization" in enabled
        else None
    )

    # --- 4. Per-frame SHARED baseline + attention pre-extraction ---------
    # We pre-compute, once per frame:
    #   • the unperturbed model prediction (used by every diagnostic that
    #     needs a baseline — memorization, modality dropout, sensitivity,
    #     chunk consistency);
    #   • the clean per-camera attention heatmap (used by attention_drift
    #     AND the Rerun overlay).
    # Both are O(1) extra cost vs. having each diagnostic recompute them.
    print(f"[4/7] per-frame baseline + attention pre-extraction across "
          f"{len(trajectory.frames)} frames ...", flush=True)
    sel_token = TokenSelector(relative="before_action")
    has_attention = bool(model.capabilities & Capability.ATTENTION)
    baselines: list[ActionResult] = []
    attention_per_frame_clean: dict[int, dict[str, np.ndarray]] = {}
    attention_failed_frames: list[int] = []
    for i, scene in enumerate(trajectory.frames):
        baselines.append(averaged_predict(model, scene, calibration.n_samples))
        if has_attention:
            try:
                am = model.extract_attention(scene, sel_token)
                scene_cams = set(scene.observations.images)
                per_cam: dict[str, np.ndarray] = {}
                for cam in am.cameras:
                    if cam not in scene_cams:
                        continue
                    clean, _ = am.image_weights_clean(cam)
                    per_cam[cam] = np.asarray(clean, dtype=np.float32)
                attention_per_frame_clean[i] = per_cam
                # The full attention tensor is the memory hog; release the
                # reference now so the next frame's allocation starts
                # from a clean GPU.
                del am
            except Exception as e:
                # A SINGLE frame's failure (e.g. a transient OOM) must NOT
                # wipe the whole episode's attention. Record the failed frame
                # and continue: the drift diagnostic skips absent frames (and
                # never measures drift across the gap), and the exporter shows
                # the overlay on the frames that succeeded. Only if EVERY frame
                # fails does attention_per_frame_clean stay empty — handled by
                # the not-applicable branch below.
                print(f"      attn frame {i} failed (skipped, kept the rest): "
                      f"{type(e).__name__}: {e}", flush=True)
                attention_failed_frames.append(int(trajectory.frame_indices[i]))
        _maybe_collect(i)
        if (i + 1) % 10 == 0:
            print(f"      baseline+attn {i + 1}/{len(trajectory.frames)}",
                  flush=True)

    # --- 5. Trajectory-level diagnostics (attention drift + chunk) -------
    trajectory_axes: dict = {}

    if "internal.attention_drift" not in enabled:
        not_applicable["internal.attention_drift"] = (
            "disabled by --diagnostics / --skip-diagnostics"
        )
    elif not (model.capabilities & Capability.ATTENTION):
        not_applicable["internal.attention_drift"] = (
            f"model {model.model_id} does not expose Capability.ATTENTION"
        )
    elif not has_attention or not attention_per_frame_clean:
        # Attention extraction failed on EVERY frame (not just some) — don't
        # double-fail by running the diagnostic, which would re-extract and
        # hit the same adapter bug. Record it once with the actual reason.
        not_applicable["internal.attention_drift"] = (
            f"model declares Capability.ATTENTION but attention extraction "
            f"failed on all {len(trajectory.frames)} frames (dataset frames "
            f"{attention_failed_frames}) — see runner stderr for the "
            "underlying error. The diagnostic refuses to retry into the "
            "same failure."
        )
    else:
        drift = AttentionDriftDiagnostic().run_trajectory(
            model, trajectory,
            attention_per_frame_clean=attention_per_frame_clean,
        )
        if drift.severity == Severity.UNKNOWN:
            not_applicable["internal.attention_drift"] = drift.explanation
        else:
            # Per-frame-pair drift series for the Rerun time-series plot. The
            # diagnostic emits displacements_pixel as [later_frame_dataset_idx,
            # px] pairs (gap-aware: only frames adjacent in the trajectory), so
            # we map it straight through.
            drift_series = [
                [int(fi), float(d)]
                for fi, d in drift.raw.get("displacements_pixel", [])
            ]
            trajectory_axes["internal.attention_drift"] = {
                "severity":         drift.severity.value,
                "scalar_score":     drift.scalar_score,
                "explanation":      drift.explanation,
                "per_frame_series": drift_series,
            }
            print(f"      attention_drift: {drift.severity.value} "
                  f"({drift.scalar_score:.1f}px)", flush=True)

    if "internal.chunk_consistency" not in enabled:
        not_applicable["internal.chunk_consistency"] = (
            "disabled by --diagnostics / --skip-diagnostics"
        )
    else:
        chunk = ChunkConsistencyDiagnostic(
            calibration=calibration,
        ).run_trajectory(model, trajectory, baselines=baselines)
        if chunk.severity == Severity.UNKNOWN:
            not_applicable["internal.chunk_consistency"] = chunk.explanation
        else:
            # Per-frame chunk-disagreement series for the Rerun plot: each
            # comparable pair's normalized delta, keyed to the analyzed
            # frame t whose chunk made the prediction.
            cfi = chunk.raw.get("per_pair_frame_indices", [])
            nds = chunk.raw.get("normalized_deltas", [])
            chunk_series = [[int(fi), float(v)] for fi, v in zip(cfi, nds)]
            trajectory_axes["internal.chunk_consistency"] = {
                "severity":         chunk.severity.value,
                "scalar_score":     chunk.scalar_score,
                "explanation":      chunk.explanation,
                "raw_mean_delta":   chunk.raw.get("raw_mean_delta"),
                "per_frame_series": chunk_series,
            }
            print(f"      chunk_consistency: {chunk.severity.value} "
                  f"(normalized mean_delta={chunk.scalar_score:.3f}, "
                  f"raw={chunk.raw['raw_mean_delta']:.3f})", flush=True)

    # --- 6. Per-frame diagnostics with shared baseline -------------------
    print("[6/7] per-frame diagnostics (shared baseline) ...", flush=True)
    per_axis: dict = {}

    if "vision.memorization" not in enabled:
        not_applicable["vision.memorization"] = (
            "disabled by --diagnostics / --skip-diagnostics"
        )
    elif cached_detector is not None:
        print("      memorization (per camera, calibrated) ...", flush=True)
        memo = TrajectoryDiagnostic(
            MemorizationDiagnostic(
                target_detector=cached_detector, calibration=calibration,
                fill_modes=memo_fills, inpainter=cached_inpainter,
            ),
            progress=True,
        )
        per_axis["vision.memorization"] = memo.run(
            model, trajectory, baselines=baselines,
        )

    # Modality pool — built once from OTHER episodes, used by the dropout
    # diagnostic. The dataset's training data (or the user's eval pool)
    # is the right substitution distribution; we never fabricate
    # substitutes when the pool is empty.
    pool = None
    if "input.modality_dropout" not in enabled:
        not_applicable["input.modality_dropout"] = (
            "disabled by --diagnostics / --skip-diagnostics"
        )
    else:
        print("      building modality dropout pool from other episodes ...", flush=True)
        from emboviz.modality_pools import build_modality_pool
        declared_mods = {
            "state":          model.required_inputs.state,
            "gripper":        model.required_inputs.gripper,
            "action_history": model.required_inputs.action_history,
            "instruction":    model.required_inputs.instruction,
            "images":         sorted(model.required_inputs.cameras),
        }
        try:
            pool = build_modality_pool(
                dataset, current_episode=int(args.episode_idx),
                declared_modalities=declared_mods,
                n_samples=int(args.modality_pool_size),
                seed=int(args.modality_pool_seed),
                instruction_must_differ_from_task=trajectory.frames[0].instruction,
                cache_dir=args.modality_pool_cache_dir,
            )
            print(f"      pool: episodes={pool.metadata.get('sampled_episodes')}",
                  flush=True)
            for mod, ref in pool.ref_distance.items():
                print(f"        ref_distance[{mod}] = {ref:.4f}", flush=True)
        except Exception as e:
            print(f"      modality pool BUILD FAILED ({type(e).__name__}: {e}) — "
                  "modality dropout SKIPPED.", flush=True)
            not_applicable["input.modality_dropout"] = (
                f"could not build a marginal-distribution pool from the "
                f"dataset ({type(e).__name__}: {e}). The diagnostic refuses "
                "to fabricate substitutes."
            )
            pool = None

    if pool is not None:
        print(f"      modality dropout (K={args.modality_k_samples} per modality, "
              "marginal sampling) ...", flush=True)
        md = TrajectoryDiagnostic(
            ModalityDropoutDiagnostic(
                pool=pool,
                calibration=calibration,
                k_samples=int(args.modality_k_samples),
                seed=int(args.episode_idx) + 2,
            ),
            progress=True,
        )
        per_axis["input.modality_dropout"] = md.run(
            model, trajectory, baselines=baselines,
        )

    if "vision.scene_sensitivity" not in enabled:
        not_applicable["vision.scene_sensitivity"] = (
            "disabled by --diagnostics / --skip-diagnostics"
        )
    else:
        print(f"      sensitivity map ({args.sensitivity_grid_side}x"
              f"{args.sensitivity_grid_side}, per camera) ...", flush=True)
        sm = TrajectoryDiagnostic(
            SensitivityMapDiagnostic(
                grid_side=args.sensitivity_grid_side, calibration=calibration,
            ),
            progress=True,
        )
        per_axis["vision.scene_sensitivity"] = sm.run(
            model, trajectory, baselines=baselines,
        )

    # --- 7. prompt paraphrase on frame 0 ---------------------------------
    print("[7/7] prompt paraphrase on frame 0...", flush=True)
    pp = PromptParaphrasePerturber()
    paraphrase_deltas = {}
    paraphrase_failed: list[str] = []   # variants whose predict raised — surfaced in summary
    baseline_action = baselines[0].action   # reuse the precomputed frame-0 baseline
    for variant in pp.variants(trajectory.frames[0]):
        try:
            pred = averaged_predict(
                model, variant.scene, calibration.n_samples,
            ).action
        except Exception as e:
            print(f"      paraphrase {variant.variant_id} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            paraphrase_failed.append(variant.variant_id)
            continue
        paraphrase_deltas[variant.variant_id] = float(
            np.linalg.norm(pred - baseline_action)
        )
        print(f'      "{variant.scene.instruction}" -> '
              f'Δ={paraphrase_deltas[variant.variant_id]:.4f}', flush=True)
    paraphrase_mean = (
        float(np.mean(list(paraphrase_deltas.values())))
        if paraphrase_deltas else float("nan")
    )

    # --- 8. Collect per-camera overlays for Rerun (no model re-runs) ----
    # We already have everything we need on hand:
    #   • attention_per_frame_clean — from step 4.
    #   • cached_detector — holds every per-(scene,camera) detection
    #     populated during memorization. We reconstruct the masked image
    #     here via the same apply_fill() the diagnostic used — pure CPU for
    #     channel_mean / gaussian_blur; a cache hit on cached_inpainter for
    #     lama_inpaint (no second LaMa forward).
    print("[8/8] collect per-camera Rerun overlays from caches ...", flush=True)
    attention_per_frame_out: dict[int, dict[str, np.ndarray]] = {}
    sensitivity_per_frame: dict[int, dict[str, np.ndarray]] = {}
    target_mask_per_frame: dict[int, dict[str, np.ndarray]] = {}
    target_detection_per_frame: dict[int, dict[str, dict]] = {}
    masked_image_per_frame: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    modality_response_per_frame: dict[int, dict[str, float]] = {}

    for i, traj_frame_idx in enumerate(trajectory.frame_indices):
        if i in attention_per_frame_clean:
            attention_per_frame_out[traj_frame_idx] = attention_per_frame_clean[i]

    sens_result = per_axis.get("vision.scene_sensitivity")
    if sens_result is not None:
        for i, r in enumerate(sens_result.per_frame):
            grids = (r.raw or {}).get("sensitivity_grid_per_camera", {})
            if not grids:
                continue
            sensitivity_per_frame[trajectory.frame_indices[i]] = {
                cam: np.asarray(g, dtype=np.float32) for cam, g in grids.items()
            }

    if cached_detector is not None and "vision.memorization" in per_axis:
        from emboviz.diagnostics.memorization import apply_fill
        from emboviz.perturb.image._image_utils import to_array
        # Re-use the fill modes the memorization diagnostic actually
        # ran (read from raw) so the overlay matches the analysed image.
        # The on-manifold fill is reconstructed via the SAME cached
        # inpainter the diagnostic used, so each (frame, camera) is a
        # cache hit — no second LaMa forward pass.
        memo_results = per_axis["vision.memorization"].per_frame
        for i, r in enumerate(memo_results):
            raw = r.raw or {}
            detected = raw.get("detected_cameras") or []
            if not detected:
                continue
            scene = trajectory.frames[i]
            fill_modes_list = list((raw.get("per_fill") or {}).keys()) or list(
                memo_fills
            )
            cam_masks: dict[str, np.ndarray] = {}
            cam_detection: dict[str, dict] = {}
            cam_masked_per_fill: dict[str, dict[str, np.ndarray]] = {}
            for cam in detected:
                det = cached_detector.lookup(scene.scene_id, cam)
                if det is None or det.mask is None:
                    continue
                cam_masks[cam] = det.mask
                cam_detection[cam] = {
                    "label":      det.label,
                    "bbox":       list(det.bbox),
                    "confidence": float(det.confidence),
                    "all_boxes":  [list(b) for b in det.all_boxes]
                                  if det.all_boxes else None,
                    "all_scores": det.all_scores,
                }
                arr = to_array(scene.observations.images[cam].data)
                cam_masked_per_fill[cam] = {
                    fm: apply_fill(
                        arr, det.mask, fm,
                        inpainter=cached_inpainter,
                        cache_key=(scene.scene_id, cam),
                    )
                    for fm in fill_modes_list
                }
            if cam_masks:
                target_mask_per_frame[trajectory.frame_indices[i]] = cam_masks
            if cam_detection:
                target_detection_per_frame[trajectory.frame_indices[i]] = cam_detection
            if cam_masked_per_fill:
                masked_image_per_frame[trajectory.frame_indices[i]] = cam_masked_per_fill

    if "input.modality_dropout" in per_axis:
        for i, r in enumerate(per_axis["input.modality_dropout"].per_frame):
            raw = r.raw or {}
            per_mod = raw.get("per_modality") or {}
            if not per_mod:
                continue
            per: dict[str, float] = {}
            for modality, sub in per_mod.items():
                if isinstance(sub, dict) and "mean_response_normalized" in sub:
                    per[modality] = float(sub["mean_response_normalized"])
            if per:
                modality_response_per_frame[trajectory.frame_indices[i]] = per

    print(f"      attention frames: {len(attention_per_frame_out)}", flush=True)
    print(f"      sensitivity frames: {len(sensitivity_per_frame)}", flush=True)
    print(f"      target-mask frames: {len(target_mask_per_frame)}", flush=True)

    # --- 9. expert delta (imitation-accuracy only) + Rerun + failures ---
    has_expert = any(
        scene.metadata.get("expert_action") is not None
        for scene in trajectory.frames
    )
    show_imitation = bool(getattr(args, "show_imitation", False)) and has_expert

    if not show_imitation:
        if has_expert:
            print("[9/9] imitation L2 vs expert: HIDDEN "
                  "(dataset has recorded expert actions but "
                  "--show-imitation was not passed; this is a BC "
                  "validation metric, not a VLA diagnostic).", flush=True)
        else:
            print("[9/9] imitation L2 vs expert: N/A "
                  "(deployment data has no expert actions to compare to).",
                  flush=True)
    else:
        print("[9/9] expert delta (imitation accuracy) + Rerun export "
              "[--show-imitation enabled]...", flush=True)
    profile = trajectory.frames[0].profile
    dim_names: list[str] = (
        list(profile.action.dim_names)
        if profile is not None and profile.action is not None
           and profile.action.dim_names is not None
        else []
    )

    expert_delta_per_frame: list[float] = []
    expert_delta_per_frame_per_dim: list[list[float]] = []
    per_dim_mean_abs: list[float] = []
    per_dim_max_abs: list[float] = []
    dim_label: list[str] = []
    mean_delta: float = float("nan")

    if show_imitation:
        n_compared_dims = 0
        for i, scene in enumerate(trajectory.frames):
            # Re-use the precomputed baseline rather than running predict
            # again — same action, no extra forwards.
            pred = baselines[i].action
            expert = scene.metadata.get("expert_action")
            if expert is None:
                continue
            expert_arr = np.asarray(expert, dtype=np.float32)
            if len(pred) != len(expert_arr):
                raise ValueError(
                    f"imitation_accuracy: model produced "
                    f"{len(pred)}-dim action but dataset's recorded "
                    f"expert action is {len(expert_arr)}-dim for scene "
                    f"'{scene.scene_id}'. This is a real shape mismatch "
                    "between the model's action space and the dataset's "
                    "action layout — the (model, dataset) pairing is "
                    "wrong. We never silently truncate; the user needs "
                    "to know they've paired incompatible action spaces."
                )
            n = len(pred)
            n_compared_dims = n
            per_dim = (pred - expert_arr).tolist()
            expert_delta_per_frame_per_dim.append(per_dim)
            expert_delta_per_frame.append(float(np.linalg.norm(per_dim)))
        if expert_delta_per_frame:
            mean_delta = float(np.mean(expert_delta_per_frame))
            arr = np.array(expert_delta_per_frame_per_dim, dtype=np.float32)
            per_dim_mean_abs = np.abs(arr).mean(axis=0).tolist()
            per_dim_max_abs  = np.abs(arr).max(axis=0).tolist()
            dim_label = (
                dim_names[:n_compared_dims]
                if dim_names and len(dim_names) >= n_compared_dims
                else [f"d{i}" for i in range(n_compared_dims)]
            )
            print(f"      imitation L2 per-frame: "
                  f"{[f'{d:.3f}' for d in expert_delta_per_frame]}",
                  flush=True)
            print(f"      mean L2: {mean_delta:.3f}", flush=True)

    moments = find_failure_moments(
        per_axis,
        expert_delta_per_frame=expert_delta_per_frame if show_imitation else None,
        min_critical_axes=2,
    )
    print("      failure moments (≥2 critical):", flush=True)
    print(format_failure_moments(moments, max_show=10), flush=True)

    # --- 10. JSON summary (written BEFORE Rerun export) -----------------
    summary = {
        "story_id":          args.story_id,
        "model":             model.model_id,
        "episode_index":     int(args.episode_idx),
        "required_cameras":  sorted(model.required_inputs.cameras),
        "scene_cameras":     sorted(trajectory.frames[0].observations.images),
        "trajectory_source": trajectory.source,
        "n_frames":          len(trajectory.frames),
        "frame_indices":     list(trajectory.frame_indices),
        "instruction":       trajectory.frames[0].instruction,
        "calibration":       calibration.to_summary(),
        "per_axis":          {axis: tr.to_summary() for axis, tr in per_axis.items()},
        "trajectory_axes":   trajectory_axes,
        "not_applicable":    not_applicable,
        "paraphrase_deltas": paraphrase_deltas,
        "paraphrase_failed": paraphrase_failed,
        "paraphrase_mean_delta": paraphrase_mean,
        "imitation_accuracy": (
            {
                "expert_delta_per_frame":         expert_delta_per_frame,
                "expert_delta_mean":              mean_delta,
                "action_dim_names":               dim_label,
                "expert_delta_per_dim_mean_abs":  per_dim_mean_abs,
                "expert_delta_per_dim_max_abs":   per_dim_max_abs,
                "expert_delta_per_frame_per_dim": expert_delta_per_frame_per_dim,
                "note": (
                    "Imitation-accuracy is per-frame distance between the "
                    "model's prediction and the dataset's recorded expert "
                    "action. Useful as a BC validation metric on training "
                    "or held-out demonstration data. NOT a deployment-"
                    "debug signal — VLAs are trained to generalize, not "
                    "to copy the demonstrator exactly."
                ),
            } if show_imitation else None
        ),
        "failure_moments": [
            {
                "frame_idx": fm.frame_idx,
                "n_critical_axes": fm.n_critical_axes,
                "critical_axes": fm.critical_axes,
                "expert_delta": fm.expert_delta,
                "notes": fm.notes,
            }
            for fm in moments
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"      wrote {out_dir}/summary.json", flush=True)

    # --- 11. Rerun export — non-fatal -----------------------------------
    rrd_path = out_dir / "rollout.rrd"
    try:
        export_rerun(
            trajectory, per_axis, rrd_path,
            application_id=f"emboviz:{args.story_id}",
            attention_per_frame=attention_per_frame_out,
            sensitivity_per_frame=sensitivity_per_frame,
            target_mask_per_frame=target_mask_per_frame,
            target_detection_per_frame=target_detection_per_frame,
            masked_image_per_frame=masked_image_per_frame,
            modality_response_per_frame=modality_response_per_frame,
            trajectory_axis_results=trajectory_axes,
        )
        print(f"      wrote {rrd_path} ({rrd_path.stat().st_size:,} bytes)",
              flush=True)
    except Exception as e:
        print(f"      [WARN] rerun export FAILED ({type(e).__name__}: {e}) — "
              f"summary.json still has all diagnostic numbers.", flush=True)

    # Final cleanup so a multi-episode loop doesn't carry tensors over.
    if cached_detector is not None:
        cached_detector.clear()
    if cached_inpainter is not None:
        cached_inpainter.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    print(f"[done] total {time.time() - t0:.1f}s", flush=True)
    if rrd_path.exists():
        print(f"  rerun {rrd_path}", flush=True)
    print(f"  cat {out_dir}/summary.json", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--story-id", required=True)
    p.add_argument("--model-builder", required=True,
                   help="module.path:attr[:arg] resolving to a VLAModel instance")
    p.add_argument("--model-kwargs-json", default="",
                   help="JSON dict of kwargs passed to the model builder")
    p.add_argument("--dataset-builder", required=True,
                   help="module.path:attr[:arg] resolving to an EpisodeSource instance")
    p.add_argument("--dataset-kwargs-json", default="",
                   help="JSON dict of kwargs passed to the dataset builder")
    p.add_argument("--episode-idx", type=int, required=True)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument(
        "--n-frames", type=int, default=-1,
        help="Number of frames in the analysis window. Default -1 means "
             "ALL frames from --frame-start to the end of the episode.",
    )
    p.add_argument(
        "--frame-stride", type=int, default=1,
        help="Stride between frames in the window. Default 1 = every frame.",
    )
    p.add_argument("--sensitivity-grid-side", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--modality-pool-size", type=int, default=20,
        help="Number of OTHER episodes to draw substitution samples from "
             "for the modality dropout marginal pool. Default 20.",
    )
    p.add_argument(
        "--modality-k-samples", type=int, default=10,
        help="Substitution samples drawn from the pool per modality per "
             "frame. Default 10 (Monte-Carlo SE ~32%%).",
    )
    p.add_argument(
        "--modality-pool-seed", type=int, default=0,
        help="RNG seed for the marginal-pool episode sampler.",
    )
    p.add_argument(
        "--modality-pool-cache-dir", default=None,
        help="Optional directory where the modality pool is cached on disk.",
    )
    p.add_argument(
        "--target-text", default="",
        help="Phrase passed to the text-prompted target detector for the "
             "memorization diagnostic (e.g. \"the mug\"). Empty → skip "
             "memorization unless --target-annotations is given.",
    )
    p.add_argument(
        "--target-annotations", default="",
        help="Path to a per-frame target-annotation manifest (JSON or COCO "
             "format). When set, replaces text-prompted detection entirely — "
             "no GroundingDINO / SAM, no SAM 3, no AI in the loop. Mutually "
             "exclusive with --target-text in spirit (we prefer annotations "
             "when both are passed).",
    )
    p.add_argument(
        "--detector", default="sam3",
        choices=["sam3", "gd-sam"],
        help="Text-to-mask detector backend. 'sam3' (default) is the single-"
             "model SAM 3 pipeline (faster, better recall, native concept "
             "prompting). 'gd-sam' is the legacy GroundingDINO + SAM combo "
             "kept as a maintained fallback. Ignored when "
             "--target-annotations is set.",
    )
    p.add_argument(
        "--fills", default=None,
        help="Comma-separated memorization mask-fill ensemble. Default "
             "'channel_mean,gaussian_blur' (both OOD-leaning, no worker). "
             "Add 'lama_inpaint' for the on-manifold fill (needs the "
             "emboviz-lama worker) so the agreement gate spans the "
             "OOD/on-manifold axis.",
    )
    p.add_argument(
        "--show-imitation", action="store_true",
        help="Compute and show per-frame imitation L2 (model action vs "
             "recorded expert action). Hidden by default.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the per-frame and per-episode forward-pass estimate "
             "without running the diagnostic suite. Useful to size GPU "
             "budgets before committing.",
    )
    args = p.parse_args()
    try:
        run_story(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
