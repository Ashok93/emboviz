"""Closed-loop world-model stress test — run a policy inside a world model.

Given a recorded episode, this finds the decisive instants, optionally edits
each seed frame (masked object removal/swap, or whole-frame Cosmos edits), then
flies the user's policy inside the world model step by step from that
(perturbed) scene and asks the reasoner what happened. The simulator *is* the
world model; the policy is the thing under test.

Everything comes from the run config's ``analysis.stress`` section, including
the backend: ``world_model: cosmos3`` (vLLM-Omni server, single-frame
conditioning) or ``world_model: ctrlworld`` (local GPU, multi-view +
pose-anchored history). Output per clip is the **dream video** + the
**reasoner verdict** — no pixel divergence, because a perturbed scene never
happened in reality and has nothing to compare against.

Run::

    uv run python -m emboviz.world_models.dream_cli \
        --config configs/ctrlworld_droid_pi0_demo.yaml --episode 312 \
        --keyframe-kinds gripper_change --near-frame 60
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from emboviz.adapters import connect_world_model
from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.world_models.keyframes import Keyframe, detect_keyframes
from emboviz.world_models.simulate import closed_loop_rollout
from emboviz.world_models.dream_rerun import DreamColumn, export_dream_rerun
from emboviz.world_models.viz import frames_to_arrays


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "edit"


def _img(scene, role: str) -> np.ndarray:
    if role not in scene.observations.images:
        raise SystemExit(
            f"camera role '{role}' (from stress.concat_cameras) is not in the "
            f"episode (available: {sorted(scene.observations.images)}). Fix the mapping."
        )
    return np.asarray(scene.observations.images[role].data, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--out", default="outputs/world_dream")
    p.add_argument(
        "--max-clips", type=int, default=None,
        help="Stop after this many clips are produced (a cheap smoke run before "
             "the full keyframe sweep). Default: all keyframes x perturbations.",
    )
    p.add_argument(
        "--keyframe-kinds", default=None,
        help="Comma-separated keyframe kinds to keep (e.g. 'gripper_change' for "
             "the grasp/release only, skipping the many 'settle' keyframes). "
             "Default: all kinds.",
    )
    p.add_argument(
        "--near-frame", type=int, default=None,
        help="Keep only the single keyframe nearest this frame index (applied "
             "after --keyframe-kinds). Use to dream one decisive moment — e.g. "
             "--keyframe-kinds gripper_change --near-frame 60 picks the grasp.",
    )
    p.add_argument(
        "--seed-frames", default=None,
        help="Comma-separated frame indices to dream, bypassing keyframe "
             "detection — one clip per index, each seeded lead_s before it. Use "
             "to tile a spread of moments across the whole task in ONE run "
             "(workers load once), e.g. '20,40,61,80,100,120'.",
    )
    args = p.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        raise SystemExit(f"--max-clips must be >= 1, got {args.max_clips}.")

    cfg = load_run_config(args.config)
    cs = cfg.analysis.stress
    if cs is None:
        raise SystemExit("config has no analysis.stress section — nothing to run.")
    if cs.policy_adapter is None:
        raise SystemExit(
            "dream_cli runs the closed-loop *policy* simulator, so stress."
            "policy_adapter is required (currently 'pi0' for the DROID demo)."
        )

    episode = args.episode
    if episode is None:
        episode = int(str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[dream] loading episode {episode} via {cfg.dataset.format} reader ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    keyframes = detect_keyframes(real)
    print(f"[dream] {len(real.frames)} frames, fps {real.fps:g}; {len(keyframes)} keyframes")

    # Optional keyframe selection: dream a chosen set of moments instead of
    # sweeping every keyframe. --seed-frames overrides detection with an explicit
    # list (one clip per frame); otherwise --keyframe-kinds filters by kind and
    # --near-frame keeps the single nearest detected keyframe.
    if args.seed_frames:
        idxs = [int(x) for x in args.seed_frames.split(",") if x.strip()]
        bad = [i for i in idxs if not 0 <= i < len(real.frames)]
        if bad:
            raise SystemExit(f"--seed-frames out of range {bad} (episode has {len(real.frames)} frames).")
        keyframes = [Keyframe(i, "manual", 0.0, 0.0) for i in idxs]
        print(f"[dream] --seed-frames: dreaming {len(keyframes)} explicit frames {idxs}")
    else:
        if args.keyframe_kinds:
            kinds = {k.strip() for k in args.keyframe_kinds.split(",") if k.strip()}
            keyframes = [kf for kf in keyframes if kf.kind in kinds]
            print(f"[dream] kept {len(keyframes)} keyframes of kind(s) {sorted(kinds)}")
        if args.near_frame is not None:
            if not keyframes:
                raise SystemExit("--near-frame: no keyframes left to choose from after --keyframe-kinds.")
            chosen = min(keyframes, key=lambda kf: abs(kf.index - args.near_frame))
            keyframes = [chosen]
            print(f"[dream] --near-frame {args.near_frame}: dreaming keyframe "
                  f"{chosen.index} ({chosen.kind})")
    if not keyframes:
        raise SystemExit("no keyframes selected — relax --keyframe-kinds / --near-frame / --seed-frames.")

    # ── backend wiring ──────────────────────────────────────────────────
    # Each backend supplies: the connected world model, the regions of its
    # stitched conditioning frame, a seed builder (per-region images -> one
    # stitched frame), a splitter (stitched frame -> per-region images), a
    # stepper factory, and the dream:episode frame-stride.
    from emboviz.adapters import connect
    from emboviz.config import _JOINT_ACTION_CONVENTIONS
    from emboviz_wire.policy_bridge import make_state_tracker

    # Joint-space conventions need forward kinematics (joints -> EE pose); build
    # the robot once. Cartesian conventions track the pose directly (kinematics
    # stays None). emboviz-robot is imported only on this branch.
    kinematics = None
    if cs.action_convention in _JOINT_ACTION_CONVENTIONS:
        from emboviz_robot import load_kinematics
        if cs.robot is not None:
            print(f"[dream] forward kinematics: preconfigured robot '{cs.robot}'")
            kinematics = load_kinematics(cs.robot)
        else:
            print(f"[dream] forward kinematics: custom URDF {cs.robot_urdf}")
            kinematics = load_kinematics(
                urdf=cs.robot_urdf, ee_frame=cs.robot_ee_frame, joint_names=cs.robot_joint_names
            )

    def tracker_for(seed_index: int):
        frame = real.frames[seed_index]
        if frame.observations.state is None or frame.observations.gripper is None:
            raise SystemExit(
                f"seed frame {seed_index} lacks state/gripper needed to anchor the "
                "policy's actions. Map the policy's state (joint_position for a "
                "joint convention, cartesian_position for a cartesian one) + gripper "
                "in the config."
            )
        return make_state_tracker(
            np.asarray(frame.observations.state.values, dtype=np.float32),
            float(frame.observations.gripper.value),
            action_convention=cs.action_convention,
            state_convention=cs.state_convention,
            kinematics=kinematics,
            control_hz=cs.control_hz,
        )

    print(f"[dream] connecting policy '{cs.policy_adapter}' (kwargs: {cs.policy_kwargs or '{}'}) ...")
    policy = connect(cs.policy_adapter, actor_kwargs=cs.policy_kwargs or None)

    editor = None
    if cs.world_model == "cosmos3":
        from emboviz_cosmos3.concat_view import build_concat_view, split_concat_view
        from emboviz_cosmos3.dream_step import PolicyDreamStepper
        from emboviz_cosmos3.perturb import CosmosImageEditor

        # fps is the conditioning frame rate Cosmos reads — it MUST equal the
        # rate at which the action deltas are sampled, or the model misreads the
        # motion dynamics. For droid_lerobot the dataset is 15 Hz (one generated
        # frame per control step), so fps = control_hz. guardrails=False matches
        # NVIDIA's robotics forward-dynamics cookbook (run_fd_with_vllm.ipynb).
        wm = connect_world_model("cosmos3", world_model_kwargs={
            "server_url": cs.server_url, "domain_name": cs.domain,
            "action_dim": cs.action_dim, "conditioning_camera": cs.conditioning_camera,
            "fps": int(round(cs.control_hz)), "guardrails": False,
        })
        editor = CosmosImageEditor(cs.server_url)
        regions = ("wrist", "exterior_left", "exterior_right")
        region_cameras = dict(cs.concat_cameras)   # validated + defaulted by the config schema
        split_view = split_concat_view
        dream_stride = 1   # one Cosmos frame per control step

        def build_seed(images_by_region: dict[str, np.ndarray]) -> np.ndarray:
            return build_concat_view(
                images_by_region["wrist"],
                images_by_region["exterior_left"],
                images_by_region["exterior_right"],
                wrist_size=cs.concat_resolution,
            )

        def stepper_for(seed_index: int) -> PolicyDreamStepper:
            return PolicyDreamStepper(
                policy.client.predict,
                tracker=tracker_for(seed_index),
                camera_map=cs.camera_map,
                instruction=real.frames[seed_index].instruction,
                n_actions=cs.n_actions,
                execute_steps=cs.execute_steps,
            )

    else:  # ctrlworld
        from emboviz_ctrlworld.dream_step import CtrlWorldDreamStepper
        from emboviz_ctrlworld.profiles import check_stress_compat, resolve_profile
        from emboviz_ctrlworld.stack_view import build_stack_view, split_stack_view

        # The checkpoint profile carries the region vocabulary, chunk quantum,
        # and native rate, so its compatibility checks run here (before any
        # worker spawns) rather than in the host config schema. A file-based
        # profile is absolutized so the worker (whose cwd differs) resolves
        # the same file.
        profile_ref = cs.profile
        profile_path = Path(cs.profile).expanduser()
        if profile_path.suffix == ".json" or profile_path.is_file():
            profile_ref = str(profile_path.resolve())
        profile = resolve_profile(profile_ref)
        try:
            region_cameras = check_stress_compat(
                profile,
                camera_map=cs.camera_map,
                concat_cameras=cs.concat_cameras,
                n_actions=cs.n_actions,
                control_hz=cs.control_hz,
            )
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"[dream] ctrl-world profile '{profile.name}': {profile.description}")

        wm = connect_world_model("ctrlworld", world_model_kwargs={
            "profile": profile_ref,
            "conditioning_camera": cs.conditioning_camera,
        })
        regions = profile.views
        # One dreamed frame spans control_hz / native_fps recorded frames
        # (3 for the droid profile's 5 Hz against DROID's 15 Hz).
        dream_stride = int(round(cs.control_hz / profile.native_fps))

        def split_view(stack: np.ndarray) -> dict[str, np.ndarray]:
            return split_stack_view(stack, views=profile.views)

        def build_seed(images_by_region: dict[str, np.ndarray]) -> np.ndarray:
            return build_stack_view(
                images_by_region, views=profile.views, view_hw=profile.view_hw
            )

        def stepper_for(seed_index: int) -> CtrlWorldDreamStepper:
            return CtrlWorldDreamStepper(
                policy.client.predict,
                profile=profile,
                tracker=tracker_for(seed_index),
                camera_map=cs.camera_map,
                instruction=real.frames[seed_index].instruction,
                n_actions=cs.n_actions,
                execute_steps=cs.execute_steps,
                control_hz=cs.control_hz,
            )

    reasoner = None
    if cs.reasoner_url:
        # The reasoner client ships with the cosmos3 shim (it talks to a Cosmos
        # reason server); it judges the dream video and is independent of which
        # world model rendered it.
        try:
            from emboviz_cosmos3.reason import CosmosReasoner
        except ImportError as e:
            raise SystemExit(
                "stress.reasoner_url is set, but the reasoner client lives in the "
                "emboviz-cosmos3 shim, which is not installed. Run `uv sync "
                "--extra cosmos3` (alongside your other extras), or remove "
                f"reasoner_url. ({e})"
            )
        reasoner = CosmosReasoner(cs.reasoner_url)

    # Masked counterfactual swap (baseline-vs-swap mode). SAM 3 locates the
    # target per camera; a non-empty replace_query paints the replacement object
    # via the SD inpaint adapter (cosmos3 backend only), an empty one removes the
    # object via LaMa. The workers are brought up the same way the runner brings
    # up the model/reader workers (connect -> auto-install venv -> spawn -> ping).
    swapper = None
    swap_label = None
    if cs.scene_swap is not None:
        from emboviz.perturb._target_detection import SAM3Detector
        from emboviz.world_models.scene_swap import SceneSwapper
        sw = cs.scene_swap
        connect("sam3", auto_spawn=True, auto_install=True)
        detector = SAM3Detector(
            target_text=sw.mask_query,
            score_threshold=sw.detector_score_threshold,
            mask_threshold=sw.detector_mask_threshold,
        )
        if sw.replace_query:
            from emboviz.perturb.image._inpaint import SDInpaintInserter
            connect("sd-inpaint", auto_spawn=True, auto_install=True)
            swapper = SceneSwapper(
                mask_query=sw.mask_query, replace_query=sw.replace_query,
                detector=detector, inserter=SDInpaintInserter(
                    guidance_scale=sw.edit_guidance_scale,
                    negative_prompt=sw.edit_negative_prompt,
                ),
                mask_dilation=sw.mask_dilation,
            )
            swap_label = f"swap → {sw.replace_query}"
            print(f"[dream] scene swap: {sw.mask_query!r} → {sw.replace_query!r} "
                  f"(SAM 3 score≥{sw.detector_score_threshold} + SD inpaint)")
        else:
            from emboviz.perturb.image._inpaint import LamaInpainter
            connect("lama", auto_spawn=True, auto_install=True)
            swapper = SceneSwapper(
                mask_query=sw.mask_query, detector=detector, inpainter=LamaInpainter(),
                mask_dilation=sw.mask_dilation,
            )
            swap_label = f"remove {sw.mask_query}"
            print(f"[dream] scene swap: remove {sw.mask_query!r} "
                  f"(SAM 3 score≥{sw.detector_score_threshold} + LaMa inpaint)")

    def seed_pose(seed_index: int) -> tuple[np.ndarray, float]:
        """The seed frame's end-effector pose + gripper for pose-anchored world
        models. Joint conventions go through the same forward kinematics the
        stepper uses; cartesian conventions read the mapped state directly."""
        frame = real.frames[seed_index]
        gripper = float(frame.observations.gripper.value)
        if cs.action_convention in _JOINT_ACTION_CONVENTIONS:
            from emboviz_wire.policy_bridge import JointStateTracker
            tracker = tracker_for(seed_index)
            assert isinstance(tracker, JointStateTracker)
            return tracker.cartesian_state(), gripper
        return np.asarray(frame.observations.state.values, dtype=np.float32)[:6], gripper

    def seed_view(seed_index: int) -> np.ndarray:
        frame = real.frames[seed_index]
        return build_seed({r: _img(frame, region_cameras[r]) for r in regions})

    def on_step(i: int, traj) -> None:
        print(f"    step {i}: {len(traj.frames)} frame(s) committed", flush=True)

    needs_pose = bool(getattr(wm, "conditions_on_history", False))

    def run_dream_from_seed(seed_index: int, seed_img: np.ndarray):
        """Fly the policy inside the world model from ``seed_img`` and split the
        dream into per-region views. A fresh stepper is built per call so the
        baseline and swap rollouts never share the (stateful) state tracker.
        Returns ``(views_by_region, dream_frames)``."""
        pose, gripper = seed_pose(seed_index) if needs_pose else (None, None)
        dream = closed_loop_rollout(
            wm, seed_img, stepper_for(seed_index),
            n_steps=cs.n_loop_steps, conditioning_camera=cs.conditioning_camera,
            instruction=real.frames[seed_index].instruction,
            execute_steps=cs.execute_steps, on_step=on_step,
            seed_state=pose, seed_gripper=gripper,
        )
        dream_frames = frames_to_arrays(dream.trajectory, cs.conditioning_camera)
        split = [split_view(f) for f in dream_frames]
        views = {region: [s[region] for s in split] for region in regions}
        return views, dream_frames

    def original_views_for(seed_index: int, n_frames: int) -> dict[str, list]:
        """The recorded episode window aligned to a dream of ``n_frames`` frames,
        one entry per region (each region's original camera role). Dream frame i
        sits ``(i + 1) * dream_stride`` recorded frames after the seed."""
        window = [
            real.frames[seed_index + (i + 1) * dream_stride]
            for i in range(n_frames)
            if seed_index + (i + 1) * dream_stride < len(real.frames)
        ]
        return {region: frames_to_arrays(window, region_cameras[region]) for region in regions}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    lead = int(round(cs.lead_s * real.fps))
    commit = cs.execute_steps if cs.execute_steps is not None else cs.n_actions

    def run_swap_clip(kf, seed_index: int) -> dict:
        """Baseline-vs-swap: one clip with the original dream and the masked-swap
        dream side by side. The seed is swapped per camera (original where SAM
        finds nothing); the swap column is left not-run only if no camera hits."""
        clip_dir = out / f"clip_{kf.index:04d}_{kf.kind}__swap"
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame = real.frames[seed_index]

        baseline_seed = seed_view(seed_index)
        swap = swapper.swap(frame, region_cameras)
        print(f"  [swap] frame {kf.index}: {swap.summary()}")
        swapped_seed = build_seed(swap.images_by_region)

        print(f"  [dream] frame {kf.index} ({kf.kind}) / baseline: "
              f"{cs.n_loop_steps} turns x dream {cs.n_actions}, commit {commit} ...")
        baseline_views, baseline_frames = run_dream_from_seed(seed_index, baseline_seed)

        swap_frames = None
        if swap.any_edited:
            print(f"  [dream] frame {kf.index} ({kf.kind}) / {swap_label}: "
                  f"{cs.n_loop_steps} turns x dream {cs.n_actions}, commit {commit} ...")
            swap_views, swap_frames = run_dream_from_seed(seed_index, swapped_seed)
            swap_column = DreamColumn(name=swap_label, views=swap_views, note=swap.summary())
            seeds = [("original", baseline_seed), (swap_label, swapped_seed)]
        else:
            print(f"  [swap] {cs.scene_swap.mask_query!r} not detected in any camera "
                  f"— swap dream not run.")
            swap_column = DreamColumn(
                name=swap_label, views=None,
                note=f"{cs.scene_swap.mask_query!r} not detected in any camera",
            )
            seeds = [("original", baseline_seed)]

        n_frames = len(baseline_views[regions[0]])
        original_views = original_views_for(seed_index, n_frames)
        rrd_path = export_dream_rerun(
            clip_dir / "dream.rrd",
            original_views=original_views,
            dream_columns=[
                DreamColumn(name="baseline", views=baseline_views, note="unedited seed"),
                swap_column,
            ],
            seeds=seeds,
            instruction=frame.instruction,
            fps=real.fps / dream_stride,
            policy_name=cs.policy_adapter,
            context_note=f"**Swap:** {swap.summary()}",
            recording_id=f"dream_{episode}_{kf.index:04d}_swap",
        )
        print(f"    saved {rrd_path}")

        verdict = None
        if reasoner is not None:
            verdict = {"baseline": reasoner.judge(baseline_frames, cs.reasoner_question)}
            if swap_frames is not None:
                verdict["swap"] = reasoner.judge(swap_frames, cs.reasoner_question)
            print(f"    verdict: {verdict}")

        record = {
            "keyframe_index": kf.index, "kind": kf.kind, "seed_index": seed_index,
            "mode": "scene_swap",
            "world_model": cs.world_model,
            "mask_query": cs.scene_swap.mask_query,
            "replace_query": cs.scene_swap.replace_query,
            "swap_per_camera": [asdict(c) for c in swap.per_camera],
            "swap_run": swap.any_edited,
            "n_loop_steps": cs.n_loop_steps, "n_actions": cs.n_actions,
            "execute_steps": commit, "n_frames": n_frames,
            "reasoner_question": cs.reasoner_question, "verdict": verdict,
        }
        (clip_dir / "verdict.json").write_text(json.dumps(record, indent=2))
        return record

    def run_perturbation_clip(kf, seed_index: int, instruction) -> dict:
        """Whole-frame Cosmos edit (or unperturbed): one dream column per clip."""
        tag = _slug(instruction) if instruction else "unperturbed"
        clip_dir = out / f"clip_{kf.index:04d}_{kf.kind}__{tag}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        seed = seed_view(seed_index)
        if instruction:
            print(f"  [edit] frame {kf.index}: {instruction!r}")
            seed = editor.edit(seed, instruction)

        print(f"  [dream] frame {kf.index} ({kf.kind}) / {tag}: "
              f"{cs.n_loop_steps} turns x dream {cs.n_actions}, commit {commit} ...")
        dream_views, dream_frames = run_dream_from_seed(seed_index, seed)
        n_frames = len(dream_views[regions[0]])
        original_views = original_views_for(seed_index, n_frames)

        col_name = "dream" if not instruction else f"edit: {instruction}"
        note = None if not instruction else "whole-frame edit (counterfactual)"
        rrd_path = export_dream_rerun(
            clip_dir / "dream.rrd",
            original_views=original_views,
            dream_columns=[DreamColumn(name=col_name, views=dream_views, note=note)],
            seeds=[("seed", seed)],
            instruction=real.frames[seed_index].instruction,
            fps=real.fps / dream_stride,
            policy_name=cs.policy_adapter,
            recording_id=f"dream_{episode}_{kf.index:04d}_{tag}",
        )
        print(f"    saved {rrd_path}")

        verdict = None
        if reasoner is not None:
            verdict = reasoner.judge(dream_frames, cs.reasoner_question)
            print(f"    verdict: {verdict}")

        record = {
            "keyframe_index": kf.index, "kind": kf.kind, "seed_index": seed_index,
            "mode": "perturbation", "perturbation": instruction,
            "world_model": cs.world_model,
            "n_loop_steps": cs.n_loop_steps, "n_actions": cs.n_actions,
            "execute_steps": commit, "n_frames": n_frames,
            "reasoner_question": cs.reasoner_question, "verdict": verdict,
        }
        (clip_dir / "verdict.json").write_text(json.dumps(record, indent=2))
        return record

    def flush_summary() -> None:
        (out / "summary.json").write_text(json.dumps(
            {"episode": episode, "policy": cs.policy_adapter,
             "world_model": cs.world_model, "clips": summary}, indent=2
        ))

    capped = False
    for kf in keyframes:
        if capped:
            break
        seed_index = max(0, kf.index - lead)
        if seed_index >= len(real.frames) - 1:
            continue  # no room to seed from this keyframe
        if swapper is not None:
            summary.append(run_swap_clip(kf, seed_index))
            flush_summary()
            if args.max_clips is not None and len(summary) >= args.max_clips:
                print(f"[dream] reached --max-clips {args.max_clips}; stopping early.")
                break
            continue
        for instruction in (list(cs.perturbations) if cs.perturbations else [None]):
            summary.append(run_perturbation_clip(kf, seed_index, instruction))
            flush_summary()
            if args.max_clips is not None and len(summary) >= args.max_clips:
                print(f"[dream] reached --max-clips {args.max_clips}; stopping early.")
                capped = True
                break

    print(f"\n[dream] DONE: {len(summary)} clips -> {out}/")


if __name__ == "__main__":
    main()
