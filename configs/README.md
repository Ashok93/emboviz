# emboviz run configs

One file = one `emboviz analyze` run. It declares **everything**: which model,
which checkpoint, the dataset mapping, the memorization target, which episodes,
and where to write the report. No CLI flag soup.

```bash
uv run emboviz analyze --config configs/pi0.yaml
```

There is one base config per model — `openvla`, `oft`, `pi0`, `gr00t`, `act`,
`smolvla` — each runnable as-is and meant to be copied and pointed at your own
checkpoint and dataset.

## Bring your own model + dataset

Copy the template closest to your setup and edit three things:

1. **`model.kwargs.checkpoint`** → the path / HF id of *your* fine-tune.
2. **`dataset.path`** → *your* dataset (HF repo id or local dir).
3. **`dataset.cameras` / `dataset.state.convention` / `dataset.gripper`** → the
   bindings the dataset format does **not** encode (which physical camera is the
   model's "primary", whether state is joint-angles vs ee-pose, gripper units).
   Everything else (dims, per-dim names, fps) is read from the data.

```bash
cp configs/pi0.yaml configs/my-run.yaml
$EDITOR configs/my-run.yaml
uv run emboviz analyze --config configs/my-run.yaml
```

## The contract is identical for every format

`dataset.format` may be `lerobot | gr00t | hdf5 | rlds` — the self-describing
"saved episode" formats (`gr00t` = LeRobot v2.1 + `modality.json`). The **schema you fill in is the same** regardless —
only the reader behind each `key` changes (a LeRobot key reads from parquet;
an HDF5 key reads from an h5 dataset; an RLDS key reads from the TFDS step
features). You never see that difference. The dims/per-dim-names are read from
each format's own schema (`meta/info.json` / the first demo's array shapes /
`builder.info.features`), never hand-typed.

For `rlds`, `dataset.path` is the **TFDS builder name** (e.g. `bridge`); put an
optional TFDS `data_dir` / `split` under `dataset.extra`, and use
`instruction.key: language_instruction` for the per-step instruction field.

(Rerun `.rrd` and MCAP/rosbag2 are *recording / debugging-viz* formats, not
dataset inputs — they are intentionally not accepted here.)

## Memorization mask-fill ensemble (`analysis.fills`)

The memorization diagnostic masks the target and checks whether the action
still moves. To avoid measuring the model's reaction to the *masking
artifact* instead of the *absence of the object*, it masks with an
**ensemble of fills** and requires agreement across them:

```yaml
analysis:
  fills: [channel_mean, gaussian_blur]      # default (no extra worker)
  # fills: [channel_mean, gaussian_blur, lama_inpaint]   # full ensemble
```

- `channel_mean`, `gaussian_blur` — pure-numpy, both OOD-leaning. Default.
- `lama_inpaint` — the **on-manifold** fill (plausible background via LaMa).
  Adding it makes the agreement gate span the on-manifold/OOD axis the
  literature prescribes (`LITERATURE.md` §1). It runs in the isolated
  `emboviz-lama` worker, which ships with emboviz core and is built
  automatically on first use — the analyze runner auto-starts it. Without it,
  every memorization result honestly flags
  `fill_ensemble.on_manifold_fill_present = false`.

## Fields the format can never encode (you always provide them)

- `dataset.state.convention` — joint-angles vs ee-pose. We refuse to guess
  (mislabeling is a silent-wrong-answer bug).
- `dataset.cameras` — maps the model's logical roles (`primary`, `wrist_left`)
  to your dataset's actual camera keys.
- `dataset.gripper` — optional. Omit it and the gripper value just stays inside
  the state vector (no gripper-specific perturbation, nothing fabricated). When
  set, declare where the gripper scalar comes from with **exactly one** of:
  `source` (its index — or per-dim name — within the `state.key` vector), or
  `key` (a *separate* dataset feature that carries the gripper on its own, e.g.
  DROID's `action.gripper_position`). The `gr00t` reader needs neither — it
  reads the index from the dataset's `meta/modality.json`.

## Closed-loop stress test (`analysis.stress`)

The world-model stress test (`emboviz.world_models.dream_cli`) is configured
under `analysis.stress`. The shipped DROID scenarios —
`ctrlworld_droid_pi0_{demo,towel,kettle,cable}.yaml` and
`cosmos_droid_pi0_demo.yaml` — are runnable as-is and meant to be copied.

```yaml
analysis:
  stress:
    world_model: ctrlworld          # ctrlworld (local GPU) | cosmos3 (vLLM-Omni server)
    # profile: droid                # ctrlworld checkpoint profile (default "droid");
    #                               # a custom fine-tune is a profile JSON path
    policy_adapter: pi0             # the policy under test
    policy_kwargs: {config_name: pi0_droid}
    action_convention: droid_joint_velocity   # the policy's chunk-row layout; never inferred
    control_hz: 15                  # the policy's control rate
    robot: franka_panda             # forward kinematics for joint-space conventions
    camera_map: {primary: exterior_1, wrist_left: wrist}   # policy roles -> frame regions
    concat_cameras: {exterior_1: primary, exterior_2: exterior_2, wrist: wrist}
    scene_swap:                     # masked counterfactual edit (baseline-vs-edit clip)
      mask_query: "yellow marker"   # SAM 3 locates it; empty replace_query -> LaMa removal
    n_actions: 4                    # frames dreamed per turn (ctrlworld: multiples of 4)
    execute_steps: 4                # frames committed before the policy re-plans
    n_loop_steps: 15                # turns per dream clip
    lead_s: 0.6                     # seed this many seconds before each keyframe
```

Backend-conditional fields are validated, not silently ignored: `server_url`,
`domain`, `action_dim`, `concat_resolution`, and whole-frame `perturbations`
apply only to `cosmos3`; `profile` only to `ctrlworld`;
`scene_swap.replace_query` (SDXL object insertion) is likewise cosmos3-only —
the ctrlworld backend supports removal. A ctrlworld checkpoint's region
vocabulary, chunk quantum, and native rate come from its profile, so those
checks run in the dream driver (before any worker spawns) rather than in the
host schema. Field-level documentation lives on `WorldStressCfg` in
`emboviz/config.py` and on `CtrlWorldProfile` in
`emboviz_ctrlworld/profiles.py`.
