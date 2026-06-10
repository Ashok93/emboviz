# emboviz-ctrlworld

**Ctrl-World** world-model adapter for emboviz — action-conditioned **forward
dynamics** on the DROID platform: given a conditioning frame, a pose-anchored
sparse history, and a sequence of end-effector poses, generate the future video
those poses produce, for all three DROID cameras jointly.

This is a world model (`emboviz_wire.world_model_protocol.WorldModel`), not a
policy. It is the substrate for emboviz's closed-loop stress test: the policy
under test acts, Ctrl-World renders the consequence, the policy reacts.

Reference: Guo, Shi, Chen, Finn, *Ctrl-World: A Controllable Generative World
Model for Robot Manipulation*, ICLR 2026 ([arXiv:2510.10125](https://arxiv.org/abs/2510.10125)),
official implementation [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)
(MIT). The inference modules are vendored under `emboviz_ctrlworld/_ctrl_world/`
(see its README for provenance and the exact modification list).

## Why this world model

Cosmos3-Nano forward dynamics conditions each chunk on a single frame, so
closed-loop rollouts accumulate error within one or two re-conditioning cycles.
Ctrl-World's contribution is the conditioning structure:

- **Multi-view joint prediction** — the two exterior cameras and the wrist
  camera are predicted together (stacked in latent space), so the wrist view
  stays consistent with the scene.
- **Pose-conditioned memory** — each forward pass attends to 6 sparse history
  frames tagged with their robot poses, anchoring the dream to where the
  rollout has actually been. The paper reports coherent rollouts past 20
  seconds on DROID.

## How it runs

The worker loads the model **locally on the GPU** (no separate server):
the 1.5B SVD-based UNet + VAE + CLIP text encoder, in bfloat16 (~5 GB VRAM,
fits beside a π0 worker on one A40). First start downloads from the Hugging
Face Hub (none gated):

| Piece | HF id | Size |
|---|---|---|
| Ctrl-World DROID checkpoint | `yjguo/Ctrl-World` (`checkpoint-10000.pt`) | ~8 GB |
| SVD base | `stabilityai/stable-video-diffusion-img2vid` | ~8 GB |
| CLIP text encoder | `openai/clip-vit-base-patch32` | ~600 MB |

## Conditioning contract

- **Frames**: one 320x576 vertical stack of `[exterior_1, exterior_2, wrist]`
  at 320x192 each (`emboviz_ctrlworld.stack_view`).
- **Actions**: absolute `[x, y, z, roll, pitch, yaw, gripper]` rows — the DROID
  `observation.state.cartesian_position` convention (`panda_link8` flange,
  extrinsic-XYZ euler) + gripper in [0, 1] — at **5 Hz**, in multiples of 4
  (one chunk = 4 future frames). Normalization to the training quantile bounds
  happens inside the adapter.
- **History**: the closed-loop driver passes the rollout's anchor frames; the
  adapter selects them per the reference `history_idx` schedule and keeps the
  loop in latent space via `Scene.metadata["ctrlworld_latent"]`.

Driving a joint-space policy (π0-DROID) through this contract — joint
velocities → integrated joint states → forward kinematics → `panda_link8`
poses — is `emboviz_ctrlworld.dream_step.CtrlWorldDreamStepper` plus
`emboviz_wire.policy_bridge`, wired by `emboviz.world_models.dream_cli`.

Note on the upstream rollout script: `models/utils.py::get_fk_solution` in the
reference computes a TCP-frame pose (flange + the -45° hand rotation +
0.1034 m), which differs from the `cartesian_position` convention the model
was trained on (`dataset/dataset_droid_exp33.py` lines 190-193). This adapter
conditions on the training convention, via the same Pinocchio FK
(`emboviz-robot`) that reproduces DROID's recorded `cartesian_position`
exactly.
