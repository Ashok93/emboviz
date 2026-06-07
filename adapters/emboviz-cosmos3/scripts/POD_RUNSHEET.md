# Cosmos 3 pod run-sheet (RunPod, A100 80 GB)

Staged: prove the pipe with a small smoke, then run a real rollout. Every paid
minute after setup is spent watching Cosmos, not debugging plumbing.

This reflects what actually works on RunPod (learned the hard way — see the
caveats at the bottom). The action-conditioned forward-dynamics server runs
**only** from NVIDIA's `vllm/vllm-omni:cosmos3` Docker image, and RunPod
**cannot run Docker inside a pod** — so the image must be the pod's *container
image*, not something you `docker run`.

## Prerequisites

- A RunPod **GPU Pod** (NOT serverless — serverless exposes a JSON job API, not
  the `/v1/videos/sync` HTTP endpoint this needs).
- A host with **CUDA 13 / driver ≥ 580** (`nvidia-smi` → `CUDA Version: 13.x`).
  vllm-cosmos3 is CUDA-13-only; a 12.x host fails with `libcudart.so.13`.
- An `HF_TOKEN` with **accepted access to BOTH** `nvidia/Cosmos3-Nano` **and**
  `nvidia/Cosmos-1.0-Guardrail` (the safety model loads at init; a valid token
  without guardrail access still 403s on startup).

## Stage 0 — deploy the server (image-as-pod)

In the RunPod pod config:

- **Container image:** `vllm/vllm-omni:cosmos3`
- **Expose HTTP ports:** `8000`
- **Environment variable:** `HF_TOKEN = hf_...`
- **Start command** (fixes "no command specified" and runs the server):
  ```
  vllm serve nvidia/Cosmos3-Nano --omni --model-class-name Cosmos3OmniDiffusersPipeline --allowed-local-media-path / --host 0.0.0.0 --port 8000 --init-timeout 1800
  ```

Deploy. RunPod gives a public URL `https://<podid>-8000.proxy.runpod.net`.
First boot pulls the image + downloads 33 GB of weights + cold-starts
(~20–40 min); `/health` returns 502 until it's up. Watch the pod **Logs** in the
dashboard — a `403` there means a gated repo wasn't accepted (fix access,
restart). Health gate from any machine:

```bash
U=https://<podid>-8000.proxy.runpod.net
until [ "$(curl -s -o /dev/null -w '%{http_code}' "$U/health")" = 200 ]; do sleep 20; done; echo READY
```

## Stage 0b — the emboviz client (runs anywhere; no GPU)

The client is a thin HTTP caller — run it locally or on any box that can reach
the URL. It needs only:

```bash
uv pip install -e adapters/emboviz-wire -e adapters/emboviz-cosmos3
uv pip install requests "imageio[ffmpeg]" av pillow
```

## Stage 1a — server smoke (isolate Cosmos, NOT our code)

One direct POST: synthetic frame, 16 zero actions → 16 generated frames. Proves
the server generates and the MP4 decodes. `--steps 8` keeps it fast; quality is
irrelevant here. **Use `--n-actions 16`** — Cosmos's video tokenizer is
temporally compressed, so a tiny chunk (e.g. 2) collapses to a single frame;
16 is the trained `action_chunk_size`.

```bash
U=https://<podid>-8000.proxy.runpod.net
uv run python adapters/emboviz-cosmos3/scripts/smoke_rollout.py --raw \
  --server-url $U --domain agibotworld --action-dim 29 --n-actions 16 --steps 8 \
  --out /tmp/smoke_raw
ls /tmp/smoke_raw            # expect 16 PNGs
```

If this fails, the problem is the server/Cosmos/domain — not emboviz. Fix here
before touching the adapter.

## Stage 1b — adapter smoke (isolate emboviz against the real server)

Same rollout through `Cosmos3WorldModel.rollout()` (drop `--raw`):

```bash
uv run python adapters/emboviz-cosmos3/scripts/smoke_rollout.py \
  --server-url $U --domain agibotworld --action-dim 29 --n-actions 16 --steps 8 \
  --out /tmp/smoke_adapter
```

A green Stage 1b means the full emboviz path works: conditioning frame + actions
→ HTTP → MP4 decode → `Trajectory`. (Synthetic frame + zero actions → a
near-static rainbow rollout; that proves the *pipe*, not anything meaningful.)

## Stage 2 — first real dream (a recorded DROID episode)

Only after both smokes are green. This flies the π0-DROID policy inside Cosmos
from a recorded episode's decisive moment and renders reality next to the
counterfactual dream.

The embodiment encoding is **implemented** for DROID (`droid_lerobot`, 10-D
`[pos_delta(3), rot6d_delta(6), gripper(1)]`, quantile-normalized — bit-faithful
to NVIDIA's `DROIDLeRobotDataset`). π0-DROID is joint-space; the FK bridge
converts its joint vector to the panda_link8 cartesian pose the encoder needs
(see `configs/cosmos_droid_pi0_demo.yaml`). Run the driver from the Mac:

```bash
uv run python -m emboviz.world_models.dream_cli \
  --config configs/cosmos_droid_pi0_demo.yaml --episode 312 \
  --keyframe-kinds gripper_change --near-frame 60 \
  --out outputs/cosmos_dream
```

Out comes the side-by-side Rerun `.rrd` (recorded | baseline dream | swap dream)
plus per-clip `summary.json`. Other embodiments need their own conditioning
bridge before they can run.

## Teardown (stop paying)

Stop / terminate the pod from the RunPod dashboard. (No persistent volume here,
so weights re-download on a fresh pod — keep that in mind for repeat runs.)

## Caveats that cost a session to learn

- **RunPod can't docker-in-docker** (unprivileged container). The cosmos image
  must be the pod's *container image*, not a `docker run` target.
- **CUDA 13 / driver ≥ 580 is mandatory** (vllm 0.21 + vllm-cosmos3 are cu13).
- **Serverless doesn't work** — wrong API surface; use a Pod + HTTP 8000.
- **Accept the guardrail gated repo** (`nvidia/Cosmos-1.0-Guardrail`), not just
  the main model, or the server 403s at init.
- **`action_chunk_size=16`** is the temporal granularity (the adapter's default).
- **BF16 only** — FP8/NVFP4 are not supported for the action path.
