# VM setup scripts — re-create the integration-test environment

These scripts rebuild the per-model venvs we use for end-to-end integration
tests on a fresh GPU pod (RunPod RTX 3090 or similar). Each VLA pins
incompatible transformers + lerobot versions, so each lives in its own
uv venv.

Snapshot taken from a working pod on 2026-05-26:

| Venv path | Python | Key pins | Repo clone |
|---|---|---|---|
| `/root/venvs/openvla` | 3.10 | torch 2.12.0, transformers 4.49.0, lerobot 0.3.2 | — (loads from HF) |
| `/root/repos/openvla-oft/.venv` | 3.10 | torch 2.7.1, lerobot 0.3.3, moojink fork (vendored transformers) | github.com/moojink/openvla-oft |
| `/root/repos/openpi/.venv` | 3.11 | torch 2.7.1, transformers 4.53.2, jax 0.5.3 | github.com/Physical-Intelligence/openpi |
| `/root/venvs/gr00t` | 3.11 | torch 2.12.0, transformers 4.57.3 (pin matters — newer GroundingDINO API) | github.com/NVIDIA/Isaac-GR00T |

## One-shot install

```bash
# 1. SSH into fresh pod, then:
curl -fsSL https://raw.githubusercontent.com/Ashok93/emboviz/main/scripts/setup/install_all.sh | bash

# 2. Drop your HF token in /root/emboviz/.env (one line: HF_TOKEN=hf_...)
# 3. source /root/.bashrc.emboviz  (or restart shell)
# 4. ready to run integration tests
```

## Manual install (per venv)

If you only need one model, run just that script:

```bash
bash scripts/setup/01_install_openvla_venv.sh   # OpenVLA-7B on Bridge
bash scripts/setup/02_install_oft_venv.sh       # OpenVLA-OFT on LIBERO
bash scripts/setup/03_install_pi0_venv.sh       # pi0 on LIBERO
bash scripts/setup/04_install_gr00t_venv.sh     # GR00T-N1.7 on DROID
bash scripts/setup/05_install_sam3_venv.sh      # SAM 3 detector (memorization target)
bash scripts/setup/06_install_lerobot_venv.sh   # LeRobot dataset reader (isolated)
```

Each model script also installs the lightweight host shims (`emboviz-wire`,
`emboviz`, `emboviz-lerobot`, `emboviz-<model>`); `06` builds the isolated
LeRobot reader venv. The reader is also auto-built on first `analyze` if you
skip `06`.

## What's NOT installed automatically

- **HF_TOKEN**: put your own in `/root/emboviz/.env`. Required for π0/GR00T (gated repos: `nvidia/Cosmos-Reason2-2B`).
- **Model checkpoints**: downloaded lazily on first inference (HF cache → `$HF_HOME=/root/hf_cache`).
- **Datasets**: same. The isolated LeRobot reader venv lazy-fetches Bridge/LIBERO from HF the first time we run.
- **droid_sample**: shipped with Isaac-GR00T; the GR00T install script clones the repo which brings it.

## After install, verify

```bash
source /root/.bashrc.emboviz
bash scripts/final_integration_test.sh   # ~25 min, runs all 4 models end-to-end
```
