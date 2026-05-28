"""`emboviz init` — interactive onboarding wizard.

Asks the user about their stack (robot, model, dataset format), then
emits a tailored install script + a starter scene-loading template.
The goal: a new user goes from "I have a checkpoint and rollouts" to
"I have a working diagnostic command line" in <5 minutes.

Why this exists: each VLA family has its own ecosystem and per-venv
dependency conflicts are real. Without `emboviz init`, a new user
spends an hour reading READMEs and a Discord thread per model. With
this wizard, they answer 4 questions and get the exact commands.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelProfile:
    """One supported model family with its install path + caveats."""

    key: str                  # internal name used in CLI args
    display: str              # human-facing name
    venv_python: str          # required Python version
    install_steps: list[str]  # shell commands to set up the venv
    notes: str = ""           # caveats / gating info
    requires_separate_venv: bool = True


MODELS: dict[str, ModelProfile] = {
    "openvla-7b": ModelProfile(
        key="openvla-7b",
        display="OpenVLA-7B (Stanford/Berkeley)",
        venv_python="3.12",
        install_steps=[
            "uv venv .venv-openvla --python 3.12",
            "source .venv-openvla/bin/activate",
            "uv pip install 'emboviz[openvla]'",
        ],
        notes="Public weights on HuggingFace. ~15GB checkpoint download on first use.",
    ),
    "smolvla": ModelProfile(
        key="smolvla",
        display="SmolVLA (HuggingFace, 450M params, runs on consumer GPU)",
        venv_python="3.10",
        install_steps=[
            "uv venv .venv-smolvla --python 3.10",
            "source .venv-smolvla/bin/activate",
            "uv pip install torch torchvision 'transformers>=4.50,<5.0' 'lerobot>=0.5'",
            "uv pip install num2words Pillow scipy scikit-learn matplotlib tqdm",
            "uv pip install 'rerun-sdk>=0.22.1' mcap imageio imageio-ffmpeg",
            "uv pip install --no-deps -e /path/to/emboviz",
        ],
        notes="Small + accessible. Uses lerobot/smolvla_base checkpoint. Dataset must be in LeRobot v2.1 format.",
    ),
    "pi0": ModelProfile(
        key="pi0",
        display="π0 / π0.5 / π0-FAST (Physical Intelligence)",
        venv_python="3.11",
        install_steps=[
            "git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git",
            "cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync",
            "source .venv/bin/activate",
            "uv pip install --no-deps -e /path/to/emboviz",
            "uv pip install 'rerun-sdk>=0.22.1' mcap",
        ],
        notes="openpi's own venv pins everything correctly. Checkpoints download from gs://openpi-assets. Choose config: pi0_aloha_sim, pi0_libero, pi0_fast_droid, pi05_libero, pi05_droid.",
    ),
    "gr00t-n1.7": ModelProfile(
        key="gr00t-n1.7",
        display="GR00T-N1.7-3B (NVIDIA humanoid foundation)",
        venv_python="3.10",
        install_steps=[
            "git clone https://github.com/NVIDIA/Isaac-GR00T.git",
            "uv venv .venv-gr00t --python 3.10",
            "source .venv-gr00t/bin/activate",
            "uv pip install 'torch>=2.7' torchvision 'transformers==4.57.3' accelerate 'numpy>=1.26'",
            "uv pip install diffusers dm-tree tyro lmdb msgpack msgpack-numpy peft termcolor",
            "uv pip install omegaconf jsonlines gymnasium einops albumentations opencv-python-headless kornia",
            "uv pip install --no-deps -e ./Isaac-GR00T",
            "uv pip install --no-deps -e /path/to/emboviz",
            "uv pip install scipy scikit-learn matplotlib tqdm huggingface_hub 'rerun-sdk>=0.22.1' mcap lerobot",
            "# Accept NVIDIA Open Model License at: https://huggingface.co/nvidia/Cosmos-Reason2-2B",
            "huggingface-cli login   # paste your HF token",
        ],
        notes="REQUIRES one-click acceptance of NVIDIA Open Model License on the Cosmos-Reason2-2B HuggingFace page. The GR00T backbone is downloaded as a separate model artifact.",
    ),
    "openvla-oft": ModelProfile(
        key="openvla-oft",
        display="OpenVLA-OFT (Stanford, faster than OpenVLA-7B + proprio)",
        venv_python="3.10",
        install_steps=[
            "git clone https://github.com/moojink/openvla-oft.git",
            "cd openvla-oft",
            "# Edit pyproject.toml: requires-python = \">=3.10,<3.11\"",
            "uv venv .venv --python 3.10",
            "source .venv/bin/activate",
            "uv pip install -e .   # heavy: TF + custom transformers fork",
            "uv pip install --no-deps -e /path/to/emboviz",
        ],
        notes="Uses the moojink/transformers-openvla-oft fork (custom). Heaviest install — pulls TensorFlow + DLIMP + custom transformers.",
    ),
}


ROBOTS = [
    "Franka Panda + Robotiq",
    "UR5 / UR10 + Robotiq",
    "Trossen (single-arm ALOHA)",
    "ALOHA (bimanual)",
    "Unitree H1 / G1 (humanoid)",
    "Other / custom",
]


DATASETS = [
    "LeRobot v3 (HF dataset)",
    "Rerun .rrd",
    "Foxglove / ROS bag (.mcap)",
    "HuggingFace generic dataset",
    "Custom format (I'll write a loader)",
]


def _ask(prompt: str, choices: list[str], default_idx: int = 0) -> int:
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        marker = " (default)" if i - 1 == default_idx else ""
        print(f"  {i}) {c}{marker}")
    while True:
        try:
            raw = input(f"> [{default_idx + 1}] ").strip()
        except EOFError:
            return default_idx
        if not raw:
            return default_idx
        try:
            n = int(raw)
            if 1 <= n <= len(choices):
                return n - 1
        except ValueError:
            pass
        print(f"  enter 1..{len(choices)}")


def _ask_text(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"\n{prompt}{suffix}\n> ").strip()
    except EOFError:
        return default
    return raw or default


def run_wizard() -> int:
    print(textwrap.dedent("""
        ──────────────────────────────────────────────────────────────
         Emboviz onboarding wizard
         Goal: get you from "I have a model + rollouts" to "I have
         working diagnostics" in 5 minutes.
        ──────────────────────────────────────────────────────────────
    """))

    robot_idx = _ask("Which robot are you running on?", ROBOTS)
    robot = ROBOTS[robot_idx]

    model_keys = list(MODELS.keys())
    model_display = [MODELS[k].display for k in model_keys]
    model_idx = _ask("Which model architecture are you using?", model_display)
    model = MODELS[model_keys[model_idx]]

    checkpoint = _ask_text(
        "Where is your model checkpoint? (HF repo_id or local path)",
        default="(use the public reference checkpoint for now)",
    )

    dataset_idx = _ask("What format are your rollouts in?", DATASETS)
    dataset = DATASETS[dataset_idx]

    rollout_path = _ask_text(
        "Where are your rollouts? (file path or HF dataset id)",
        default="(I'll use a public reference dataset)",
    )

    # Emit setup
    print("\n──────────────────────────────────────────────────────────────")
    print(" Setup plan")
    print("──────────────────────────────────────────────────────────────")
    print(f"Robot:      {robot}")
    print(f"Model:      {model.display}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Format:     {dataset}")
    print(f"Rollouts:   {rollout_path}")
    print(f"\nVenv: a fresh Python {model.venv_python} environment "
          f"({'isolated from other models' if model.requires_separate_venv else 'shareable'})")
    if model.notes:
        print(f"\nHeads-up: {model.notes}")

    print("\nRun these commands:\n")
    for step in model.install_steps:
        print(f"    {step}")

    print("\nThen test the setup:")
    print(f"    emboviz diagnose --model {model.key} \\")
    print(f"        --checkpoint <your-checkpoint> \\")
    print(f"        --rollout <your-rollout> \\")
    print(f"        --suite quick_smoke")

    print("\nIf the smoke runs cleanly, run the full battery:")
    print(f"    emboviz diagnose --model {model.key} \\")
    print(f"        --checkpoint <your-checkpoint> \\")
    print(f"        --rollout <your-rollout> \\")
    print(f"        --suite full_profile")

    print("\nFor playback in Rerun (your rollout + Emboviz overlays):")
    print(f"    emboviz diagnose --model {model.key} ... --export-rerun out.rrd")
    print(f"    rerun out.rrd")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emboviz onboarding wizard — choose your stack, get a tailored install + run plan.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List supported models + robots and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Supported models:")
        for k, p in MODELS.items():
            print(f"  {k:15s}  — {p.display}")
        print("\nSupported robots:")
        for r in ROBOTS:
            print(f"  - {r}")
        return 0

    return run_wizard()


if __name__ == "__main__":
    sys.exit(main())
