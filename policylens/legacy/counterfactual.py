"""Counterfactual instruction harness — does the VLA actually listen?

Method (per LIBERO-Plus, IGAR, NOTICE):
  Hold the image and the rest of the rollout fixed; only change the
  *instruction*. If the predicted action is statistically indistinguishable
  across instructions, the model is **language-blind on this rollout** —
  it's reading from vision alone. That is the documented #1 VLA failure
  mode and the founder's spoon/fork scenario, made measurable.

Key metrics returned:
  • Per-frame action divergence (L2 in action space) between baseline and
    each variant.
  • Instruction Sensitivity Score (ISS) = mean per-frame divergence per
    variant. ISS ≈ 0 across variants ⇒ vision-blind. High ISS ⇒ grounded.
  • Intra-baseline noise floor (variance in baseline alone — should be 0 for
    deterministic decoding, but kept for safety / sanity check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from policylens.dataset_bridge import BridgeEpisode
from policylens.openvla import OpenVLAInference


@dataclass
class CounterfactualResult:
    """Output of running OpenVLA on the same scene with several instructions."""

    base_instruction: str
    variant_instructions: list[str]
    frame_indices: list[int]
    # Shape: (n_variants + 1, n_frames, action_dim).
    # Row 0 is the baseline (base_instruction); rows 1..N are variants.
    actions: np.ndarray
    # divergences[i][k] = ||actions[i+1, k] - actions[0, k]||₂
    divergences: np.ndarray         # (n_variants, n_frames)
    instruction_sensitivity: dict[str, float]   # variant → mean divergence

    def headline(self) -> str:
        worst_variant, worst_iss = max(
            self.instruction_sensitivity.items(), key=lambda kv: kv[1]
        )
        weakest_variant, weakest_iss = min(
            self.instruction_sensitivity.items(), key=lambda kv: kv[1]
        )
        return (
            f"max ISS = {worst_iss:.3f} (variant: '{worst_variant}')  ·  "
            f"min ISS = {weakest_iss:.3f} (variant: '{weakest_variant}')"
        )


def run_counterfactuals(
    vla: OpenVLAInference,
    episode: BridgeEpisode,
    variant_instructions: list[str],
    frame_indices: list[int] | None = None,
) -> CounterfactualResult:
    """Run the policy on the same set of frames with each variant instruction.

    The baseline run uses `episode.instruction` (whatever Bridge provided).
    Variants are stored in the returned `variant_instructions` field in the
    same order they were passed.
    """
    if frame_indices is None:
        frame_indices = list(range(episode.num_frames))

    instructions_all = [episode.instruction] + list(variant_instructions)
    n_inst = len(instructions_all)
    n_frames = len(frame_indices)
    action_dim = int(episode.expert_actions.shape[-1])

    actions = np.zeros((n_inst, n_frames, action_dim), dtype=np.float32)

    for i, instr in enumerate(instructions_all):
        label = f"cf [{i}/{n_inst-1}] '{instr[:40]}'"
        for fi, t in enumerate(tqdm(frame_indices, desc=label, leave=False)):
            pred = vla.predict(episode.images[t], instr)
            actions[i, fi] = pred.action

    # Divergences: variants minus baseline.
    divergences = np.linalg.norm(actions[1:] - actions[0:1], axis=-1)  # (V, F)
    iss = {
        variant_instructions[i]: float(divergences[i].mean())
        for i in range(divergences.shape[0])
    }

    return CounterfactualResult(
        base_instruction=episode.instruction,
        variant_instructions=list(variant_instructions),
        frame_indices=frame_indices,
        actions=actions,
        divergences=divergences,
        instruction_sensitivity=iss,
    )


def classify_grounding(
    result: CounterfactualResult,
    blindness_threshold: float = 0.05,
    grounded_threshold: float = 0.30,
) -> tuple[str, str]:
    """Return (verdict_tag, verdict_text).

    Verdict logic:
      • If max ISS across ALL variants is below `blindness_threshold` →
        actions are essentially identical regardless of instruction ⇒
        LANGUAGE-BLIND.
      • If mean ISS is above `grounded_threshold` ⇒ GROUNDED (model uses
        language meaningfully).
      • In between ⇒ PARTIAL.

    Thresholds are in Bridge action units (7-DOF, ~m and rad). A divergence
    of 0.05 corresponds to ~5 cm or ~3° — within measurement noise for these
    actions. A divergence of 0.30 is a clearly different motion.
    """
    iss_values = list(result.instruction_sensitivity.values())
    if not iss_values:
        return ("unknown", "No counterfactual variants run.")

    max_iss = max(iss_values)
    mean_iss = float(np.mean(iss_values))

    if max_iss < blindness_threshold:
        return (
            "language_blind",
            f"Language blindness confirmed. Maximum divergence across "
            f"{len(iss_values)} instruction variants is {max_iss:.3f}, "
            f"below the noise threshold ({blindness_threshold}). The model "
            f"produces statistically indistinguishable actions whether the "
            f"instruction names the correct object, a non-present object, "
            f"or is empty.",
        )
    if mean_iss >= grounded_threshold:
        return (
            "grounded",
            f"Properly grounded. Mean instruction sensitivity = "
            f"{mean_iss:.3f}, above the grounded threshold ({grounded_threshold}). "
            f"The model's actions track the instruction.",
        )
    return (
        "partial",
        f"Partial grounding. Mean ISS = {mean_iss:.3f}, max = {max_iss:.3f}. "
        f"The model uses some instructions but ignores others.",
    )
