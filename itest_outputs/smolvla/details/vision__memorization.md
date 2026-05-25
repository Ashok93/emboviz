# vision.memorization

**diagnostic**: `memorization_test`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.1252` (lower is worse)  
**model**: `smolvla_base`  
**scene**: `lerobot:lerobot/aloha_sim_transfer_cube_human:0:0`

## Finding

Even with the target masked, the model produces an action that is nearly identical to the original (Δ=0.125) and has substantial magnitude (0.917). It's memorizing the trajectory rather than reading the scene.

## Per-variant scores

| variant | score |
|---|---|
| `diff_vs_baseline` | 0.1252 |
| `diff_vs_blank` | 0.0948 |
| `action_magnitude` | 0.9173 |

## Raw data (debugging)

<details><summary>show</summary>

- **baseline_action**: [-0.1460820436477661, -0.8169873952865601, 0.22811028361320496, 0.34582188725471497, -0.11253318935632706, -0.22790485620498657]
- **action_target_masked**: [-0.1559811532497406, -0.778413712978363, 0.29582375288009644, 0.2502610385417938, -0.1250952184200287, -0.21302998065948486]
- **action_blank_scene**: [-0.12425778806209564, -0.7035348415374756, 0.25361090898513794, 0.2585124671459198, -0.1048847883939743, -0.2018951177597046]

</details>
