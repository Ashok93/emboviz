# vision.memorization

**diagnostic**: `memorization_test`  
**severity**: 🟧 MODERATE  
**scalar score**: `0.0000` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

With target masked, action stays similar (Δ=0.000). Partial memorization.

## Per-variant scores

| variant | score |
|---|---|
| `diff_vs_baseline` | 0.0000 |
| `diff_vs_blank` | 0.1561 |
| `action_magnitude` | 0.0864 |

## Raw data (debugging)

<details><summary>show</summary>

- **baseline_action**: [-0.00937943160533905, -0.010784436948597431, 0.033536043018102646, -0.0007774750120006502, 0.04878938943147659, -0.061315037310123444, 0.0]
- **action_target_masked**: [-0.00937943160533905, -0.010784436948597431, 0.033536043018102646, -0.0007774750120006502, 0.04878938943147659, -0.061315037310123444, 0.0]
- **action_blank_scene**: [0.004041023086756468, 0.029038021340966225, 0.03067796491086483, -0.0806017518043518, 0.028023583814501762, 0.06440521031618118, 0.0]

</details>
