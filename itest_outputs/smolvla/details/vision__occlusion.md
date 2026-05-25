# vision.occlusion

**diagnostic**: `sweep.occlusion`  
**severity**: 🟩 PASS  
**scalar score**: `0.1817` (higher is worse)  
**model**: `smolvla_base`  
**scene**: `lerobot:lerobot/aloha_sim_transfer_cube_human:0:0`

## Finding

AUC over sweep = 0.182; robust along vision.occlusion.

## Per-variant scores

| variant | score |
|---|---|
| `cov010` | 0.0381 |
| `cov025` | 0.2055 |
| `cov050` | 0.1980 |
| `cov075` | 0.1970 |

## Raw data (debugging)

<details><summary>show</summary>

- **levels**: [0.1, 0.25, 0.5, 0.75]
- **divergences**: [0.03807277977466583, 0.20546786487102509, 0.19803889095783234, 0.19699683785438538]
- **records**: [{'variant_id': 'cov010', 'level': 0.1, 'divergence': 0.03807277977466583, 'parameters': {'coverage': 0.1, 'bbox': None}}, {'variant_id': 'cov025', 'level': 0.25, 'divergence': 0.20546786487102509,...

</details>
