# vision.occlusion

**diagnostic**: `sweep.occlusion`  
**severity**: 🟩 PASS  
**scalar score**: `0.3862` (higher is worse)  
**model**: `GR00T-N1.7-3B`  
**scene**: ``

## Finding

AUC over sweep = 0.386; robust along vision.occlusion.

## Per-variant scores

| variant | score |
|---|---|
| `cov010` | 0.0264 |
| `cov025` | 0.0796 |
| `cov050` | 0.7544 |
| `cov075` | 0.3560 |

## Raw data (debugging)

<details><summary>show</summary>

- **levels**: [0.1, 0.25, 0.5, 0.75]
- **divergences**: [0.026440443471074104, 0.07955334335565567, 0.7544344067573547, 0.3560222387313843]
- **records**: [{'variant_id': 'cov010', 'level': 0.1, 'divergence': 0.026440443471074104, 'parameters': {'coverage': 0.1, 'bbox': None}}, {'variant_id': 'cov025', 'level': 0.25, 'divergence': 0.07955334335565567...

</details>
