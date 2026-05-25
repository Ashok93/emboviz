# vision.occlusion

**diagnostic**: `sweep.occlusion`  
**severity**: 🟧 MODERATE  
**scalar score**: `0.8118` (higher is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

AUC over sweep = 0.812; moderate sensitivity along vision.occlusion.

## Per-variant scores

| variant | score |
|---|---|
| `cov010` | 0.6431 |
| `cov025` | 1.0186 |
| `cov050` | 0.6160 |
| `cov075` | 0.9738 |

## Raw data (debugging)

<details><summary>show</summary>

- **levels**: [0.1, 0.25, 0.5, 0.75]
- **divergences**: [0.6431252956390381, 1.0185590982437134, 0.6160358786582947, 0.97380530834198]
- **records**: [{'variant_id': 'cov010', 'level': 0.1, 'divergence': 0.6431252956390381, 'parameters': {'coverage': 0.1, 'bbox': None}}, {'variant_id': 'cov025', 'level': 0.25, 'divergence': 1.0185590982437134, '...

</details>
