# vision.distractor

**diagnostic**: `sweep.distractor_inject`  
**severity**: 🟩 PASS  
**scalar score**: `0.0000` (higher is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

AUC over sweep = 0.000; robust along vision.distractor.

## Per-variant scores

| variant | score |
|---|---|
| `n1` | 0.0000 |
| `n3` | 0.0000 |
| `n5` | 0.0000 |

## Raw data (debugging)

<details><summary>show</summary>

- **levels**: [1.0, 3.0, 5.0]
- **divergences**: [0.0, 0.0, 0.0]
- **records**: [{'variant_id': 'n1', 'level': 1.0, 'divergence': 0.0, 'parameters': {'n_distractors': 1}}, {'variant_id': 'n3', 'level': 3.0, 'divergence': 0.0, 'parameters': {'n_distractors': 3}}, {'variant_id':...

</details>
