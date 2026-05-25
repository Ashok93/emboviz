# language.refusal_absent

**diagnostic**: `counterfactual.refusal_absent`  
**severity**: 🟧 MODERATE  
**scalar score**: `0.6194` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

Mean divergence 0.619 is between noise (0.5) and grounded (2.0). Partial sensitivity to language.refusal_absent.

## Per-variant scores

| variant | score |
|---|---|
| `spoon_to_elephant` | 0.6495 |
| `spoon_to_trombone` | 0.5893 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'spoon_to_elephant', 'axis': 'language.refusal_absent', 'description': "target='elephant' (absent from scene)", 'instruction': 'put small elephant from basket to tray', 'parameters'...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.refusal_absent'
- **perturber_affects**: ['instruction']

</details>
