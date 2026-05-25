# language.ood_task

**diagnostic**: `counterfactual.ood_task`  
**severity**: 🟧 MODERATE  
**scalar score**: `0.5550` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

Mean divergence 0.555 is between noise (0.5) and grounded (2.0). Partial sensitivity to language.ood_task.

## Per-variant scores

| variant | score |
|---|---|
| `ood_0` | 0.5550 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'ood_0', 'axis': 'language.ood_task', 'description': 'OOD: press the red button', 'instruction': 'press the red button', 'parameters': {'task': 'press the red button'}, 'divergence'...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.ood_task'
- **perturber_affects**: ['instruction']

</details>
