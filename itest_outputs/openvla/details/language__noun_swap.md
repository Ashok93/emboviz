# language.noun_swap

**diagnostic**: `counterfactual.noun_swap`  
**severity**: 🟧 MODERATE  
**scalar score**: `0.5677` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

Mean divergence 0.568 is between noise (0.5) and grounded (2.0). Partial sensitivity to language.noun_swap.

## Per-variant scores

| variant | score |
|---|---|
| `spoon_to_fork` | 0.5354 |
| `spoon_to_knife` | 0.5782 |
| `spoon_to_spatula` | 0.5895 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'spoon_to_fork', 'axis': 'language.noun_swap', 'description': 'spoon → fork', 'instruction': 'put small fork from basket to tray', 'parameters': {'target': 'spoon', 'swap_to': 'fork...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.noun_swap'
- **perturber_affects**: ['instruction']

</details>
