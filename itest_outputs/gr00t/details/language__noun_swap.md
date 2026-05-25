# language.noun_swap

**diagnostic**: `counterfactual.noun_swap`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.1428` (lower is worse)  
**model**: `GR00T-N1.7-3B`  
**scene**: ``

## Finding

Action divergence under noun_swap averages 0.143, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.noun_swap cue.

## Per-variant scores

| variant | score |
|---|---|
| `spoon_to_fork` | 0.0518 |
| `spoon_to_knife` | 0.3257 |
| `spoon_to_spatula` | 0.0510 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'spoon_to_fork', 'axis': 'language.noun_swap', 'description': 'spoon → fork', 'instruction': 'put small fork from basket to tray', 'parameters': {'target': 'spoon', 'swap_to': 'fork...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.noun_swap'
- **perturber_affects**: ['instruction']

</details>
