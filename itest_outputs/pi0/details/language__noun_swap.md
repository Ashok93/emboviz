# language.noun_swap

**diagnostic**: `counterfactual.noun_swap`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0884` (lower is worse)  
**model**: `pi0_aloha_sim`  
**scene**: ``

## Finding

Action divergence under noun_swap averages 0.088, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.noun_swap cue.

## Per-variant scores

| variant | score |
|---|---|
| `cube_to_block` | 0.1195 |
| `cube_to_ball` | 0.0796 |
| `cube_to_sphere` | 0.0660 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'cube_to_block', 'axis': 'language.noun_swap', 'description': 'cube → block', 'instruction': 'grab the red block and place it in the bin', 'parameters': {'target': 'cube', 'swap_to'...
- **baseline_instruction**: 'grab the red cube and place it in the bin'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.noun_swap'
- **perturber_affects**: ['instruction']

</details>
