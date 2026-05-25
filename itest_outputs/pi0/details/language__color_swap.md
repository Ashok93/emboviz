# language.color_swap

**diagnostic**: `counterfactual.color_swap`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0953` (lower is worse)  
**model**: `pi0_aloha_sim`  
**scene**: ``

## Finding

Action divergence under color_swap averages 0.095, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.color_swap cue.

## Per-variant scores

| variant | score |
|---|---|
| `red_to_blue` | 0.1142 |
| `red_to_green` | 0.0765 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'red_to_blue', 'axis': 'language.color_swap', 'description': 'red → blue', 'instruction': 'grab the blue cube and place it in the bin', 'parameters': {'from': 'red', 'to': 'blue'}, ...
- **baseline_instruction**: 'grab the red cube and place it in the bin'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.color_swap'
- **perturber_affects**: ['instruction']

</details>
