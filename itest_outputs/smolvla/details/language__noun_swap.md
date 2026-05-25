# language.noun_swap

**diagnostic**: `counterfactual.noun_swap`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.1646` (lower is worse)  
**model**: `smolvla_base`  
**scene**: `lerobot:lerobot/aloha_sim_transfer_cube_human:0:0`

## Finding

Action divergence under noun_swap averages 0.165, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.noun_swap cue.

## Per-variant scores

| variant | score |
|---|---|
| `cube_to_block` | 0.2198 |
| `cube_to_ball` | 0.1436 |
| `cube_to_sphere` | 0.1304 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'cube_to_block', 'axis': 'language.noun_swap', 'description': 'cube → block', 'instruction': 'pick up the red block and transfer it to the other arm', 'parameters': {'target': 'cube...
- **baseline_instruction**: 'pick up the red cube and transfer it to the other arm'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.noun_swap'
- **perturber_affects**: ['instruction']

</details>
