# language.ood_task

**diagnostic**: `counterfactual.ood_task`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0433` (lower is worse)  
**model**: `smolvla_base`  
**scene**: `lerobot:lerobot/aloha_sim_transfer_cube_human:0:0`

## Finding

Action divergence under ood_task averages 0.043, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.ood_task cue.

## Per-variant scores

| variant | score |
|---|---|
| `ood_0` | 0.0433 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'ood_0', 'axis': 'language.ood_task', 'description': 'OOD: press the red button', 'instruction': 'press the red button', 'parameters': {'task': 'press the red button'}, 'divergence'...
- **baseline_instruction**: 'pick up the red cube and transfer it to the other arm'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.ood_task'
- **perturber_affects**: ['instruction']

</details>
