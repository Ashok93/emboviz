# language.empty

**diagnostic**: `counterfactual.empty`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0970` (lower is worse)  
**model**: `smolvla_base`  
**scene**: `lerobot:lerobot/aloha_sim_transfer_cube_human:0:0`

## Finding

Action divergence under empty averages 0.097, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.empty cue.

## Per-variant scores

| variant | score |
|---|---|
| `empty` | 0.0970 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'empty', 'axis': 'language.empty', 'description': '(empty instruction — pure vision)', 'instruction': '', 'parameters': {}, 'divergence': 0.09700632840394974, 'baseline_action': [-0...
- **baseline_instruction**: 'pick up the red cube and transfer it to the other arm'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.empty'
- **perturber_affects**: ['instruction']

</details>
