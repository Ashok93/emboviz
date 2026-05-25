# language.refusal_absent

**diagnostic**: `counterfactual.refusal_absent`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.1204` (lower is worse)  
**model**: `pi0_aloha_sim`  
**scene**: ``

## Finding

Action divergence under refusal_absent averages 0.120, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.refusal_absent cue.

## Per-variant scores

| variant | score |
|---|---|
| `cube_to_elephant` | 0.1420 |
| `cube_to_trombone` | 0.0987 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'cube_to_elephant', 'axis': 'language.refusal_absent', 'description': "target='elephant' (absent from scene)", 'instruction': 'grab the red elephant and place it in the bin', 'param...
- **baseline_instruction**: 'grab the red cube and place it in the bin'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.refusal_absent'
- **perturber_affects**: ['instruction']

</details>
