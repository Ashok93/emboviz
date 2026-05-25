# language.negation

**diagnostic**: `counterfactual.negation`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0823` (lower is worse)  
**model**: `pi0_aloha_sim`  
**scene**: ``

## Finding

Action divergence under negation averages 0.082, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.negation cue.

## Per-variant scores

| variant | score |
|---|---|
| `do_not` | 0.0943 |
| `never` | 0.0704 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'do_not', 'axis': 'language.negation', 'description': "'do not' prefix", 'instruction': 'do not grab the red cube and place it in the bin', 'parameters': {'prefix': 'do not'}, 'dive...
- **baseline_instruction**: 'grab the red cube and place it in the bin'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.negation'
- **perturber_affects**: ['instruction']

</details>
