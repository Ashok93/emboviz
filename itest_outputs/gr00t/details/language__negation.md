# language.negation

**diagnostic**: `counterfactual.negation`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0575` (lower is worse)  
**model**: `GR00T-N1.7-3B`  
**scene**: ``

## Finding

Action divergence under negation averages 0.057, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.negation cue.

## Per-variant scores

| variant | score |
|---|---|
| `do_not` | 0.0700 |
| `never` | 0.0450 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'do_not', 'axis': 'language.negation', 'description': "'do not' prefix", 'instruction': 'do not put small spoon from basket to tray', 'parameters': {'prefix': 'do not'}, 'divergence...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.negation'
- **perturber_affects**: ['instruction']

</details>
