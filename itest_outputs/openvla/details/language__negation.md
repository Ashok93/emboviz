# language.negation

**diagnostic**: `counterfactual.negation`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0000` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

Action divergence under negation averages 0.000, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.negation cue.

## Per-variant scores

| variant | score |
|---|---|
| `do_not` | 0.0000 |
| `never` | 0.0000 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'do_not', 'axis': 'language.negation', 'description': "'do not' prefix", 'instruction': 'do not put small spoon from basket to tray', 'parameters': {'prefix': 'do not'}, 'divergence...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.negation'
- **perturber_affects**: ['instruction']

</details>
