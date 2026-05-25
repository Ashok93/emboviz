# language.empty

**diagnostic**: `counterfactual.empty`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0664` (lower is worse)  
**model**: `GR00T-N1.7-3B`  
**scene**: ``

## Finding

Action divergence under empty averages 0.066, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the language.empty cue.

## Per-variant scores

| variant | score |
|---|---|
| `empty` | 0.0664 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'empty', 'axis': 'language.empty', 'description': '(empty instruction — pure vision)', 'instruction': '', 'parameters': {}, 'divergence': 0.06641194969415665, 'baseline_action': [0....
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.empty'
- **perturber_affects**: ['instruction']

</details>
