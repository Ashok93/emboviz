# language.empty

**diagnostic**: `counterfactual.empty`  
**severity**: 🟧 MODERATE  
**scalar score**: `0.6859` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

Mean divergence 0.686 is between noise (0.5) and grounded (2.0). Partial sensitivity to language.empty.

## Per-variant scores

| variant | score |
|---|---|
| `empty` | 0.6859 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'empty', 'axis': 'language.empty', 'description': '(empty instruction — pure vision)', 'instruction': '', 'parameters': {}, 'divergence': 0.68593829870224, 'baseline_action': [-0.00...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'language.empty'
- **perturber_affects**: ['instruction']

</details>
