# vision.lighting

**diagnostic**: `counterfactual.lighting_shift`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.0000` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

Action divergence under lighting_shift averages 0.000, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the vision.lighting cue.

## Per-variant scores

| variant | score |
|---|---|
| `bright060` | 0.0000 |
| `bright140` | 0.0000 |
| `gamma070` | 0.0000 |
| `gamma140` | 0.0000 |
| `sat040` | 0.0000 |
| `sat160` | 0.0000 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'bright060', 'axis': 'vision.lighting', 'description': 'brightness ×0.60', 'instruction': 'put small spoon from basket to tray', 'parameters': {'kind': 'brightness', 'factor': 0.6},...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'vision.lighting'
- **perturber_affects**: ['images.primary']

</details>
