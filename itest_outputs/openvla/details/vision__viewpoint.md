# vision.viewpoint

**diagnostic**: `counterfactual.viewpoint_jitter`  
**severity**: 🟧 MODERATE  
**scalar score**: `0.5581` (lower is worse)  
**model**: `openvla-7b`  
**scene**: `bridge_v2:0:12`

## Finding

Mean divergence 0.558 is between noise (0.5) and grounded (2.0). Partial sensitivity to vision.viewpoint.

## Per-variant scores

| variant | score |
|---|---|
| `rot-10` | 0.8684 |
| `rot-5` | 0.5822 |
| `rot+5` | 0.5798 |
| `rot+10` | 0.7415 |
| `shiftx-20` | 0.0000 |
| `shiftx+20` | 0.6447 |
| `zoom90` | 0.4529 |
| `zoom110` | 0.5951 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'rot-10', 'axis': 'vision.viewpoint', 'description': 'rotate -10°', 'instruction': 'put small spoon from basket to tray', 'parameters': {'kind': 'rotation', 'deg': -10}, 'divergence...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'vision.viewpoint'
- **perturber_affects**: ['images.primary']

</details>
