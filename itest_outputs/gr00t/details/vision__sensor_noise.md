# vision.sensor_noise

**diagnostic**: `counterfactual.gaussian_noise`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.2460` (lower is worse)  
**model**: `GR00T-N1.7-3B`  
**scene**: ``

## Finding

Action divergence under gaussian_noise averages 0.246, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the vision.sensor_noise cue.

## Per-variant scores

| variant | score |
|---|---|
| `std005` | 0.1512 |
| `std015` | 0.3084 |
| `std030` | 0.2783 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'std005', 'axis': 'vision.sensor_noise', 'description': 'gaussian noise σ=5', 'instruction': 'put small spoon from basket to tray', 'parameters': {'sigma': 5.0}, 'divergence': 0.151...
- **baseline_instruction**: 'put small spoon from basket to tray'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'vision.sensor_noise'
- **perturber_affects**: ['images.primary']

</details>
