# vision.sensor_noise

**diagnostic**: `counterfactual.gaussian_noise`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.2132` (lower is worse)  
**model**: `smolvla_base`  
**scene**: `lerobot:lerobot/aloha_sim_transfer_cube_human:0:0`

## Finding

Action divergence under gaussian_noise averages 0.213, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the vision.sensor_noise cue.

## Per-variant scores

| variant | score |
|---|---|
| `std005` | 0.1163 |
| `std015` | 0.2509 |
| `std030` | 0.2724 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'std005', 'axis': 'vision.sensor_noise', 'description': 'gaussian noise σ=5', 'instruction': 'pick up the red cube and transfer it to the other arm', 'parameters': {'sigma': 5.0}, '...
- **baseline_instruction**: 'pick up the red cube and transfer it to the other arm'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'vision.sensor_noise'
- **perturber_affects**: ['images.primary']

</details>
