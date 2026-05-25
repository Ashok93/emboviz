# vision.lighting

**diagnostic**: `counterfactual.lighting_shift`  
**severity**: 🟥 CRITICAL  
**scalar score**: `0.1471` (lower is worse)  
**model**: `smolvla_base`  
**scene**: `lerobot:lerobot/aloha_sim_transfer_cube_human:0:0`

## Finding

Action divergence under lighting_shift averages 0.147, below the noise floor (0.5). The model produces nearly identical actions across variants — it isn't using the vision.lighting cue.

## Per-variant scores

| variant | score |
|---|---|
| `bright060` | 0.0945 |
| `bright140` | 0.1584 |
| `gamma070` | 0.1649 |
| `gamma140` | 0.1722 |
| `sat040` | 0.1610 |
| `sat160` | 0.1314 |

## Raw data (debugging)

<details><summary>show</summary>

- **variants**: [{'variant_id': 'bright060', 'axis': 'vision.lighting', 'description': 'brightness ×0.60', 'instruction': 'pick up the red cube and transfer it to the other arm', 'parameters': {'kind': 'brightness...
- **baseline_instruction**: 'pick up the red cube and transfer it to the other arm'
- **noise_floor**: 0.5
- **grounded_threshold**: 2.0
- **perturber_axis**: 'vision.lighting'
- **perturber_affects**: ['images.primary']

</details>
