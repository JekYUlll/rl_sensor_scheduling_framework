# Validation-Frozen Mechanism Ablation

All macro margins use validation-partition static-candidate denominators. Positive margins are validation-selected static loss minus PD-PPO loss.

| Variant | Macro wins | Step wins | Zero-abort | Mean macro margin [95% CI] | Paired delta vs full [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full PD-PPO | 24/24 | 24/24 | 24/24 | 0.0778 [0.0665, 0.0896] | -- |
| No imitation guide | 24/24 | 24/24 | 24/24 | 0.0739 [0.0623, 0.0858] | -0.0040 [-0.0159, 0.0069] |
| No event context auxiliary | 24/24 | 22/24 | 24/24 | 0.0748 [0.0596, 0.0914] | -0.0030 [-0.0149, 0.0108] |
| No balanced training loss | 24/24 | 24/24 | 24/24 | 0.0703 [0.0585, 0.0827] | -0.0075 [-0.0186, 0.0018] |
