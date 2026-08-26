# V146 sequential multi-scene curriculum

V146 carried one independent-head PD-PPO checkpoint through training scenes
1501, 1502, 1503, and 1504. Each stage used only that scene's policy-training
partition for optimization and calibration/validation partition for checkpoint
selection. Test metrics were not inspected between stages. The checkpoint after
scene1504 was frozen before evaluation on held-out scene1505.

The held-out gate failed. PD-PPO loss was `0.281380` versus `0.236810` for the
validation-selected static subset, an ordinary margin of `-0.044570`. Its
static-normalized macro was `1.007787` versus `0.898615`, a margin of
`-0.109172`. The best conventional dynamic loss was `0.270239`, so the PD-PPO
margin was also negative (`-0.011141`).

Behavior passed: no channel was always on or always off, five channels had
mid-range duty, switching was `0.018092` per step, and warm-up aborts were zero.
The result closes sequential curriculum and motivates episode-level interleaving
with one shared model and optimizer.
