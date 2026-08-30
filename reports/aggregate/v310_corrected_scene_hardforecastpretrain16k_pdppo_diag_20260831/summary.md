# V310 corrected-scene hard forecast-value pretraining diagnostic

- Hard forecast-value pretraining: 16,384 training-partition steps, temperature not used, `TOTAL_TIMESTEPS=0` (zero PPO updates).
- Scene seeds 6811/6812; policy seeds 6951/6952.
- Labels are the feasible candidate argmin under the frozen forecaster; this is distinct from legacy subtype-static `hard` AWBC mode.

## Results
- Seed 6811: custom ordinary `0.452110049`, macro `1.003076706`, abort `0`, always-on/off `0/0`, mid-duty `5`, switches/step `0.016066001`.
- Seed 6812: custom ordinary `0.508228258`, macro `0.999781995`, abort `0`, always-on/off `0/3`, mid-duty `3`, switches/step `0.008684325`.

## Win counts (lower loss is better)
- `validation_selected_static`: ordinary 0/2, mean margin (baseline - custom) -0.047346992; macro 1/2, mean margin -0.098955218.
- `feasible_static_projected`: ordinary 0/2, mean margin (baseline - custom) -0.023210615; macro 1/2, mean margin -0.013038725.
- `full_open_unconstrained`: ordinary 0/2, mean margin (baseline - custom) -0.036980871; macro 1/2, mean margin -0.089266199.
- `aoi`: ordinary 1/2, mean margin (baseline - custom) -0.024279862; macro 1/2, mean margin -0.065010311.
- `random`: ordinary 1/2, mean margin (baseline - custom) -0.006793012; macro 1/2, mean margin +0.009311045.
- `round_robin`: ordinary 1/2, mean margin (baseline - custom) -0.001448814; macro 1/2, mean margin -0.016534080.

## Decision
- **Rejected as a primary improvement.** Hard forecast-value labels were fit accurately (`0.830/0.834`) and each custom rollout passed abort and constant-channel checks, but the learned action mapping did not transfer to held-out closed-loop forecast quality: it lost to validation static and every tested dynamic reference in both scenes, and also lost to full-open.
- Do not promote or expand this pretraining branch. The failure is now isolated from label softness and points to state-distribution or closed-loop return/credit mismatch.
