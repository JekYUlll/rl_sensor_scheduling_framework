# V301 low-entropy decision-only PD-PPO development result

## Configuration

- Corrected six-channel quality scene, budget 1.75, startup peak budget 2.15.
- Two development seeds: 6811 and 6812; policy seeds 6871 and 6872.
- Decision-only policy updates enabled; entropy coefficient reduced from 0.02 to 0.002.
- Forecast-loss reward, context encoder, temporal encoder, and ordinary feasibility constraints retained.

## Outcome

The variant is rejected. It does not improve the corrected-scene decision-only baseline.
The PD-PPO mean forecast loss is worse than the validation-selected static schedule in both
seeds (0/2 wins; mean margin -0.046824). The static-normalized event macro score also loses
in both seeds (0/2 wins; mean margin -0.114378). It does not beat the feasible static,
original dynamic, or full-open references in aggregate.

| seed | PD-PPO loss | static loss | PD-PPO macro | static macro | switches/step | always on | always off | mid-duty | aborts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6811 | 0.426813 | 0.358921 | 0.912743 | 0.688610 | 0.084310 | 0 | 0 | 6 | 0 |
| 6812 | 0.532480 | 0.506724 | 1.120961 | 1.116338 | 0.032132 | 0 | 0 | 6 | 0 |

## Interpretation

Lowering the entropy coefficient did not produce useful exploitation. The final training
entropy remained high (approximately 2.60--2.56 at the last update), while the policy lost
to static schedules. The result points away from entropy as the primary bottleneck and
supports checking reward scale, return/value conditioning, and checkpoint score alignment.

## Provenance

- Raw runs: `reports/v301_corrected_scene_decision_only_lowentropy_pdppo_dev_seed6811_b1p75_20260822/`
  and `reports/v301_corrected_scene_decision_only_lowentropy_pdppo_dev_seed6812_b1p75_20260822/`.
- Aggregate source: `seed_metrics.csv` in this directory.
- Evaluation command was executed by `run_v301_corrected_scene_decision_only_lowentropy_pdppo_dev_20260831.sh`.
