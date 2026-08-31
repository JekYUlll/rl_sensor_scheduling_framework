# V337 policy-seed variance screen

## Protocol

- Fixed clean V319 configuration, corrected scenes 6811/6812.
- Fresh policy seeds 7261--7264, with two policy seeds per scene.
- Static-normalized forecast reward, hard forecast-value pretraining,
  decision-only PPO updates, minimum dwell 6, and valid same-scene validation
  checkpoint selection.

## Predictive result

The value is `baseline - custom_ppo`; positive values indicate a PD-PPO win.

| Comparator | Ordinary-loss wins | Mean ordinary margin | Macro wins | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 0/4 | -0.081477 | 0/4 | -0.167606 |
| feasible static projected | 0/4 | -0.057341 | 0/4 | -0.081690 |
| AoI | 0/4 | -0.058410 | 0/4 | -0.133661 |
| round-robin | 1/4 | -0.035579 | 0/4 | -0.085185 |
| random | 1/4 | -0.040923 | 1/4 | -0.059339 |
| full-open unconstrained | 0/4 | -0.071111 | 0/4 | -0.157917 |

## Operational result

- Warm-up aborts: 0/4.
- Constant-on channels: 0/4.
- Constant-off channels: 1/4.
- Mid-duty channel counts: 6, 5, 4, and 4.
- Switching rates: 0.025474, 0.016790, 0.016500, and 0.012737 per step.

## Decision

The screen confirms substantial policy-seed sensitivity but does not identify
a positive seed-stable configuration. V337 is excluded from primary evidence;
the current clean V319 branch is not ready for final evaluation.
