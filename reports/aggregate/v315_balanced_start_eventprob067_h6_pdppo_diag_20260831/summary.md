# V315 balanced training-start diagnostic

## Protocol

V315 was intended to retain the V311 corrected-scene, feasible-only
hard forecast-value pretraining protocol and change only the training start
sampler (`event_start_prob=1.0` to `0.67`). The actual metadata shows two
additional configuration differences: V315 used unnormalised forecast loss
(`reward_loss_normalization=none` rather than V311's `staticnorm_subtype`) and
`lambda_warmup_abort=0.08` rather than `1.0`. The teacher horizon remained 4,
the executable minimum dwell remained 6, and the forecast horizon remained 6.
The two scene seeds were 6811 and 6812.

## Result

Under this configuration drift, PD-PPO lost to the
validation-selected static schedule and the feasible static schedule in both
seeds, and also lost to the full-open, AoI, random, and round-robin references
in both seeds on ordinary forecast loss and static-normalized macro loss.

Mean margins are defined as reference loss minus PD-PPO loss; positive values
would favor PD-PPO.

| Reference | Ordinary wins | Macro wins | Mean ordinary margin | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 0/2 | 0/2 | -0.093409 | -0.157037 |
| feasible static projected | 0/2 | 0/2 | -0.095565 | -0.107725 |
| full-open unconstrained | 0/2 | 0/2 | -0.143490 | -0.243828 |
| AoI | 0/2 | 0/2 | -0.099802 | -0.160858 |
| random | 0/2 | 0/2 | -0.079080 | -0.088338 |
| round-robin | 0/2 | 0/2 | -0.080216 | -0.113008 |

## Behaviour

Seed 6811 had 1 always-on, 3 always-off and 2 mid-duty channels. Seed 6812
had 0 always-on, 1 always-off and 5 mid-duty channels. Warm-up aborts were
zero, but the always-on/off gate failed in both seeds. Switching remained very
low (`0.000579` and `0.002895` per step).

## Decision

This run is not a valid single-variable test of balanced training starts and
must not be used to claim that `event_start_prob=0.67` caused the degradation.
It is retained as an audit of configuration drift. A clean rerun must restore
V311's reward normalization and warm-up coefficient before testing any sampler
change. The next scientific decision remains closed-loop return/state
alignment, not further tuning of this invalid comparison.
