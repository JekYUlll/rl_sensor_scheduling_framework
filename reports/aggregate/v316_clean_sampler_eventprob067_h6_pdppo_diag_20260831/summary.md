# V316 clean balanced-start diagnostic

## Protocol

V316 is the valid single-variable follow-up to V311. It restores V311's
`staticnorm_subtype` reward normalization and `lambda_warmup_abort=1.0`, and
changes only the training-start sampler from `event_start_prob=1.0` to `0.67`.
The corrected six-channel scene, feasible-only hard forecast-value teacher,
teacher lookahead 4, minimum dwell 6, forecast horizon 6, and final windows
are unchanged. Seeds are 6811 and 6812.

## Result

Positive margins are reference loss minus PD-PPO loss. V316 lost to both
static references in both seeds, and also lost to full-open, AoI, random, and
round-robin in both seeds on ordinary forecast loss and static-normalized
macro loss.

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
zero, but the required no-constant-channel behaviour gate failed in both
seeds; switching rates were `0.000579` and `0.002895` per step.

## Decision

Balanced training starts are rejected as a sufficient state-distribution
repair. The clean comparison shows that changing the sampler alone does not
restore closed-loop transfer or deployment-like channel usage. No fresh final
evaluation should be launched from V316. The next intervention must change a
different clean layer, such as closed-loop return/credit alignment, while
preserving the feasibility mask and forecast-loss objective.
