# V317 decision-block credit diagnostic (2026-08-31)

## Protocol

V317 keeps the corrected six-channel scene and V302's forecast reward,
static-normalized loss, feasibility mask, evaluation windows, and 50k PPO
budget. It changes only the existing semi-Markov credit path: actor updates
receive the discounted reward accumulated between executable decisions. No
bandit action, test label, or additional teacher signal is used.

## Result

The decision-block credit intervention did not produce a reliable repair.
Against validation-selected static, ordinary-loss and macro wins were both
`0/2`, with mean margins (reference minus PD-PPO) of `-0.081869` and
`-0.073892`. Against feasible static, wins were `1/2` on both endpoints, but
the second seed degraded substantially. Ordinary-loss wins were `1/2` against
full-open, random, and round-robin, and `0/2` against AoI; macro wins were
`1/2` against all five dynamic/reference rows.

## Behaviour

Seed 6811 had zero always-on/off channels and four mid-duty channels. Seed 6812
had zero always-on, one always-off, and four mid-duty channels. Warm-up aborts
were zero; switching rates were `0.013750` and `0.018237` per step.

## Decision

Reject decision-block credit as a sufficient primary repair. It improves one
seed and preserves feasible execution, but does not pass static transfer or
the no-constant-channel behavior gate across both seeds. The next intervention
must address the state/action representation or initialization-transfer
problem, not stack another credit variant.

Raw evidence is stored in the two seed directories under `reports/`.
