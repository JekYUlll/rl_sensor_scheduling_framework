# V608 decision-block-credit audit

## Configuration

- Seed: `7182`
- Training: `100352` timesteps, `49` updates
- Changes relative to V607: `decision_only_policy_updates` and
  `decision_block_credit`; forecast-loss reward, masked categorical PPO,
  dynamic resource trace, six-step dwell, and candidate set were unchanged.
- Evaluation rows: `6144`

## Performance

The saved `v2_custom_ppo_metrics.csv` reports:

| policy | forecast loss | macro event loss | warm-up aborts | switches/step |
|---|---:|---:|---:|---:|
| PD-PPO | 0.797708 | 0.988881 | 0 | 0.016469 |
| feasible static | 0.743940 | 0.922661 | 0 | 0.001899 |
| AoI | 0.752199 | 0.931105 | 1 | 0.004341 |
| random | 0.780534 | 0.963419 | 0 | 0.048158 |
| round robin | 0.883248 | 1.101820 | 1 | 0.029519 |

Relative to the feasible static policy, PD-PPO has mean forecast-loss margin
`-0.053768` and macro margin `-0.066219`; negative means that PD-PPO is worse.
The full-open and feasible-static rows are identical in this rollout, so the
full-open row is not an independent performance advantage here.

## Operational audit

The mandatory `cr1000xe_backbone` is excluded from optional-channel duty
counts. For PD-PPO: optional always-on `0`, optional always-off `1`, and
optional mid-duty `4`. All selected actions satisfy the dynamic optional
effective-power budget (`2.15 W`); the feasible optional-mask count ranges
from `8` to `28`. The policy therefore remains executable and non-collapsed,
but it does not yet exploit the available dynamic value geometry.

## Decision

V608 is a negative single-seed development result for the credit-assignment
hypothesis. It does not authorize a confirmation wave or a positive claim.
The result supports stopping this specific intervention before adding further
learner patches. The physics-first route remains the main evidence boundary.
