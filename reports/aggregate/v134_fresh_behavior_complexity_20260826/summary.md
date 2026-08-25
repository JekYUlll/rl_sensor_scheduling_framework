# V134 fresh-confirmation behavior-complexity audit

## Scope

- Frozen PD-PPO rollouts for fresh scene seeds 1401--1424.
- Each rollout contains 2,304 executed decisions over the six physical-system
  logical channels.
- Audit command: `scripts/71_v31_behavior_complexity_audit.py` with its locked
  default thresholds.
- This audit was performed after the 24-seed predictive aggregate. It was not
  used for checkpoint selection or tuning.

## Aggregate result

- Fixed-like execution: `0/24` seeds.
- Simple deterministic-cycle execution: `0/24` seeds.
- Event- or subtype-dependent behavior: `23/24` seeds.
- Full behavior-complexity gate: `22/24` seeds.
- Unique executed masks: mean `11.46`, range `6--18`.
- Dominant-mask share: mean `0.442`, range `0.206--0.618`.
- Mask entropy: mean `2.315` bits, range `1.469--3.195` bits.
- Transition entropy: mean `2.839` bits, range `1.704--3.899` bits.
- Switching rate: mean `0.02805` per step, range `0.00883--0.04053`.

## Gate exceptions

- Seed 1407 used six masks and remained event/subtype dependent, but its mask
  entropy was `1.469` bits, just below the prespecified `1.50`-bit threshold.
- Seed 1415 used nine masks and was neither fixed-like nor cyclic, but its
  event/subtype conditional differences remained below the prespecified state-
  dependence thresholds.

## Decision

The fresh policies are broadly dynamic and do not collapse to a fixed subset or
a deterministic rotation. The strict state-dependent behavior claim is supported
in 22/24 seeds, not 24/24. Seeds 1401--1424 remain sealed confirmation evidence;
the two exceptions will not be used for post-hoc tuning.
