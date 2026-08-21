# Flexible-subset v6 development replication

## Configuration

- Six physical-system logical channels.
- All 24 power-feasible subsets at `B=1.25`, with no required channel or
  cardinality cap.
- Strong physical-prototype behavior-cloning warm start followed by PPO with
  no continuing AWBC.
- Static-normalized subtype forecast reward and separate actor/critic gradient
  clipping.
- Development seeds 405--407; these are not final confirmation seeds.

## Result

PD-PPO beat the validation-selected static subset on both endpoints in 2/3
seeds. Mean margins were +0.008988 for mean forecast loss and +0.089263 for the
macro endpoint. Against the strongest conventional dynamic policy in each seed,
PD-PPO won both endpoints in 2/3 seeds; mean margins were +0.004541 and
-0.000391.

The policy executed 19, 8, and 11 distinct subsets. Seed 405 used all six
channels at intermediate duty. Seed 406 never selected the laser and used FC4
at 1.6% duty, while seed 407 selected the laser at 2.9% duty. No warm-up abort
occurred, and switching rates ranged from 0.0320 to 0.0459 per step.

## Decision

V6 establishes that the arbitrary-subset formulation can learn broad,
state-dependent schedules and pass all gates in an individual seed. The
replication does not support freezing the configuration for fresh final
evaluation because the dynamic macro margin is not positive on average and
action coverage varies materially across seeds. The next bounded investigation
must target cross-seed actor/action-coverage stability without changing the
scene, power geometry, reward definition, or baseline protocol.
