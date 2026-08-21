# Flexible-subset v2 30k development replication

The v2 scene passed the two co-primary static-baseline gates in one of three
development seeds. Seed 402 improved mean loss by 0.031940 and macro loss by
0.209488. Seed 403 lost both endpoints, while seed 404 improved macro loss by
0.004232 but lost mean loss by 0.005471. PD-PPO also beat the best conventional
dynamic policy on both endpoints only in seed 402.

All runs were feasible and had no always-on channel. Behaviour was not stable:
seed 403 used only four subsets, left one channel inactive, and switched at
0.007237 per step. Seed 404 used eleven subsets, had no always-on or always-off
channel, and switched at 0.019540 per step.

This 30k configuration therefore fails the replication gate and must not be
expanded to confirmation seeds. The next bounded diagnostic holds the v2 scene,
costs, budget, and training scaffold fixed and extends seed 403 to 100k PPO
steps. Its purpose is to test whether the failure is primarily undertraining or
requires a training-design correction.
