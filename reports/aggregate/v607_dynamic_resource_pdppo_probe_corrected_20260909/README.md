# Corrected V607 dynamic-resource PPO probe

This is a four-seed probe under the V602 development effective-resource
envelope, with 32 candidate subsets, six-step dwell, online resource-state
features, and forecast-loss reward. The baseline execution correction is
included in commit `848e65a`.

The first V607 probe is excluded because dynamic fallback projection collapsed
the baseline trajectories. The authoritative 7181 output is stored remotely
under `seed7181_retry`; the other authoritative outputs are `seed7182`--
`seed7184`.

## Result

PD-PPO beats validation static on the macro endpoint in `1/4` seeds and the
best original dynamic heuristic in `1/4` seeds. Mean macro loss is `1.051377`
for PD-PPO versus `1.034383` for validation static. Seed 7184 has one
warm-up abort. These results are diagnostic only and do not support a positive
PPO claim.

The probe remains useful because it verifies that the corrected baselines have
distinct trajectories and that the 32-subset dynamic-resource interface is
being exercised. No duty-constrained baseline rows were enabled in this probe.

## Constraint audit correction

`dynamic_constraint_audit.csv` and `.md` recompute optional-channel cost from
the frozen resource traces. The mandatory `cr1000xe_backbone` is reported
separately from optional duty counts. All corrected rollouts selected feasible
optional subsets: there were zero dynamic-cost violations, and the number of
currently feasible optional masks ranged from 8 to 28 over evaluated rows.
The earlier statement that every PD-PPO rollout had one always-on sensor was a
metric-labeling artifact because it counted the mandatory backbone. Excluding
that channel, PD-PPO had zero optional always-on channels, optional always-off
counts of 1, 1, 0, and 0 for seeds 7181--7184, and optional mid-duty counts of
4, 3, 4, and 3. The seed-7184 warm-up abort remains real.
