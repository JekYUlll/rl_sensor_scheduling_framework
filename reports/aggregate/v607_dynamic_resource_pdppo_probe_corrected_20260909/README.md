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
for PD-PPO versus `1.034383` for validation static. The behavior gate also
fails: seed 7184 has one warm-up abort, and every seed has one always-on
sensor. These results are diagnostic only and do not support a positive PPO
claim.

The probe remains useful because it verifies that the corrected baselines have
distinct trajectories and that the 32-subset dynamic-resource interface is
being exercised. No duty-constrained baseline rows were enabled in this probe.
