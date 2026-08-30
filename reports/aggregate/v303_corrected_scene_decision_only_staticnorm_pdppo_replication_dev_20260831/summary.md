# V303 static-normalized reward replication result

V303 kept V302 unchanged and changed only policy initialization seeds to
`6891--6892`. It does not confirm V302.

- Ordinary-loss wins: validation static `0/2`, feasible static `0/2`,
  full-open `0/2`, AoI `0/2`, random `1/2`, round-robin `1/2`.
- Mean ordinary margins in that order were `-0.047333`, `-0.023196`,
  `-0.036967`, `-0.024266`, `-0.006779`, and `-0.001435`.
- Static-normalized macro wins: validation static `1/2`, feasible static
  `1/2`, full-open `0/2`, AoI `0/2`, random `1/2`, round-robin `1/2`.
- Mean macro margins were `-0.113674`, `-0.027758`, `-0.103985`,
  `-0.079730`, `-0.005408`, and `-0.031253`.
- Behavior: zero warm-up aborts and zero always-on channels. Both seeds had
  five mid-duty channels and one always-off channel; switching rates were
  `0.060067` and `0.033000`.

## Decision

Reject V303 as a confirmation of V302. The negative static transfer and
one always-off channel in both seeds show that V302's positive dynamic result
was not stable enough for promotion. No final 24-seed launch is justified
before auditing training/rollout action-value alignment and seed sensitivity.

Raw runs are under the two `reports/v303_corrected_scene_decision_only_staticnorm_pdppo_replication_dev_seed6811/6812_b1p75_20260822/`
directories; the aggregate source is `seed_metrics.csv` in this directory.
