# V343 Frozen Policy-Alignment Audit

## Scope

This offline audit evaluates the frozen V342 decision-only BC policies on the
exact V342 final-test starts for V338 scenes 6871 and 6872. At each comparable
state, all feasible candidate masks are scored with the frozen forecast
evaluator using snapshot/restore. Candidate costs are not exposed to the
policy and do not alter training, rewards, or selection.

## Results

| scene | audit rows | valid rows | mean cost regret | mean relative regret | best-action match | mean action rank / 22 | mean entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6871 | 2,303 | 2,302 | 0.089073 | 0.461962 | 13.07% | 11.47 | 2.604 |
| 6872 | 2,303 | 2,302 | 0.337326 | 1.522831 | 8.25% | 13.54 | 2.818 |
| pooled | 4,606 | 4,604 | 0.213200 | 0.992397 | 10.66% | 12.50 | 2.711 |

One terminal row per scene had no finite candidate cost and was excluded from
cost/rank statistics. The policy uses deterministic top-action selection;
therefore the selected-action probability equals the maximum action
probability in these rows.

## Interpretation

The frozen policies do not map their observations to the locally best
forecast-value action: the selected action is in the top two candidates in
15.28% of pooled rows and matches the best candidate in 10.66% of rows. This
is direct evidence of policy-to-action alignment failure, not evidence that
the recalibrated scene lacks dynamic action value. The independent V338
same-state audit found all 22 actions selected at least once and mean
best-versus-second-best relative gaps of 0.022620 and 0.023032 for scenes
6871 and 6872, respectively.

The next method decision is therefore to address observation-to-action
mapping and long-horizon credit assignment with a clean learner design. No
bandit-dependent reward, imitation target, residual action, or final-test
label is justified by this audit.

## Provenance

- Remote source: `reports/v343_policy_alignment_audit_seed6871_b1p75_20260822/`
  and the corresponding seed6872 directory on `remote-gpu`.
- Frozen checkpoints: V342 decision-only BC diagnostic, policy seeds 7351 and
  7352.
- Exact starts: nested `final_test.eval_starts` in each V342 metadata file.
- Local raw copies: `seed6871/policy_alignment_audit.csv` and
  `seed6872/policy_alignment_audit.csv` in this directory.
