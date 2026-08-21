# Results Convergence Memo - 2026-05-24

## Superseded Diagnostic Table

Do not use the 100k storm-window curriculum PD-PPO runs (`seed=41--45`) as a
main learned-policy result. The 2026-05-26 protocol audit found identical
storm training/evaluation starts in all five seeds, overlap in the later
full-distribution replay, reconstructed oracle overlap, and no declared
training-only normalisation. See
`reports/energy_account_protocol_audit_20260526/energy_account_protocol_audit_summary.json`.

Recorded descriptive values from the archived procedure:

- Storm-window: PD-PPO `0.4153 +/- 0.0051`.
- Storm-window AoI: `0.4176 +/- 0.0105`.
- Storm-window static projection: `0.4742 +/- 0.0236`.
- Full-distribution: PD-PPO `0.3155 +/- 0.0133`.
- Full-distribution AoI: `0.3168 +/- 0.0135`.
- Full-distribution static projection: `0.3318 +/- 0.0062`.

## Claim Boundary After Protocol Audit

Permitted as retrospective mechanism diagnostics only:

- Under its stored non-independent procedure, PD-PPO records lower storm-window
  loss than feasible static projection, round-robin, and random scheduling.
- These values motivate a corrected chronological split experiment; they do not
  measure held-out learned-policy performance.

Not supported:

- Submission-level learned-policy comparison in the energy-account setting.
- Held-out or full-distribution generalization from these archived runs.
- PD-PPO robustly dominates AoI.
- PD-PPO reliably learns clean event-triggered laser gating.
- Fixed-budget V3.1 results alone prove dynamic scheduling value; those results are
  compatible with strong static projection.

## Diagnostic Probe Summary

| probe | scenario | ppo_oracle_loss | aoi_oracle_loss | static_oracle_loss | ppo_minus_aoi | ppo_abort | ppo_power |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100k baseline | storm | 0.410586 | 0.413007 | 0.503249 | -0.00242123 | 5 | 1.01277 |
| 100k baseline | full | 0.310647 | 0.311557 | 0.335817 | -0.000910162 | 8 | 1.06262 |
| 300k baseline | storm | 0.405265 | 0.412465 | 0.502676 | -0.00719944 | 66 | 0.967311 |
| 300k baseline | full | 0.312219 | 0.311815 | 0.336666 | 0.000404318 | 75 | 0.989352 |
| event-gated actor | storm | 0.410535 | 0.412833 | 0.504105 | -0.0022986 | 38 | 0.968379 |
| event-gated actor | full | 0.312827 | 0.311714 | 0.335038 | 0.00111231 | 35 | 0.96514 |
| SOC auxiliary | storm | 0.410464 | 0.414366 | 0.503722 | -0.00390176 | 16 | 0.894915 |
| SOC auxiliary | full | 0.313817 | 0.312702 | 0.336291 | 0.00111539 | 12 | 0.988825 |
| SOC soft penalty | storm | 0.406949 | 0.412766 | 0.501387 | -0.00581667 | 9 | 0.996007 |
| event reward x1.5 | storm | 0.408807 | 0.411809 | 0.502055 | -0.00300188 | 7 | 1.04801 |
