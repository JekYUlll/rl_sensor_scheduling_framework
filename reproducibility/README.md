# Reproducibility blocks

Seed-level aggregates for the reported experiments, kept separate from the run artefacts
so that every summary number can be re-derived without the raw rollout files.

## `pdppo_evidence_20260718/`

24-seed evidence used for the main comparison. Each subdirectory is one evidence block:

| Directory | Content |
| --- | --- |
| `pdppo_clean_validation_frozen_24seed_20260718` | Primary macro comparison against validation-selected static and rule-based references, with bootstrap intervals and diagnostic boundaries |
| `pdppo_clean_matched_reward_24seed_20260718` | Reward-matched control (same reward signal, different learner) |
| `pdppo_matched_dqn_clean_24seed_20260718` | Matched DQN control |
| `pdppo_framework_baselines_clean_24seed_20260718` | Full-open / periodic / round-robin / random baselines |
| `pdppo_secondary_forecaster_24seed_20260718` | Paired downstream-forecaster evaluation |
| `pdppo_clean_mechanism_24seed_20260718` | Policy behaviour and per-sensor duty diagnostics |
| `pdppo_warning_information_20260729` | Warning lead-time and information-content diagnostics |
| `pdppo_full_final_partition_24seed_20260718` | Sensitivity to the final evaluation partition |
| `full_partition_source_rows` | Per-seed rows underlying the partition sensitivity block |

Protocol files (`*_protocol.json`) record the seed list, the comparison construction and
the reference-selection rule, so the aggregates can be audited without rerunning training.

## `pdppo_evidence_20260710/`

Earlier validation-frozen blocks (24-seed mechanism ablation, 6-seed schedule robustness,
scenario-balance variants). Superseded by the 20260718 blocks but kept because some
reports cite them.

All numbers are computed from simulated data; no field measurements are included.
