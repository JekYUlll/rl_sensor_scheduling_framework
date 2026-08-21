# Evidence Ledger for the ESWA Full Rewrite

Date: 2026-05-25

Note: this file was created during the earlier CRST rewrite track, but the
current target journal is *Expert Systems with Applications* (ESWA). Treat the
filename and older path references as historical; the ledger below controls
ESWA-facing claims unless superseded by later evidence.

## Purpose

This ledger defines which completed artifacts may support the rewritten
*Expert Systems with Applications* manuscript. It replaces prose inherited from
earlier drafts as the claim-control source. The paper is an applied intelligent
sensing-system study of prediction-driven, power-constrained sensor scheduling in
a blowing-snow microclimate digital-twin benchmark; it is not a field-validation
paper and does not claim a complete physical battery model.

## Study Identity

Proposed research question:

> Under what energy-availability conditions does forecast-oriented adaptive sensor
> scheduling provide value beyond fixed sensor allocations for simulated Antarctic
> blowing-snow monitoring with heterogeneous sensor warm-up delays?

Proposed manuscript role of the algorithm:

- PD-PPO is the tested prediction-driven scheduler used to evaluate whether a
  forecast-relevant dynamic scheduling opportunity is learnable under operating
  constraints.
- The principal ESWA-facing finding is about constrained intelligent scheduling:
  when static allocation explains most forecast variance, it is a strong
  engineering baseline; when event/regime-conditioned specialist information is
  required, a learned state-dependent scheduler can create forecast value.

## Evidence Streams That Must Not Be Mixed

| Stream | Status in rewritten paper | Reason |
| --- | --- | --- |
| V3.1 fixed-budget PD-PPO experiments | Historical/same-protocol diagnostic only; superseded by corrected rerun | Completed 3-budget by 10-seed output, but prior selection, training and evaluation did not implement the claimed held-out protocol. |
| V3.1 split-protocol rerun | Primary fixed-budget evidence | Completed `3`-budget by `10`-seed final-test evaluation with explicit chronological partitions, validation-selected static comparison and non-overlapping final-test windows. |
| Physical-event v4 energy-account experiments | Mechanism/opportunity diagnostic only; learned-policy protocol audit failed | The oracle diagnostic illustrates an energy-account opportunity, but all 5 curriculum runs evaluate on training starts and the no-retrain full evaluation also intersects training/oracle windows. |
| V3.1-aligned A1/A2/H1 experiments | Supporting algorithm diagnostic | Completed, but only selected component conclusions are statistically supported. |
| Route A frozen-forecast mainline documented in `AGENTS.md` | Excluded from this manuscript unless explicitly recast as separate work | Different truth/split/configuration and DQN/ensemble-oracle lineage; combining it with V3.1/PD-PPO would make methods and result attribution internally inconsistent. |
| Failed physical-event v2/v3/v4 probes and 200k/300k variants | Development history, normally excluded from main text | Useful for internal design traceability, but not final evidence unless a limitation requires a brief mention. |

## E1a: Legacy Fixed-Budget V3.1 Benchmark

Submission status: **not primary evidence**. Retain the locked tables as a
reproducible development diagnostic only; the completed split-protocol result in
E1b replaces them.

Artifacts:

- `reports/v31_s2_main/v31_s2_main_stats.csv`
- `reports/v31_s2_main/v31_s2_significance.csv`
- `reports/v31_s2_main/v31_s2_event_fraction_stats.csv`
- `reports/v31_s2_main/done/` (30 completion markers)
- `paper/tables/main_results_v31.tex`
- `paper/tables/condition_results_v31.tex`

Design:

- Budgets: `B = 1.65`, `1.70`, `1.75`.
- Independent seeds: `41--50`, giving `n = 10` per budget/policy.
- Main metric: forecast-weighted MAE (FW-MAE), lower is better.
- Full observation is an unconstrained diagnostic reference.
- Feasible static projection is a fixed-priority projected baseline; it is not an
  exhaustive global static optimum.

Locked central values:

| Policy | B = 1.65 | B = 1.70 | B = 1.75 |
| --- | ---: | ---: | ---: |
| Full observation | 0.1487 +/- 0.0118 | 0.1493 +/- 0.0123 | 0.1502 +/- 0.0123 |
| Static projection | 0.1593 +/- 0.0116 | 0.1597 +/- 0.0119 | 0.1612 +/- 0.0115 |
| PD-PPO | 0.1628 +/- 0.0140 | 0.1620 +/- 0.0142 | 0.1661 +/- 0.0145 |
| Round-robin | 0.1671 +/- 0.0138 | 0.1674 +/- 0.0141 | 0.1687 +/- 0.0137 |
| AoI | 0.1700 +/- 0.0139 | 0.1844 +/- 0.0191 | 0.1933 +/- 0.0184 |
| Random feasible | 0.1803 +/- 0.0142 | 0.1862 +/- 0.0173 | 0.1914 +/- 0.0163 |

Safe claims:

- In the stored same-sequence diagnostic, across all tested fixed budgets, PD-PPO
  has lower mean FW-MAE than round-robin, AoI, and random feasible scheduling.
- At `B = 1.70`, its mean FW-MAE is approximately 3.2% below round-robin and
  12.1% below AoI, while approximately 1.5% above static projection.
- The fixed-budget results are compatible with near-static allocation being highly
  effective; they do not establish a value for frequent dynamic switching.
- Improvement over AoI and random is Bonferroni-significant in the completed report;
  the round-robin margin is smaller and not uniformly significant.

Claims prohibited from E1 alone:

- These values represent chronologically held-out or non-overlapping testing.
- These values are submission-level generalization evidence.
- Adaptive switching is superior to static allocation.
- PD-PPO is statistically superior to every comparator at every budget.
- Static projection is the globally optimal static solution.

### E1 Protocol Audit and Replacement

Artifacts:

- `reports/v31_s2_protocol_audit/v31_s2_protocol_audit_summary.json`
- `scripts/56_v31_protocol_audit.py`
- `scripts/57_v31_independent_replay.py`
- `scripts/58_v31_split_protocol_run.py`

Audit result:

- Reconstructed candidate-prior windows overlap final evaluation windows in
  `21/30` stored S2 runs.
- Final evaluation windows overlap one another in `21/30` runs; candidate-prior
  windows overlap one another in `9/30` runs.
- PPO training in the locked outputs was not restricted to a disjoint train
  partition.
- A new-truth replay smoke test of the old checkpoint is inadequate as a repair:
  the old frozen TCN ranked full observation worse than the selected static
  comparator and PD-PPO on that test truth.

Replacement gate:

- The split-protocol runner reserves chronological `oracle_pretrain`, `rl_train`,
  `validation`, and `final_test` partitions at ratios `35/50/7.5/7.5`.
- Candidate-prior construction and actor normalisation are confined to `rl_train`;
  static comparator selection occurs in `validation`; final-test windows are
  non-overlapping and unconsulted during selection.
- A server gate for `B=1.70`, seed `41` completed under
  `reports/v31_split_protocol_gate/budget1p70_seed41`. Its final test ranks full
  observation best (`FW-MAE=0.1114`), followed by validation-selected static
  (`0.1195`), PD-PPO (`0.1222`), round-robin (`0.1243`), random (`0.1301`), and
  AoI (`0.1319`). The gate passes protocol/oracle sanity but does not show
  PD-PPO superiority over selected static allocation.
- The replacement `3`-budget by `10`-seed split-protocol grid completed
  successfully in server session `v31_split_main_20260526` (`30/30` runs,
  driver exit `0`) and was synchronized locally on 2026-05-26 under
  `reports/v31_split_protocol_main`.

## E1b: Corrected Fixed-Budget Split-Protocol Result

Artifacts:

- `reports/v31_split_protocol_main/v31_s2_main_stats.csv`
- `reports/v31_split_protocol_main/v31_s2_significance.csv`
- `reports/v31_split_protocol_main/v31_s2_budget_check.csv`
- `reports/v31_split_protocol_main/raw/*/split_protocol_manifest.json`

Protocol verification:

- All `30/30` manifests pass chronological partition ordering and non-overlapping
  final-test-window checks.
- The validation-selected static mask is
  `met_station_core|radiometer_basic|snow_particle_counter` in `26/30` runs and
  the corresponding laser-disdrometer subset in `4/30` runs.
- Full observation has the lowest mean FW-MAE at every budget and is the
  run-level minimum in `29/30` runs; one `B=1.75`, seed `46` run has selected
  static slightly below full observation (`0.138730` versus `0.138961`).

Corrected fixed-budget final-test values:

| Policy | `B = 1.65` | `B = 1.70` | `B = 1.75` |
| --- | ---: | ---: | ---: |
| Full observation | `0.1218 +/- 0.0103` | `0.1225 +/- 0.0095` | `0.1231 +/- 0.0094` |
| Validation-selected static | `0.1315 +/- 0.0108` | `0.1329 +/- 0.0108` | `0.1311 +/- 0.0104` |
| PD-PPO | `0.1304 +/- 0.0101` | `0.1334 +/- 0.0110` | `0.1316 +/- 0.0104` |
| Round-robin | `0.1400 +/- 0.0121` | `0.1408 +/- 0.0119` | `0.1390 +/- 0.0113` |
| AoI | `0.1391 +/- 0.0097` | `0.1432 +/- 0.0100` | `0.1435 +/- 0.0118` |
| Random feasible | `0.1421 +/- 0.0101` | `0.1425 +/- 0.0098` | `0.1425 +/- 0.0112` |

Permitted primary claims:

- Under the corrected held-out protocol, PD-PPO has lower mean FW-MAE than
  round-robin, AoI and random feasible scheduling at all three budgets; every
  one of those comparisons remains significant after the recorded Bonferroni
  correction (`p_adj <= 0.0235`).
- PD-PPO and the validation-selected static comparator are statistically
  indistinguishable in this experiment: PD-PPO is better in `6/10` seeds only at
  `B=1.65` and in `4/10` seeds at `B=1.70` and `B=1.75`; each corrected
  static comparison has `p_adj = 1.0`.
- The fixed-instantaneous-budget result therefore supports effective allocation
  close to a selected static subset, not a general dynamic-over-static advantage.

Manuscript integration status (2026-05-26):

- `paper/tables/main_results_v31.tex`, `paper/tables/condition_results_v31.tex`
  and `paper/tables/physical_unit_mae.tex` have been regenerated from the
  split-protocol final-test artifacts.
- The paper power-error and behavior-diagnostic figures now read
  `reports/v31_split_protocol_main` and display the validation-selected static
  comparator.
- The current abstract/results/discussion/conclusion use E1b values rather than
  the superseded E1a values.

## E2: Energy-Account Opportunity Diagnostic

Artifacts:

- `reports/physical_event_v4_energy_cal_h092_cap180_storm_tcn_b120_seed41/oracle_lift_summary.json`
- `paper/tables/energy_account_storm_oracle.tex`
- `docs/05-24-claim-audit.md`

Design:

- Normalized energy-account setting: instantaneous `B = 1.20`,
  harvest `h = 0.92`, capacity `C = 180`, reserve `20`.
- Evaluation windows: six highest-event 1024-step windows in the formal
  physical-event truth sequence.
- This is an oracle/reference-policy diagnostic using event labels, not an
  operational policy deployment test.
- The oracle is trained and evaluated on the same source truth sequence; this
  section can establish a simulated mechanism opportunity only, not held-out
  learned-policy performance or generalization.

Locked central values:

| Reference policy | Overall oracle loss | Event loss | Non-event loss | Min SOC | Guard drops | Warm-up aborts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic snow core to event laser + FC4 | 0.4169 | 0.3190 | 0.5442 | 48.48 | 0 | 0 |
| Static snow core | 0.4248 | 0.3517 | 0.5199 | 180.00 | 0 | 0 |
| Static laser diagnostic | 0.4413 | 0.3535 | 0.5554 | 20.00 | 438 | 0 |

Safe claims:

- Under the simplified calibrated account and event-rich windows, the dynamic
  reference reduces overall loss by about 1.9% and event loss by about 9.3%
  relative to static snow core.
- Long-term energy accounting changes feasibility in a way a per-step budget alone
  does not capture: an instantaneously feasible static laser diagnostic is clipped
  by the energy guard.

Claims prohibited:

- This is a validated physical battery/harvesting model.
- The event label is available in deployed real time.
- The diagnostic demonstrates full-distribution dynamic superiority.

## E3: Learned Policy in the Calibrated Energy-Account Regime

Submission status: **historical/same-protocol diagnostic only; independence audit
failed**. The values below may guide a corrected experiment design, but cannot
carry a learned-policy result or held-out generalization claim in the rewritten
manuscript.

Artifacts:

- `reports/energy_account_convergence_20260524/energy_account_main_summary.csv`
- `reports/energy_account_convergence_20260524/energy_account_main_long.csv`
- `paper/tables/energy_account_curriculum_results.tex`
- `scripts/52_energy_account_convergence_assets.py`
- `reports/energy_account_protocol_audit_20260526/energy_account_protocol_audit_summary.json`
- `reports/energy_account_protocol_audit_20260526/energy_account_protocol_audit_by_seed.csv`
- `scripts/60_energy_account_protocol_audit.py`

Design:

- 100k-step storm-window curriculum PD-PPO runs, seeds `41--45`.
- Storm-window evaluation and full-distribution no-retrain evaluation.

Protocol audit result:

- All `5/5` storm runs record identical `train_start_indices` and
  `eval_start_indices`; the storm score is a replay over training windows.
- All `5/5` no-retrain full-distribution evaluations overlap the recorded
  default-length training windows and the storm evaluation windows.
- Reconstructing deterministic oracle sampling from saved metadata shows oracle
  windows overlapping storm and full-distribution evaluation in `5/5` seeds.
- Two of five full-distribution evaluations contain internal window overlap, and
  none of the five metadata files declares training-only normalisation statistics.

Locked central values:

| Scenario | PD-PPO | AoI | Static projection | Round-robin | Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Storm windows | 0.4153 +/- 0.0051 | 0.4176 +/- 0.0105 | 0.4742 +/- 0.0236 | 0.4451 +/- 0.0167 | 0.4565 +/- 0.0140 |
| Full distribution | 0.3155 +/- 0.0133 | 0.3168 +/- 0.0135 | 0.3318 +/- 0.0062 | 0.3375 +/- 0.0195 | 0.3431 +/- 0.0188 |

Win counts for PD-PPO:

- Storm windows: `5/5` versus static projection, round-robin and random; `3/5`
  versus AoI.
- Full distribution: `4/5` versus AoI and static projection; `5/5` versus
  round-robin and random.

Permitted retrospective diagnostic statements only:

- Under the stored, non-independent curriculum/evaluation procedure, PD-PPO has
  lower recorded loss than static projection, round-robin and random on selected
  storm windows.
- AoI remains a strong comparator; PD-PPO has a small mean advantage with non-uniform
  per-seed wins.
- These observations motivate a corrected energy-account split-protocol test; they
  are not paper-level comparative performance evidence.

Prohibited claims:

- Held-out, independent or full-distribution generalization from the stored energy
  curriculum outputs.
- Learned-policy superiority in the calibrated energy-account regime unless a new
  split-protocol run supplies it.
- Robust superiority to AoI.
- Clean learned event-triggered laser activation. Existing mechanism diagnosis reports
  an approximately neutral laser event/non-event activation ratio.

Replacement status (2026-05-26):

- Implemented `scripts/61_energy_account_split_protocol_run.py` with disjoint
  oracle-pretrain, RL-train, validation-static-selection and final-test partitions,
  training-only normalisation and non-overlapping conditional evaluation windows.
- An initial seed-41 gate was aborted before policy results after its prepared
  `clustered` truth exposed temporal coverage bias: selected mean event rates
  were approximately `0.331`, `0.271`, and `0.276` in RL-train, validation,
  and final-test despite the intended storm-conditional role. Its intermediate
  artifacts are quarantined under `reports/energy_account_split_protocol_invalid_clustered_gate/`.
- The replacement runner now defaults to the V3.1 `semi_markov` event generator.
  A gate may launch only after its manifest confirms that disjoint validation and
  final-test segments represent the declared event regime.
- The replacement seed-41 preflight passed: partition-wide event rates are
  approximately `0.321/0.307/0.307/0.300` for oracle/RL/validation/final,
  while independently selected final conditional windows average `0.521`.
  Training is running in `tmux` session `energy_split_semimarkov_gate_20260526`
  under `reports/energy_account_split_protocol_gate_semimarkov/budget1p20_seed41`.
- Pending completion of that gate, the current paper text excludes the archived
  learned-policy curriculum table from comparative evidence and reports E2 only
  as a reference-policy mechanism diagnostic.

## E4: Supporting Ablation Evidence

Artifacts:

- `reports/v31_ablation_aligned/v31_aligned_a1_stats.csv`
- `reports/v31_ablation_aligned/v31_aligned_a2_stats.csv`
- `reports/v31_ablation_aligned/v31_aligned_h1_stats.csv`
- `reports/v31_ablation_aligned/v31_aligned_completion_check.csv`

Safe interpretation:

- A1/A2/H1 completion is recorded as `80/80`, `40/40`, and `45/45`.
- Removing AWBC and oracle prior jointly worsens FW-MAE from
  `0.1629 +/- 0.0137` to `0.1853 +/- 0.0209` and is significant under the recorded
  Bonferroni threshold.
- MaskedActor-only is also significantly worse than the full variant.
- Individual removal of ActionEmbedding, EventAwareCritic, or action mask is not
  significant in this experiment batch and must not be described as independently
  proven performance drivers.

## Verification Still Required Before Final Manuscript

- Server verification completed on 2026-05-25:
  - Local and remote `reports/v31_s2_main/v31_s2_main_stats.csv` share SHA-256
    `74e666b891ebe44fd0c542f7f6d7e87df3a9e5e83b88a6e5245f217cacad1534`;
    the remote run directory contains `30` completion markers.
  - The energy-account summary is locally generated from synced run artifacts rather
    than stored as a remote aggregate. The five storm metric CSV inputs and thirty
    full-distribution rollout NPZ inputs used by
    `scripts/52_energy_account_convergence_assets.py` share the same combined
    local/remote SHA-256 manifest digest:
    `f0e1c24228efd908fdd773f853fbaba2c08fd6fd6a1ffb3d55dad341a6f93e23`.
- Verify table generation scripts do not silently alter captions or result labels.
- Energy-account protocol audit completed on 2026-05-26: the five-seed learned
  curriculum table is diagnostic only and must not be imported into primary
  manuscript results without corrected retraining.
- Verify the reference list for every factual field/datasheet claim.
- Decide how generated truth sequences and aggregate result data will satisfy
  current ESWA/Elsevier data availability requirements.
- Have all final claims independently reviewed after full rewrite, not only against
  the historical shortened manuscript.

## First Independent Structural Review Gate (Zeno, 2026-05-25)

The first structure reviewer concluded that incremental repair is not sufficient.
Binding issues to resolve before drafting:

- Current title/abstract foreground PD-PPO although the defensible result is a
  regime-dependent energy-availability diagnosis.
- Fixed-budget and energy-account evidence cannot answer dynamic-vs-static value
  conclusively while the comparator is only `feasible_static_projected`; with eight
  logical channels, an exhaustive feasible static comparison should be produced if
  the stored artifacts/code permit it.
- The old text describes both chronological within-seed splitting and disjoint
  evaluation seeds; the rewritten protocol must derive the exact claim from code and
  metadata, not inherit either statement unverified.
- Truth-derived event context and event-selected storm windows are mechanism/upper
  bound diagnostics unless an operational event proxy is evaluated.
- A single frozen TCN both trains/scores the policies, so claims must remain tied to
  that forecast-surrogate evaluation unless independent predictor/physical-endpoint
  evidence is available.
- FW-MAE weight choices require explicit rationale and component reporting.
- The algorithm component narrative must be limited to AWBC/prior evidence supported
  by the completed ablation results.
- ESWA/Elsevier packaging deficiencies in the archived manuscript include
  abstract/highlight compliance, missing CRediT, incomplete data-policy
  treatment, and insufficiently specific AI-use declaration.
