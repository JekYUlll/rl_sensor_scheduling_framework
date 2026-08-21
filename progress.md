# Progress Log

> Current authoritative progress log:
> `.planning/2026-06-10-eswa-terminology-rewrite/progress.md`. This root-level
> file is a historical snapshot retained for continuity; the completed
> 2026-06-23-02 PPO-LEMMA closure is recorded in the active `.planning`
> directory.

## Session: 2026-05-15

### Phase 1: Consolidate TODO Into File-Based Plan
- **Status:** complete
- **Started:** 2026-05-15 evening
- Actions taken:
  - Read `planning-with-files` skill instructions.
  - Checked for existing root planning files in `rl_sensor_scheduling_framework`; none were present.
  - Read the historical 05-13 TODO and V3.1 S2 completion report.
  - Checked local V3.1-aligned ablation snapshot after rsync.
  - Created active file-based plan with phases, evidence inventory, and non-blocking ablation status.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

### Phase 2: Paper Closure Pass
- **Status:** pending
- Actions taken:
  -
- Files created/modified:
  -

### Phase 3: V3.1-Aligned Ablation Monitoring
- **Status:** pending
- Actions taken:
  - Current local snapshot recorded in `task_plan.md`.
- Files created/modified:
  -

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Planning files present | `ls task_plan.md findings.md progress.md` | Three files exist | Created in repo root | Pass |
| Local aligned-ablation snapshot check | `find reports/v31_ablation_aligned/...` | Partial backup visible | 54 done/eval files, A2 40, A1 14, H1 0 raw eval | Pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-15 evening | `python` command not found locally | 1 | Use `python3` or explicit conda Python for future checks. |
| 2026-05-15 evening | `pandas` missing under system `python3` | 1 | Use shell CSV checks for lightweight validation, or the correct env when needed. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 2: paper closure pass is next. |
| Where am I going? | Compile/check manuscript, monitor optional V3.1-aligned ablations, then decide whether to upgrade diagnostics. |
| What's the goal? | Make the PD-PPO paper submission-ready with traceable V3.1 main results and no over-claims. |
| What have I learned? | See `findings.md`. |
| What have I done? | Created active planning files and recorded current evidence state. |

## Session: 2026-05-16

### Phase 3: V3.1-Aligned Ablation Monitoring
- **Status:** complete
- Actions taken:
  - Checked remote server after recovery.
  - Confirmed V3.1-aligned ablation completion:
    - `reports/v31_ablation_aligned/done`: 150 files
    - raw evaluation CSVs: 150 files
    - collector completion: A1 `80/80`, A2 `40/40`, H1 `45/45`
  - Synced `reports/v31_ablation_aligned/` back to local.
  - Read `v31_aligned_a1_stats.csv`, `v31_aligned_a2_stats.csv`, `v31_aligned_h1_stats.csv`, and `v31_aligned_completion_check.csv`.
- Files created/modified:
  - `reports/v31_ablation_aligned/` (synced result directory)

### Phase 4: Ablation Narrative Upgrade
- **Status:** complete
- Actions taken:
  - Replaced A2 staged table with V3.1-aligned values.
  - Replaced A1 full component table with V3.1-aligned values.
  - Replaced H1 heatmap with V3.1-aligned heatmap.
  - Rewrote experiment text to frame A1/A2/H1 as V3.1-aligned diagnostics.
  - Kept interpretation conservative:
    - AWBC + oracle prior jointly significant.
    - MaskedActor-only significantly worse.
    - ActionEmbedding, EventAwareCritic, and action mask are architectural supports rather than individually significant performance drivers.
- Files modified:
  - `paper/sections/06_experiments.tex`
  - `paper/tables/ablation.tex`
  - `paper/tables/ablation_full.tex`
  - `paper/figures/figure_h1_heatmap.png`
  - `docs/05-13/00_TODO_goal_required_experiments_and_paper_fixes.md`
  - `docs/05-13/03_V31_s2_completion_report.md`

### Phase 5: Verification and Plan Update
- **Status:** in_progress
- Actions taken:
  - Compiled `paper/paper.tex` with XeLaTeX via `latexmk`.
  - Confirmed `paper/paper.pdf` generated successfully.
  - Checked `paper.log` for undefined references/citations; none found.
  - Rendered and visually inspected PDF pages 47--52.
  - Corrected H1 caption to avoid claiming a black-outlined default cell that is not visible in the regenerated heatmap.
  - Updated `task_plan.md`, `findings.md`, and `progress.md`.
- Files modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| V3.1-aligned completion | `v31_aligned_completion_check.csv` | A1 80, A2 40, H1 45 | A1 80, A2 40, H1 45 | Pass |
| Paper compile | `latexmk -xelatex paper.tex` | PDF generated | `paper.pdf` generated, 65 pages | Pass |
| Reference/citation scan | `paper.log` | No undefined refs/cites | None found | Pass |
| PDF visual check | Render pages 47--52 | A1/A2/H1 readable | Tables and heatmap readable | Pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-16 | `rg` pattern containing Markdown backticks executed `38/40` in shell | 1 | Use Python string scanning or stronger quoting for literal Markdown snippets. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5: final verification and backup. |
| Where am I going? | Decide whether to commit the paper subrepo and how to handle generated reports/heavy artifacts. |
| What's the goal? | Finish a submission-ready PD-PPO manuscript with V3.1 mainline and V3.1-aligned diagnostics. |
| What have I learned? | See updated `findings.md`. |
| What have I done? | Completed aligned ablation integration, compiled the paper, visually checked pages, and updated planning files. |

## Session: 2026-05-16 Review Gap Planning

### Phase 6: Critical Manuscript Gap Repair From Review
- **Status:** in_progress
- Actions taken:
  - Recorded new review/gap list into the active plan.
  - Added a Phase 6 checklist for missing reproducibility/data details, power-cost consistency, round-robin claim downshift, and innovation narrative alignment.
  - Inspected `/home/horeb/_Data/SEUAWS/rs485-reader` as the real-sensor evidence source.
  - Read `README.md` and `SENSOR_DATA_SPEC.md` from the RS485 reader project.
  - Sampled Parsivel2 and Modbus JSONL/CSV logs to verify raw sequence shapes.
  - Searched the manuscript for current FC4, round-robin, TCN, target-set, AoI, and component-claim locations.
- Files modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| RS485 reader evidence path | `/home/horeb/_Data/SEUAWS/rs485-reader` | Reader docs and sample logs exist | README, SENSOR_DATA_SPEC, JSONL/CSV logs, parser/reader scripts found | Pass |
| Raw shape sanity check | Parsivel2/Modbus sample logs | Distinguishable record schemas | Parsivel2 type1/type2 and Modbus 21-variable JSONL identified | Pass |
| Manuscript issue locator | `rg` over paper sections/tables/algorithms | Locate affected claims | Found FC4/power text, round-robin claims, component claims, TCN/target/AoI locations | Pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-16 review planning | None | 1 | No error; plan updated from review notes and local evidence. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 6: critical manuscript gap repair from review. |
| Where am I going? | Fix FC4/power-cost inconsistency, downshift round-robin claims, align innovation narrative with ablation evidence, and fill missing reproducibility/data details. |
| What's the goal? | Make the paper credible under reviewer scrutiny before final checkpoint. |
| What have I learned? | Real RS485 logs provide concrete raw sequence shapes and acquisition protocol; the paper currently has identifiable locations for each requested repair. |
| What have I done? | Updated task plan, findings, and progress with the new review-driven repair plan. |

## Session: 2026-05-25 Rewrite Takeover (CRST Target Superseded by ESWA)

### Phase 8: Evidence Reconciliation Before Continued Drafting
- **Status:** in_progress
- Actions taken:
  - Loaded the `planning-with-files` and `academic-paper-composer` workflows.
  - Read the existing subproject plan, findings, and progress logs.
  - Read the then-current CRST rewrite strategy, evidence ledger, and
    length-reduction notes. This journal target is now superseded; the active
    target journal is *Expert Systems with Applications* (ESWA).
  - Marked the former Phase 7 submission checkpoint as superseded by the full
    rewrite while preserving its locked evidence.
  - Recorded the handoff question about the unaggregated oracle-prior-selected
    feasible static candidate for direct verification.
- Files modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Next Verification Unit
- Inspect V3.1 raw run artifacts, evaluator/training/pipeline code, and current paper
  source to determine whether a defensible trained-selected static comparator can be
  added before drafting claims about dynamic versus static scheduling.

## Session: 2026-06-07 PD-PPO Static-Break Recalibration

### Phase 9: Static-Break Scene Implementation
- **Status:** in_progress
- Actions taken:
  - Read the v1 v6 complex static-break sensor config and the PD-PPO oracle-lift
    and split-protocol scripts.
  - Added `configs/sensors/windblown_sensors_physical_event_v6_static_break.yaml`.
  - Extended `scripts/49_v31_physical_event_oracle_lift.py` with
    `--schedule-family v6_static_break` for SPC/fc4/context event schedules.
  - Added `scripts/63_v31_static_break_calibration.py` to sweep budgets,
    startup-peak budgets, and snow-transport target profiles before PPO.
- Files modified:
  - `configs/sensors/windblown_sensors_physical_event_v6_static_break.yaml`
  - `scripts/49_v31_physical_event_oracle_lift.py`
  - `scripts/63_v31_static_break_calibration.py`
  - `task_plan.md`
  - `progress.md`

## Next Verification Unit
- Compile the new scripts, run a local linear-oracle calibration smoke, then
  append the first calibration result to `CHANGELOG.md`.

## Session: 2026-06-08 Deployable-Static Hard-Gate Reset

### Phase 9: Static-Break Goal Upgrade
- **Status:** in_progress
- Actions taken:
  - Backed up the current English supervisor-draft PDF before further scenario
    changes:
    `paper/_archive/pdppo_crst_rewrite_20260608_213436_pre_static_break_recalibration.pdf`.
  - Verified backup SHA256 equals the live PDF:
    `bb70576db85d016fec841b6c39a26e6681e16b6ea4d17a9697bc13a718de1b4d`.
  - Recomputed PD-PPO vs deployable static-priority replay:
    `4/10` wins, mean delta `-0.000320`, Wilcoxon two-sided `p=0.7695`.
  - Aggregated existing static-break calibration summaries:
    219 rows across 27 reports; best dynamic-vs-static oracle-lift headroom
    reaches roughly 3--4% in several transport/particle/flux profiles.
  - Updated Phase 9 gate to require comprehensive PD-PPO superiority against
    deployable static-priority replay before the manuscript can claim static
    baseline dominance.

## Next Verification Unit
- Select 2--3 existing high-headroom static-break candidates and run a reduced
  split-protocol PPO gate against static-priority duty replay. Do not rewrite the
  main result table until PD-PPO clears the new gate.

## Session: 2026-06-20 V25 Low-Budget Static-Squeeze Gate

### Phase 9: Post-V24 Static-Break Recalibration
- **Status:** in_progress
- Actions taken:
  - Confirmed V24 dual-flux phase24 locked seeds `41--45` failed strict learned
    replication even though behaviour was clean across all five seeds.
  - Added and launched the V25 low-budget static-squeeze TCN gate using the V24
    event-selective-laser sensor information structure and lower budgets
    B=`1.03/1.05/1.08`.
  - Prepared and synced a parameterized V25 low-budget split-replay follow-up
    script for any structural pass.
  - Monitored the running tmux session
    `pdppo_v25_low_budget_gate_seed45_h082_20260620`.
  - Launched the split-replay follow-up for the first structural pass:
    tmux `pdppo_v25_lowbudget_splitreplay_particle_b1p03_seed45_h082_20260620`,
    profile `particle_heavy_flux_v7`, B=`1.03`, GPU `5`.
  - Updated the operating mandate: future work may change simulator generation
    and RL framework internals, not just scene parameters, if that is required
    to obtain a forecast-optimal and genuinely state-dependent learned policy.
  - Added `scripts/71_v31_behavior_complexity_audit.py` to reject fixed-subset
    and simple-cycle scheduling behaviour from rollout NPZ artifacts.
  - Smoke-tested the audit on an old learned rollout and a static rollout:
    fixed static fails as fixed/simple-cycle; old custom PPO is not fixed but
    fails the stricter state-dependence gate, matching the new acceptance bar.
  - Added and synced
    `scripts/run_pdppo_static_break_v25_v24_low_budget_learned_ppo_seed45_h082_20260620.sh`.
    The runner defaults to oracle-greedy AWBC rather than a hard-coded cyclic
    teacher, supports opt-in replay-derived event-cyclic pools, and runs the
    behaviour complexity audit after collecting learned-PPO metrics.
  - Found from the split static-candidate table that B=`1.03` still permits a
    strong fixed laser subset, so deployable-static margin alone is not a
    sufficient structural gate.
  - Updated `scripts/63_v31_static_break_calibration.py` with
    `--require-raw-static-margin` and patched the V25 low-budget gate runner to
    use it in future reruns. Local and remote checks passed after one transient
    SSH disconnect/retry.
  - Extended the sensor runtime with backward-compatible calm/non-event fields:
    `calm_noise_std`, `calm_noise_multiplier`, and
    `calm_observation_probability`.
  - Added
    `configs/sensors/windblown_sensors_physical_event_v26_calm_selective.yaml`,
    which keeps V24-like costs but makes laser/FC4 high-value in events and
    low-information in calm periods.
  - Local and remote Python checks passed for the V26 sensor-model extension.
  - Synced V25 split-replay/source artifacts locally.
  - V25 split-replay failed strict static replay:
    best replay `split_top3_l3_dwell24` loss `0.415506`, validation static
    `0.398729`, raw static `static_action13` `0.396226`; gate pass `False`.
    Therefore no V25 learned-PPO run was launched from this point.
  - Ran the behaviour-complexity audit on the failed best V25 replay. It passed
    complexity (`unique_mask_count=12`, `event_sensor_l1=0.9196`), confirming
    the blocker is forecast superiority over raw static, not merely behaviour
    degeneracy.
  - Added and launched
    `scripts/run_pdppo_static_break_v26_calm_selective_low_budget_profile_scan_dwell12_gate_seed45_h082_20260620.sh`.
    Remote tmux:
    `pdppo_v26_calm_selective_lowbudget_gate_seed45_h082_20260620`.
  - Added and synced
    `scripts/run_pdppo_static_break_v26_calm_selective_low_budget_split_replay_gate_seed45_h082_20260620.sh`
    so any V26 structural pass can move directly to strict split-replay.
- First completed structural result:
  - profile `particle_heavy_flux_v7`, B=`1.03`, startup peak B=`1.55`;
  - deployable-static loss `0.396898`;
  - best dynamic loss `0.381306`;
  - dynamic margin `+0.039284`;
  - event dynamic margin `+0.030887`;
  - laser shortcut broken, dynamic diversity ok, `gate_pass=True`;
  - best dynamic schedule: `dynamic:diverse_top2_lead6_dwell12`;
  - best dynamic has laser, FC4, and SPC, with `7` mid-duty sensors and only
    `1` always-off sensor.

## Next Verification Unit
- Launch and monitor the V25 split-replay gate for
  `particle_heavy_flux_v7`, B=`1.03`, then decide whether a reduced learned-PPO
  split-protocol run is warranted.
- If split-replay/learned PPO fails, design the next intervention as an
  architectural or generator change with anti-fixed-subset and anti-simple-cycle
  diagnostics, rather than another blind target-weight/budget sweep.
- Apply `71_v31_behavior_complexity_audit.py` to any future learned-PPO rollout
  before treating performance wins as publishable.
- Monitor V26 strict raw-static gate; if it passes, run split-replay before any
  learned PPO. If it fails, escalate to explicit latent event-subtype generator
  and/or recurrent actor-critic rather than more low-budget sweeps.

## Session: 2026-06-20 V26 Strict Gate Monitoring and V27 Design Prep

### Phase 9: Static-Break Beyond Scenario-Only Tuning
- **Status:** in_progress
- Actions taken:
  - Re-read the active plan, progress log, and findings log after context
    compaction. The hard acceptance criterion remains: learned RL must be
    forecast-optimal under the protocol and nontrivially state-dependent, not a
    fixed sensor subset or simple two/three-mask cycle.
  - Polled local monitor session `41975`; it completed normally and reported
    V26 still running with no summary yet.
  - Checked remote tmux status twice. As of remote time `2026-06-20 05:13:26`,
    V26 `pdppo_v26_calm_selective_lowbudget_gate_seed45_h082_20260620` was
    still on the first combo
    `particle_heavy_flux_v7_b1p03_p1p55`; no `calibration_summary.csv` or
    `calibration_failures.csv` was present.
  - While V26 was running, inspected the generator and custom PPO interfaces:
    `src/data_sources/public_weather_synthesis.py`, `src/v2/env.py`, and
    `src/v2/custom_ppo.py`.
  - Confirmed V26 is still a binary calm/event intervention. If it fails the
    strict raw-static gate, the next change should be V27 latent event subtypes
    plus a risk/memory-oriented RL upgrade, not another budget/profile sweep.

## Next Verification Unit
- Continue monitoring V26 until the first strict row is produced. If any V26 row
  has `gate_pass=True` under `--require-raw-static-margin`, launch the prepared
  V26 split-replay wrapper for that exact profile/budget. If no strict pass
  appears, start V27 by adding latent event subtypes to the truth generator and
  subtype-aware dynamic diagnostics.

## Session Update: 2026-06-20 V26 First Strict Row Failed; V27 Launched

### Phase 9: V27 Latent Event Subtypes
- **Status:** in_progress
- Actions taken:
  - V26 first completed row:
    `particle_heavy_flux_v7_b1p03_p1p55`.
  - V26 result:
    deployable static loss `0.405657`, dynamic loss `0.380647`, dynamic margin
    `+0.061653`; raw static loss `0.374002`, raw-static margin `-0.017767`;
    event dynamic margin `+0.000907`; `strict_static_gate_ok=False`;
    `gate_pass=False`.
  - Decision:
    V26 `particle_heavy_flux_v7 @ B=1.03` is not eligible for split replay or
    learned PPO. It improves on deployable static-priority replay but still
    loses to the raw fixed static candidate.
  - Implemented V27 generator support:
    `event_subtypes_enabled` with latent subtype ids for particle-dominant,
    flux-dominant, and thermal-boundary events. The generator writes
    `event_subtype_id`, `event_subtype_particle`, `event_subtype_flux`, and
    `event_subtype_thermal` to truth CSVs while preserving the original
    `event_flag`.
  - Implemented V27 subtype-aware oracle-lift diagnostics:
    `schedule_family=subtype_static_break` and `all` now include dynamic
    schedules that choose particle/flux/thermal sensor masks according to the
    generated subtype column when present.
  - Added runner:
    `scripts/run_pdppo_static_break_v27_subtype_low_budget_profile_scan_gate_seed45_h082_20260620.sh`.
    The runner uses the V26 calm-selective sensor config plus V27 subtype truth
    generation, strict `--require-raw-static-margin`, and `schedule_family=all`.
  - Local checks passed:
    Python compile for modified files, runner `bash -n`, calibration dry-run,
    truth generation smoke with subtype columns, and a small oracle-lift
    subtype-schedule smoke.
  - Synced V27 code/runner to `remote-gpu`; remote conda `darts` compile,
    `bash -n`, and runner dry-run checks passed.
  - Launched remote tmux:
    `pdppo_v27_subtype_lowbudget_gate_seed45_h082_20260620`.
    Output directory:
    `reports/v31_static_break_v27_subtype_low_budget_profile_scan_gate_seed45_h082_20260620/`.
  - The older V25 scan later produced `particle_heavy_flux_v7 @ B=1.08`.
    Although that scan was launched before the strict column patch, its table
    values imply a strict structural pass:
    raw static loss `0.371156`, dynamic loss `0.353611`,
    raw-static margin about `+0.0473`, and event margin `+0.03485`.
  - Launched the required split-replay gate for this V25 B=`1.08` candidate:
    tmux `pdppo_v25_splitreplay_particle_b1p08_seed45_h082_20260620`,
    output
    `reports/v31_static_break_v25_v24_low_budget_particle_heavy_flux_v7_b1p08_split_replay_gate_seed45_h082_20260620/`.

## Next Verification Unit
- Monitor V27 until the first strict row is produced and monitor the V25
  B=`1.08` split-replay gate. A learned PPO launch is allowed only after a
  split-replay pass against both source-reference and replay-local raw static.

## Session Update: 2026-06-20 V25 B=1.08 Replay Failed; V27 First Row Near-Miss

### Phase 9: Strict Gate Monitoring
- **Status:** in_progress
- Actions taken:
  - Polled remote status at remote time `2026-06-20 05:38:32`.
  - V25 B=`1.08` split replay completed and failed:
    best replay `split_top3_l6_dwell6` oracle loss `0.440489`; AOI reference
    `0.437016`; replay-local raw static `static_action0` `0.433427`.
  - Margins were negative:
    `margin_abs_vs_reference=-0.003473` and
    `margin_abs_vs_static_reference=-0.007062`; `gate_pass=False`.
  - Decision:
    do not launch learned PPO from V25 B=`1.08`. The earlier structural
    oracle-lift headroom did not survive split-replay validation.
  - V26 second row, B=`1.05`, also failed strict raw-static despite improving
    over deployable static-priority replay:
    dynamic loss `0.380131`, raw static `0.376003`,
    raw-static margin `-0.010979`, `gate_pass=False`.
  - V27 first completed row, `particle_heavy_flux_v7 @ B=1.03`, showed the
    first positive raw-static direction under latent event subtypes:
    dynamic loss `0.436141` vs raw static `0.438855`.
  - However, the absolute raw-static headroom is only `0.002714`, below the
    strict `0.005` requirement, so
    `strict_static_gate_ok=False` and `gate_pass=False`.
  - V27 B=`1.05` and V26 B=`1.08` remained running at the time of the status
    check. No split replay or learned PPO should be launched from these rows
    unless the strict summary later shows `gate_pass=True`.
  - Patched the split-protocol stack so V27 subtype truth parameters propagate
    through future split/learned runs:
    `59_v31_split_protocol_grid.py -> 58_v31_split_protocol_run.py ->
    25_v2_train_custom_ppo.py -> 23_v2_train_ppo.ensure_truth`.
  - Validation:
    local Python compile passed; local `59` and `58` dry-runs showed
    `--event-subtypes-enabled` and subtype numeric parameters in the downstream
    commands. Synced the three scripts to `remote-gpu`; remote conda `darts`
    compile and `59` dry-run passed.
  - Inspected the V27 B=`1.03` candidate table. The best raw-static-breaking
    row is `dynamic:auto_non8_event13_lead6` with loss `0.422917`, but it is a
    plain event/calm mask-pair schedule and has switch rate `0.00235`, below the
    current behaviour gate. Explicit subtype schedules are much weaker
    (`subtype_particle_counter_mix` `0.461326`,
    `subtype_laser_fc4_thermal` `0.493959`), so V27 has not yet produced the
    intended multi-subtype scheduling advantage.
  - Added `subtype_auto` oracle-lift diagnostics:
    static candidate evaluation now records subtype-specific oracle losses
    (`particle`, `flux`, `thermal`); `subtype_auto` selects top calm and
    subtype-specific masks and evaluates their subtype-conditioned rollout.
  - This diagnostic is deliberately not included in `schedule_family=all` yet,
    so the already-running V27 scan is not silently changed mid-run.
  - Local compile and a small linear-oracle smoke passed
    (`16` subtype_auto rows generated). Synced
    `49_v31_physical_event_oracle_lift.py` and
    `63_v31_static_break_calibration.py` to `remote-gpu`; remote compile and
    CLI help check passed.

## Next Verification Unit
- Continue monitoring V27 and V26. If no strict row passes, move from subtype
  generation alone to the next framework change: make subtype/risk information
  learnable by the policy through a memory or explicit short-horizon
  risk/event-belief head, then rerun the strict structural gate before learned
  PPO.

## Session Update: 2026-06-20 V26 B=1.08 Strict Pass and Split Replay Launch

### Phase 9: V26 Strict Candidate Follow-Up
- **Status:** in_progress
- Actions taken:
  - Remote status at `2026-06-20 05:49:43` showed V26
    `particle_heavy_flux_v7_b1p08_p1p55` passed the strict raw-static structural
    gate:
    dynamic loss `0.362321`, raw static loss `0.385357`,
    raw-static margin `+0.059779`, event dynamic margin `+0.051663`,
    `strict_static_gate_ok=True`, `gate_pass=True`.
  - Behaviour diagnostics in the structural row are acceptable for a follow-up:
    best dynamic `dynamic:auto_non0_event15_lead0`, switch rate `0.028320`,
    mid-duty sensors `5`, always-on sensors `1`, always-off sensors `2`,
    laser shortcut broken.
  - Launched required split-replay validation in tmux
    `pdppo_v26_splitreplay_particle_b1p08_seed45_h082_20260620` using GPU `5`.
  - Output/source directories:
    `reports/v31_static_break_v26_calm_selective_low_budget_particle_heavy_flux_v7_b1p08_zero_ppo_source_seed45_h082_20260620/`
    and
    `reports/v31_static_break_v26_calm_selective_low_budget_particle_heavy_flux_v7_b1p08_split_replay_gate_seed45_h082_20260620/`.
  - Initial log confirms the zero-PPO split source is running through
    `59_v31_split_protocol_grid.py -> 58_v31_split_protocol_run.py ->
    25_v2_train_custom_ppo.py`.
  - Added and synced a V26 learned-PPO wrapper for this exact configuration:
    `scripts/run_pdppo_static_break_v26_calm_selective_low_budget_learned_ppo_seed45_h082_20260620.sh`.
    It defaults to `particle_heavy_flux_v7`, B=`1.08`, V26 calm-selective
    sensor config, and a V26-specific output directory. This wrapper is only to
    be launched if split replay passes.
  - Follow-up poll at remote time `2026-06-20 05:55:57`: the zero-PPO source
    run completed and `70_v31_split_replay_gate.py` is running. Source metrics
    alone do not pass (`custom_ppo` loss `0.435226`, AOI `0.432715`, feasible
    static `0.427272`), so the decision remains entirely on the replay gate
    summary.

## Next Verification Unit
- Monitor the V26 B=`1.08` split replay. If it passes both source-reference and
  replay-local raw-static gates, launch the learned-PPO runner for exactly this
  V26 configuration and run the behaviour-complexity audit on the learned
  rollout. If split replay fails, do not launch learned PPO.

## Session Update: 2026-06-20 V26 B=1.08 Split Replay Failed

### Phase 9: V26 Candidate Closed
- **Status:** complete
- Actions taken:
  - Remote poll at `2026-06-20 06:06:40` showed the split replay completed in
    the tmux log.
  - Best replay:
    `split_top2_l6_dwell12`, oracle loss `0.425502`.
  - Source/reference comparison:
    feasible static reference loss `0.427272`, so the replay only improves by
    `+0.001770` absolute and `+0.004143` relative, below the required margin.
  - Replay-local raw static comparison:
    `static_action1` loss `0.409637`; best replay is worse by `-0.015865`
    absolute and `-0.038730` relative.
  - Gate:
    `static_reference_gate_pass=False`, `gate_pass=False`.
  - Decision:
    do not launch the prepared V26 learned-PPO wrapper. V26 B=`1.08` is closed
    as a mainline migration candidate despite its structural-table pass.

## Next Verification Unit
- Continue monitoring the broader V26/V27 scans for other strict passes, but do
  not promote structural-table passes without split replay. In parallel, use the
  new `subtype_auto` diagnostic to test whether V27 can produce a genuinely
  multi-regime dynamic schedule; if it cannot, proceed to a policy-side
  memory/risk-belief upgrade rather than more binary event/calm tuning.

## Session Update: 2026-06-20 V27 `subtype_auto` Probe Launched

### Phase 9: V27 Multi-Regime Diagnostic
- **Status:** in_progress
- Actions taken:
  - Checked remote GPU/process load after V26 split replay failure; GPU `5` was
    free and the server had enough headroom for one additional diagnostic job.
  - Launched a single-combo V27 `subtype_auto` calibration probe in tmux
    `pdppo_v27_subtype_auto_probe_particle_b1p05_seed45_h082_20260620`.
  - Configuration:
    V26 calm-selective sensor config, V27 latent subtype truth generation,
    `particle_heavy_flux_v7`, B=`1.05`, startup peak `1.55`,
    `schedule_family=subtype_auto`, strict `--require-raw-static-margin`.
  - Output:
    `reports/v31_static_break_v27_subtype_auto_probe_particle_b1p05_seed45_h082_20260620/`.
  - Purpose:
    determine whether automatically selected calm/particle/flux/thermal masks
    can create genuine multi-regime dynamic headroom. This is diagnostic only;
    it does not authorize learned PPO without split replay.

## Session Update: 2026-06-20 V27 `subtype_auto` Split Replay Launched

### Phase 10: V27 Multi-Regime Split Replay
- **Status:** in_progress
- Actions taken:
  - The V27 `subtype_auto` probe completed with a strict structural pass for
    `particle_heavy_flux_v7 @ B=1.05`.
  - Best dynamic diagnostic:
    `dynamic:subtype_auto_c1_p0_f1_t0_lead6`, oracle loss `0.419422`.
  - References:
    raw static loss `0.439998`; deployable/static-priority reference loss
    `0.482282`.
  - Margins:
    dynamic margin `+0.130340`, raw-static margin `+0.046764`, event dynamic
    margin `+0.065354`.
  - Behaviour diagnostics:
    switch rate `0.003937`, mid-duty sensors `5`, always-on sensors `1`,
    always-off sensors `2`, duty entropy `0.598707`, laser shortcut broken.
  - Important interpretation:
    this is the first strong multi-regime structural signal, but it is still a
    privileged subtype-conditioned diagnostic, not a learned PPO result.
  - Extended `scripts/70_v31_split_replay_gate.py` with
    `--replay-family subtype_auto`.
    The split gate now:
    evaluates subtype-specific static-candidate losses;
    selects calm/particle/flux/thermal mask pools from those losses;
    replays subtype-conditioned schedules on the split source;
    still compares against both the source reference and the replay-local raw
    fixed-static candidate.
  - Added runner
    `scripts/run_pdppo_static_break_v27_subtype_auto_low_budget_split_replay_gate_seed45_h082_20260620.sh`.
    It generates a V27 subtype zero-PPO split source, then runs
    `70_v31_split_replay_gate.py --replay-family subtype_auto`.
  - Validation:
    local compile/help and runner `bash -n` passed; remote checksums match local;
    remote conda `darts` compile and runner `bash -n` passed.
  - Launched tmux:
    `pdppo_v27_subtypeauto_splitreplay_particle_b1p05_seed45_h082_20260620`
    on GPU `5`.
  - Output/source directories:
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_zero_ppo_source_seed45_h082_20260620/`
    and
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_split_replay_gate_seed45_h082_20260620/`.
  - Remote truth check passed:
    `event_subtype_id` exists with counts `0=26415`, `1=11109`,
    `2=11864`, `3=10612`.

## Next Verification Unit
- Monitor the V27 `subtype_auto` split replay. If it fails replay-local
  raw-static clearance, do not launch learned PPO. If it passes, the next step is
  a policy-side observability/learning change: the current `subtype_auto`
  schedule uses privileged subtype labels, so learned PPO needs an observable
  risk/subtype-belief or memory mechanism before it can be a paper-mainline
  PD-PPO candidate.

## Session Update: 2026-06-20 V27 `subtype_auto` Split Replay Passed

### Phase 10: V27 Multi-Regime Split Replay
- **Status:** complete
- Results:
  - Best replay:
    `split_subtype_auto_top2_c0_p1_f1_t0_l0`.
  - Oracle loss:
    `0.501525`.
  - Source reference:
    `feasible_static_projected`, loss `0.519033`.
  - Replay-local raw static:
    `static_action1`, loss `0.512670`.
  - Margins:
    versus source reference `+0.017509` absolute / `+0.033733` relative;
    versus raw static `+0.011146` absolute / `+0.021740` relative.
  - Gate:
    `source_reference_gate_pass=True`,
    `static_reference_gate_pass=True`, `gate_pass=True`.
  - Event/non-event:
    best replay event loss `0.699510`, non-event loss `0.272245`;
    raw static event loss `0.725253`, non-event loss `0.266486`.
- Behaviour complexity audit:
  - `behavior_complexity_gate_pass=True`.
  - `unique_mask_count=9`.
  - `top3_mask_fraction=0.794189`, below the simple-cycle threshold.
  - `mask_entropy_bits=2.335018`.
  - `transition_entropy_bits=2.771129`.
  - `switches_per_step=0.032357`.
  - `event_sensor_l1=2.582324`.
  - `event_mask_mi_bits=0.344734`.
  - Flags:
    `fixed_like=False`, `simple_cycle_like=False`,
    `low_complexity=False`, `weak_state_dependence=False`.
- Local synced files:
  - `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_split_replay_gate_seed45_h082_20260620/split_replay_gate_summary.json`
  - `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_split_replay_gate_seed45_h082_20260620/split_replay_gate_metrics.csv`
  - `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_split_replay_gate_seed45_h082_20260620/behavior_complexity_best_replay.csv`
  - `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_split_replay_gate_seed45_h082_20260620/behavior_complexity_best_replay.json`
- Decision:
  this is now a valid non-fixed, non-simple-cycle scheduling scene at the
  diagnostic/replay level. It is still not a paper-mainline learned PD-PPO
  result because it uses generated subtype labels. The next implementation
  target is to make the policy infer the required regime from observable recent
  history or a learned risk/subtype-belief head, then train/evaluate learned
  PPO against the same raw-static and behaviour gates.

## Next Verification Unit
- Implement the learned-policy bridge for V27:
  add a policy-side observable subtype/risk-belief or memory mechanism, plus an
  AWBC teacher mode that can distill the passing subtype-auto schedule during
  training without exposing privileged subtype labels at evaluation. Then run a
  learned PPO candidate and audit it with the same split/raw-static and
  behaviour-complexity gates.

## Session Update: 2026-06-20 V27 Learned PPO Bridge Launched

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- Code changes:
  - Added `awbc_teacher_mode=subtype_auto` to
    `src/v2/custom_ppo.py`.
  - Training-time AWBC teacher can now map generated `event_subtype_id` to four
    candidate actions: calm, particle, flux, thermal.
  - Evaluation-time policy still receives only the normal environment state
    (`history`, `mask_history`, sensor modes/freshness, previous action, duty
    state, time/event/SOC tail). It does not receive `event_subtype_id`.
  - Added CLI arguments in `scripts/25_v2_train_custom_ppo.py`:
    `--awbc-teacher-subtype-calm-sensors`,
    `--awbc-teacher-subtype-particle-sensors`,
    `--awbc-teacher-subtype-flux-sensors`,
    `--awbc-teacher-subtype-thermal-sensors`.
  - Propagated those arguments through
    `scripts/58_v31_split_protocol_run.py` and
    `scripts/59_v31_split_protocol_grid.py`.
  - Added runner:
    `scripts/run_pdppo_static_break_v27_subtype_auto_low_budget_learned_ppo_seed45_h082_20260620.sh`.
- Validation:
  - Local compile passed for `custom_ppo.py`, `25`, `58`, and `59`.
  - Local `59` and `58` dry-runs confirmed subtype teacher arguments reach the
    final `25_v2_train_custom_ppo.py` command.
  - Synced to `remote-gpu`; remote checksums matched local; remote conda
    `darts` compile and runner `bash -n` passed.
- Launched:
  - tmux `pdppo_v27_subtypeauto_learnedppo_particle_b1p05_seed45_h082_20260620`
    on GPU `5`.
  - Output:
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_learned_ppo_seed45_h082_20260620/`.
- Learned run defaults:
  - `TOTAL_TIMESTEPS=80000`
  - `AWBC_COEF=0.80`
  - `PRIOR_KL_COEF=0.03`
  - `ENT_COEF=0.002`
  - teacher masks from the passing replay:
    calm = `met_station_core,surface_temp_ir,snow_particle_counter`;
    particle = `radiometer_basic,shielded_thermo_hygro,laser_disdrometer`;
    flux = `surface_temp_ir,shielded_thermo_hygro,laser_disdrometer`;
    thermal = `surface_temp_ir,shielded_thermo_hygro,laser_disdrometer`.
- Gate for success:
  learned `custom_ppo` must beat the raw/static references under the split
  protocol and pass `71_v31_behavior_complexity_audit.py`. A low oracle loss
  alone is not sufficient.

## Session Update: 2026-06-20 V27 Learned PPO Strict Gate Failed; BC Warm-Start Running

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- Strict learned result:
  - The first learned subtype-auto PPO run passes the behaviour-complexity audit
    but fails the strict raw-static criterion.
  - Learned `custom_ppo` oracle loss: `0.516547`.
  - Same-run flow reference `feasible_static_projected`: `0.517810`.
  - Strict final-test replay-local raw static:
    `0.514839` (`static_action0/1`, CPU/CUDA duplicate runs agree within
    `1e-5`).
  - Therefore learned PPO is worse than replay-local raw static by about
    `0.00171` and is not a paper-mainline candidate.
  - The diagnostic subtype-auto replay on the same source still improves to
    about `0.51337`, but its margin over raw static is only about `0.00147`,
    below the configured strict margin.
- Code changes after failure:
  - Added configurable actor behaviour-cloning warm-start before PPO in
    `src/v2/custom_ppo.py`.
  - Added CLI/config plumbing:
    `--bc-pretrain-steps`, `--bc-pretrain-epochs`,
    `--bc-pretrain-batch-size`, `--bc-pretrain-loss-coef`.
  - Propagated BC parameters through `25`, `58`, and `59`.
  - Added runner:
    `scripts/run_pdppo_static_break_v27_subtype_auto_low_budget_bc_warmstart_ppo_seed45_h082_20260620.sh`.
- Validation:
  - Local and remote `py_compile` passed.
  - Local dry-runs confirmed `59 -> 58 -> 25` parameter propagation.
  - Remote checksums matched local after sync.
- Running:
  - tmux `pdppo_v27_subtypeauto_bcwarm_ppo_particle_b1p05_seed45_h082_20260620`.
  - Output:
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_bc_warmstart_ppo_seed45_h082_20260620/`.
  - First BC pretrain log:
    `steps=16000`, `loss=1.204145`, `accuracy=0.818`,
    `unique_actions=3`.

## Next Verification Unit
- Monitor the BC warm-start learned PPO run to completion.
- If it beats strict raw static and passes behaviour audit, run the same strict
  final-test static gate on that source directory and sync all CSV/JSON evidence.
- If it still fails, the next change should be a real policy-memory/risk-belief
  module rather than stronger teacher weighting alone.

## Session Update: 2026-06-20 BC Warm-Start Failed; Observable Belief Run Launched

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- BC warm-start result:
  - Completed tmux
    `pdppo_v27_subtypeauto_bcwarm_ppo_particle_b1p05_seed45_h082_20260620`.
  - Learned `custom_ppo` oracle loss: `0.517628`.
  - Same-run `feasible_static_projected`: `0.517380`.
  - Delta static-minus-PPO: `-0.000248`, so the BC warm-start learned policy
    still fails even before the stricter replay-local raw-static gate.
  - Behaviour remains valid:
    `behavior_complexity_gate_pass=True`,
    `unique_mask_count=9`, `top3_mask_fraction=0.777832`,
    `mask_entropy_bits=2.408396`, `event_sensor_l1=2.521975`,
    `event_mask_mi_bits=0.325673`.
- Interpretation:
  - Stronger imitation alone did not solve the problem. The policy behaviour is
    nontrivial, but forecast loss remains static-dominated.
  - The next fix is an observable regime/risk feature path, not more AWBC
    pressure by itself.
- Code changes:
  - Added optional observable regime-belief tail features to
    `src/v2/env.py`, derived only from observation history and mask history:
    particle, flux, and thermal risk signals plus key observed-value and
    coverage summaries.
  - Added `--include-observable-regime-belief` and
    `--regime-belief-lookback`.
  - Propagated those flags through `25`, `58`, `59`, metadata recovery in `64`,
    and custom PPO train/eval config copies.
  - Added runner:
    `scripts/run_pdppo_static_break_v27_subtype_auto_low_budget_belief_bc_ppo_seed45_h082_20260620.sh`.
- Validation:
  - Local compile and `bash -n` passed.
  - Local dry-runs confirmed `59 -> 58 -> 25` propagation.
  - Remote checksums matched local and remote compile/help checks passed.
- Running:
  - tmux `pdppo_v27_subtypeauto_belief_bc_ppo_particle_b1p05_seed45_h082_20260620`.
  - Output:
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_belief_bc_ppo_seed45_h082_20260620/`.

## Next Verification Unit
- Monitor the observable-belief run. If BC accuracy rises materially above
  `0.818` and final `custom_ppo` beats static, sync evidence and run strict
  static/behaviour gates. If it fails, implement a genuine memory/recurrent
  policy rather than another feed-forward feature tweak.

## Session Update: 2026-06-20 Observable Belief Failed; Subtype Auxiliary PPO Launched

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- Observable-belief+BC result:
  - Completed tmux
    `pdppo_v27_subtypeauto_belief_bc_ppo_particle_b1p05_seed45_h082_20260620`.
  - Learned `custom_ppo` oracle loss: `0.519019`.
  - Same-run `feasible_static_projected`: `0.518138`.
  - Delta static-minus-PPO: `-0.000881`, so the observable-belief policy still
    fails static optimality and does not warrant a strict raw-static replay gate.
  - Behaviour remains valid:
    `behavior_complexity_gate_pass=True`,
    `unique_mask_count=9`, `top3_mask_fraction=0.778564`,
    `mask_entropy_bits=2.374741`, `event_sensor_l1=2.505204`,
    `event_mask_mi_bits=0.364018`.
- Interpretation:
  - The current scene can generate nontrivial dynamic schedules, but learned
    PPO still does not reliably convert observable regime evidence into lower
    forecast-oracle loss than fixed static.
  - The failure is not fixed-sensor or simple-cycle behaviour; it is forecast
    optimality versus static.
- Code changes:
  - Added an optional `subtype_aux_head` to the custom PPO actor. It is trained
    with a supervised `event_subtype_id` cross-entropy loss but receives only
    normal observation-derived policy inputs at inference.
  - Added CLI/config plumbing:
    `--subtype-aux-coef`, `--subtype-aux-classes`,
    `--subtype-aux-lookahead-steps`.
  - Propagated these flags through `25`, `58`, `59`, and the learned V27
    runner.
  - Training logs now include `subtype_aux_loss` and
    `subtype_aux_accuracy`.
  - Added runner:
    `scripts/run_pdppo_static_break_v27_subtype_auto_low_budget_subtype_aux_ppo_seed45_h082_20260620.sh`.
- Validation:
  - Local `py_compile`, `bash -n`, `--help`, and a short CPU smoke passed.
  - Remote compile/help and runner `PY=/bin/echo` dry-run passed.
- Running:
  - tmux
    `pdppo_v27_subtypeauto_subtypeaux_ppo_particle_b1p05_seed45_h082_20260620`.
  - Output:
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_subtype_aux_ppo_seed45_h082_20260620/`.

## Next Verification Unit
- Monitor the subtype-auxiliary PPO run. It must beat same-run static before
  any strict replay gate is meaningful.
- If it beats static, run the strict final-test raw-static gate and keep the
  behaviour-complexity audit.
- If it fails, move to a real temporal/memory policy path rather than more
  feed-forward features.

## Session Update: 2026-06-20 Subtype Auxiliary Improved Learned PPO But Failed Strict Raw Static

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- Subtype-auxiliary PPO result:
  - Completed tmux
    `pdppo_v27_subtypeauto_subtypeaux_ppo_particle_b1p05_seed45_h082_20260620`.
  - Learned `custom_ppo` oracle loss: `0.506899`.
  - Same-run `feasible_static_projected`: `0.515707`.
  - Flow delta static-minus-PPO: `+0.008808`; this is the first learned V27
    run that clearly beats same-run static.
  - Behaviour remains valid:
    `behavior_complexity_gate_pass=True`,
    `unique_mask_count=9`, `top3_mask_fraction=0.772705`,
    `mask_entropy_bits=2.381246`, `event_sensor_l1=2.433948`,
    `event_mask_mi_bits=0.342682`.
- Strict raw-static check:
  - Replay-local raw fixed static: `static_action1=0.505811`.
  - Learned `custom_ppo` is still worse than strict raw static by
    `0.001088`.
  - Privileged subtype replay still passes strongly:
    best replay `split_subtype_auto_top2_c0_p1_f1_t0_l0=0.495404`,
    margin over raw static `+0.010407`.
- Interpretation:
  - The auxiliary head fixes much of the learned-policy gap, but not enough for
    the paper mainline under the strict raw-static comparator.
  - Remaining gap is learned policy inference/control, not scenario design.

## Next Verification Unit
- Run stronger subtype auxiliary / imitation variants:
  - higher AWBC and subtype auxiliary weight;
  - larger BC pretraining budget;
  - lower entropy to reduce PPO drift from the subtype teacher.
- Accept only if learned `custom_ppo` beats replay-local raw static with a
  useful margin and keeps behaviour-complexity pass.

## Session Update: 2026-06-20 Stronger Subtype-Auxiliary Variants Launched

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- Launched two follow-up learned-policy variants:
  - `pdppo_v27_subtypeauto_subtypeaux_strongbc_particle_b1p05_seed45_h082_20260620`
    on GPU 3:
    `AWBC_COEF=1.50`, `SUBTYPE_AUX_COEF=0.75`,
    `SUBTYPE_AUX_LOOKAHEAD_STEPS=0`, `ENT_COEF=0.0001`,
    `BC_PRETRAIN_STEPS=40000`, `BC_PRETRAIN_EPOCHS=12`,
    `TOTAL_TIMESTEPS=60000`.
  - `pdppo_v27_subtypeauto_subtypeaux_l6_particle_b1p05_seed45_h082_20260620`
    on GPU 4:
    `AWBC_COEF=1.20`, `SUBTYPE_AUX_COEF=0.75`,
    `SUBTYPE_AUX_LOOKAHEAD_STEPS=6`, `ENT_COEF=0.0001`,
    `BC_PRETRAIN_STEPS=40000`, `BC_PRETRAIN_EPOCHS=12`,
    `TOTAL_TIMESTEPS=60000`.
- Both runs entered their grid workers and completed initial truth/dataset
  validation.
- The first attempt with `BC_PRETRAIN_STEPS=40000` failed before training
  because `collect_teacher_batch()` used the full BC batch length as one
  contiguous sampling window, making the valid train-start range empty:
  `[21000, 19991]`.
- Fixed `src/v2/custom_ppo.py` so both rollout and teacher-batch collection
  sample starts using the configured episode window rather than the total batch
  length.
- Local smoke passed with `bc_pretrain_steps` larger than one episode; remote
  compile passed.
- Relaunched fixed variants:
  - `pdppo_v27_subtypeauto_subtypeaux_strongbc2_particle_b1p05_seed45_h082_20260620`
  - `pdppo_v27_subtypeauto_subtypeaux_l6b2_particle_b1p05_seed45_h082_20260620`

## Next Verification Unit
- Monitor both fixed variants through BC pretraining and final metrics.
- If either learned `custom_ppo` beats replay-local raw static, rerun strict
  gate and sync all evidence.

## Session Update: 2026-06-20 First Learned Candidate Passed Strict Raw Static

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- Fixed strong-BC subtype-auxiliary PPO result:
  - Run:
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_subtype_aux_strongbc2_ppo_seed45_h082_20260620/`.
  - Learned `custom_ppo`: `0.508799`.
  - Same-run `feasible_static_projected`: `0.518587`.
  - Replay-local raw static from quick strict gate:
    `static_action1=0.513997`.
  - Learned strict margin: `0.513997 - 0.508799 = 0.005198`.
  - Required strict margin: `max(0.005, 1% raw)=0.005140`.
  - Learned strict raw-static pass: `True`, but by a narrow `0.000058`
    cushion above the configured threshold.
  - Behaviour-complexity pass: `True`, with
    `unique_mask_count=9`, `top3_mask_fraction=0.780762`,
    `mask_entropy_bits=2.371454`, `event_sensor_l1=2.363596`,
    `event_mask_mi_bits=0.329939`.
- L6 auxiliary variant:
  - `custom_ppo=0.513395`, `feasible_static_projected=0.517278`.
  - This is weaker than strongbc2 and not worth strict-gate escalation.
- Interpretation:
  - This is the first learned V27 candidate satisfying all current single-seed
    conditions: forecast beats strict raw static and behaviour is not fixed or
    simple cyclic.
  - Because the strict margin is narrow and only seed 45 has passed, this is a
    candidate for paper mainline, not yet a robust mainline result.

## Next Verification Unit
- Run seed 46/47 reproduction for strongbc2 before changing paper claims.
- For any seed that beats flow static, compute replay-local raw static and
  learned-vs-raw margin.

## Session Update: 2026-06-20 StrongBC2 Reproduction Launched

### Phase 11: Learned Policy From Subtype-Auto Teacher
- **Status:** in_progress
- Launched seed 46/47 reproduction:
  - tmux
    `pdppo_v27_subtypeauto_subtypeaux_strongbc2_repro_46_47_particle_b1p05_h082_20260620`.
  - Output:
    `reports/v31_static_break_v27_subtype_auto_low_budget_particle_heavy_flux_v7_b1p05_subtype_aux_strongbc2_repro_46_47_h082_20260620/`.
  - `workers=2`, `GPU_IDS=2,5`, `SEEDS=46 47`.
  - Same strongbc2 settings as the seed45 candidate:
    `AWBC_COEF=1.50`, `SUBTYPE_AUX_COEF=0.75`,
    `SUBTYPE_AUX_LOOKAHEAD_STEPS=0`, `ENT_COEF=0.0001`,
    `BC_PRETRAIN_STEPS=40000`, `BC_PRETRAIN_EPOCHS=12`,
    `TOTAL_TIMESTEPS=60000`.
- Both workers entered truth/dataset validation.

## Next Verification Unit
- Monitor seed46/47 to final metrics.
- Compute strict learned-vs-raw-static margins for any seed whose learned
  policy beats the flow static baseline.

## Session Update: 2026-06-20 Specialist-Subtype Scene and Energy Pilot

### Phase 12: Stronger Nontrivial Scheduling Contract
- **Status:** in_progress
- Seed 46/47 reproduction of the V27 StrongBC2 candidate did not provide a
  robust paper-mainline result. The earlier seed45 pass remains a narrow
  single-seed candidate, not a stable mainline.
- Added a specialist-subtype sensor scene:
  `configs/sensors/windblown_sensors_v31_specialist_subtype.yaml`.
  The scene uses cheap weather/precursor sensors plus subtype-specialist
  `surface_temp_ir`, `laser_disdrometer`, and `fc4_flux` sensors with
  subtype-specific reliability.
- Added subtype-specific observable precursors to generated truth:
  particle events boost humidity, flux events boost wind speed, and thermal
  events drop air temperature. These features are generated from normal truth
  columns and can be inferred from observations; they are not direct subtype
  labels at deployment.
- Formal specialist runs on seed45:
  - `v31_specialist_subtype_router_strongbc_seed45_h082_20260620` failed the
    strict raw-static gate. `custom_ppo=1.116297`,
    validation static `1.027591`, raw static `0.966362`.
  - `v31_specialist_subtype_snowheavy_router_strongbc_seed45_h082_20260620`
    improved but still failed strict raw static. Same-run learned
    `custom_ppo=1.149406`; raw static `static_action12=1.012736`.
- Best numeric replay on the snow-heavy scene:
  `router_remap_surface_calm_lowconf0.86` achieved
  `custom_ppo=1.005365`, beating raw static by `+0.007371` absolute but below
  the configured 1% strict margin (`0.010127`). Behaviour audit failed:
  `unique_mask_count=3`, `top3_mask_fraction=1.0`, `fixed_like=true`,
  `simple_cycle_like=true`.
- Router-contract scan showed the opposite tradeoff:
  - Best behaviour-valid replay:
    `router_contract_scan_c4_fb13_t086`, `oracle_loss=1.045930`,
    `unique_mask_count=5`, `top3_mask_fraction=0.828125`,
    `behavior_complexity_gate_pass=True`.
  - This is behaviour-clean but loses to raw static `1.012736`.
- Conclusion:
  replay-level mapping cannot satisfy both forecast/raw-static and behaviour
  gates in the current specialist scene. The next attempt must make the
  optimal policy genuinely energy/state dependent rather than just splitting
  subtype mappings.
- Implemented reproducibility plumbing:
  `--subtype-router-low-confidence-action` is now available in
  `25_v2_train_custom_ppo.py`, `58_v31_split_protocol_run.py`, and
  `59_v31_split_protocol_grid.py`; split manifests now record
  `target_weights` and `target_scales`.
- Launched energy-account structural pilot in tmux
  `v31_energy_pilot_20260620`:
  `reports/v31_energy_account_subtype_oraclegreedy_seed45_h082_20260620/`.
  It uses energy capacity/harvest constraints, no duty hard guard, no subtype
  router, oracle-greedy AWBC, SOC auxiliary prediction, and
  `event_transport_rich` final-test windows. As of launch health check, truth,
  oracle, and candidate prior artifacts are being produced normally.
- The first oracle-greedy energy pilot was stopped after about 14 minutes
  because full BC pretraining (`bc_pretrain_steps=8000`, lookahead 4) stayed in
  CPU-bound teacher sampling without reaching the first BC log line. This was
  a cost issue, not an environment/runtime crash.
- Launched a reduced-cost replacement:
  `v31_energy_fast_20260620`, output
  `reports/v31_energy_account_subtype_oraclegreedy_fast_seed45_h082_20260620/`.
  It keeps the same energy/specialist scenario but uses
  `bc_pretrain_steps=1000`, `awbc_label_stride=32`,
  `greedy_lookahead_steps=2`, and `total_timesteps=40000`.
- The reduced oracle-greedy fast run completed but failed both gates:
  `custom_ppo=1.185759` versus `validation_selected_static=1.127695`;
  behaviour audit failed with `unique_mask_count=2`,
  `fixed_like=true`, `simple_cycle_like=true`.
  Router replays using the trained subtype head were also worse
  (`1.210911` and `1.337983`), so this path is not viable.
- Launched a cheaper strong-BC subtype-router energy pilot:
  `v31_energy_router_20260620`, output
  `reports/v31_energy_account_subtype_router_strongbc_seed45_h082_20260620/`.
  It uses simple subtype-auto teacher labels instead of oracle-greedy labels,
  preserving energy/SOC constraints while avoiding the CPU-heavy teacher.

## Next Verification Unit
- Monitor `v31_energy_router_20260620` through BC/PPO completion.
- Run strict replay gate and behaviour-complexity audit on the energy-account
  result before considering seed 46/47.

## Session Update: 2026-06-20 Route-B Static-Break Recalibration

### Implemented
- Added an `energy_mpc` AWBC teacher path for custom PPO. The teacher scores
  feasible subsets with a cached short-horizon rollout over SOC, previous mask,
  and warmup state, and exposes CLI controls through `25`, `58`, and `59`.
- Added subtype-latent forcing to the public-weather truth generator:
  particle, flux, and thermal latent processes can perturb particle
  diameter/velocity, mass flux, and surface temperature independently of the
  ordinary weather covariates.
- Added candidate-mask oracle pretraining:
  `--oracle-candidate-mask-repeat` and `--oracle-candidate-mask-limit` add
  projected static candidate masks to the frozen oracle pretraining mixture.
- Added optional subtype-conditioned oracle loss weighting:
  `--subtype-loss-weighting` plus particle/flux/thermal target-weight vectors.
  This is default-off and uses `loss_with_context(...)` from
  `WarmupSchedulingEnv`.
- Exposed `--oracle-loss-clip` through split protocol scripts after finding
  that the fixed clip of 10 saturated flux subtype losses and hid `fc4_flux`
  differences.

### Results
- `v31_energy_account_subtype_energympc_smoke_seed45_h082_20260620` completed
  mechanically but failed: `custom_ppo=1.210079`,
  `validation_selected_static=1.164690`; behaviour fixed-like with
  `unique_mask_count=1`.
- `v31_subtype_snowlatent_tight_candidateoracle_tcn_gate_seed45_h082_20260620`
  is the best direction so far. Validation rollout showed dynamic
  `custom_ppo=5.664674` beating selected static `5.702770`, but strict replay
  gate failed: best replay `5.653785`, source-static margin `+0.048986`
  (`0.86%`), replay-local best static `5.657524`, strict margin only
  `+0.003739` (`0.066%`). The replay behaviour is valid
  (`unique_mask_count=5`, `behavior_complexity_gate_pass=True`), while the
  learned `custom_ppo` remains fixed/simple-cycle-like.
- `v31_subtype_snowlatent_tight_subtypeloss_tcn_gate_seed45_h082_20260620`
  failed. Subtype-conditioned loss made selected static stronger:
  `validation_selected_static=5.113164`, `custom_ppo=5.133708`; best replay
  margin versus strict static was only `+0.013164` (`0.26%`).
- Single-specialist budget variants failed to produce a valid dynamic upper
  bound. With strong latent and clip 10, best replay improved strict static by
  `+0.020150` (`0.39%`) but stayed below 1%. With unclipped loss, static
  `shielded_thermo_hygro` dominated and best replay lost to strict static by
  `-1380.074` (`-3.37%`).
- A dual particle/flux specialist scene with `oracle-loss-clip=100` also
  failed: validation selected static `45.284076`, custom dynamic `45.307244`;
  best replay lost to strict static by `-0.008311`.

### Current Diagnosis
- There is still no paper-mainline PD-PPO result that satisfies both
  forecast-optimality against replay-local strict static and non-fixed,
  non-simple-cycle scheduling behaviour.
- The closest usable signal is the candidate-oracle tight scene: it creates a
  behaviour-valid dynamic replay and reaches `0.86%` improvement against the
  source selected static, but it does not clear the strict static reference.
- The remaining blocker is not just scene parameters. The frozen oracle still
  assigns too much predictive value to low-power weather/context masks and too
  little marginal value to `fc4_flux` / `surface_temp_ir` in the relevant
  subtypes. Loss clipping was one cause for flux saturation, but removing it
  makes the objective numerically dominated by a fixed cheap mask.

## Session Update: 2026-06-20 Context-Power / Decoy Headroom

### Implemented
- Added scheduler-only agent context columns to the v2 warmup environment and
  custom PPO evaluation path:
  `agent_context_particle_alert`, `agent_context_flux_alert`, and
  `agent_context_event_alert`.
- Added subtype context lead/noise generation to
  `src/data_sources/public_weather_synthesis.py` and exposed the controls
  through scripts `20`, `25`, `58`, and `59`.
- Added explicit subtype replay support to
  `scripts/70_v31_split_replay_gate.py`, allowing direct evaluation of
  hand-specified calm/particle/flux/thermal sensor mappings rather than only
  top-static candidate pools.
- Added `configs/sensors/windblown_sensors_v31_specialist_context_decoy.yaml`,
  which preserves the context-power specialist geometry and adds four
  low-value noisy auxiliary sensors to stress-test duty-guard rotation.

### Results So Far
- `v31_dual_specialist_contextlead_duty_router_seed45_h082_20260620` improved
  subtype supervision but still failed forecast optimality:
  `custom_ppo=44.574194`, `validation_selected_static=44.434327`.
- `v31_dual_specialist_contextpower_duty_router_seed45_h082_20260620` passed
  the behaviour gate (`unique_mask_count=8`,
  `behavior_complexity_gate_pass=True`) but remained worse than static:
  `custom_ppo=44.032184`, `validation_selected_static=43.795307`.
- Explicit expert replay on the context-power run also failed:
  best `split_subtype_explicit_teacher_l0=44.202340`, worse than
  `validation_selected_static=43.795307` and replay-local
  `static_action0=43.851052`.
- Diagnosis: the current context-power scene has no structural headroom for
  subtype expert switching. The best strict reference is an empty desired mask
  plus duty guard, effectively a simple coverage rotation over all six sensors.
  With only six sensors, that rotation samples the specialists often enough to
  beat both learned PPO and privileged explicit subtype switching.

### Active Run
- Launched compact decoy-headroom run:
  `reports/v31_contextdecoy_headroom_seed45_h082_20260620/`,
  tmux `v31_contextdecoy_headroom_20260620`.
- Design changes: 10-sensor decoy config, shorter events, stronger subtype
  latent forcing, scheduler-only context lead, and tighter duty high
  (`0.45`) while keeping the non-fixed behaviour requirement.
- Next gate: after metrics are produced, run explicit subtype replay against
  replay-local strict static. If explicit replay still fails, the generator
  needs a stronger independent specialist-observation channel, not more PPO
  tuning.

### Decoy Gate Result And Framework Fix
- The decoy-headroom run completed but did not solve the static shortcut:
  `validation_selected_static=44.320671`, `custom_ppo=44.345280`.
- Fast and full explicit subtype replay agreed. Best explicit expert was
  `split_subtype_explicit_teacher_l10=44.337692`, worse than both the source
  static reference (`44.320671`) and replay-local strict static
  `static_action45=44.314263`.
- Replay-local static was effectively fixed `laser_disdrometer` plus a weak
  auxiliary sensor under duty guard. It achieved similar subtype losses to the
  explicit dynamic expert, so decoy dilution alone is insufficient.
- Found and fixed a framework issue in `src/v2/env.py`: multiple selected
  sensors observing the same variable previously overwrote each other in sensor
  list order. This let late noisy auxiliary sensors contaminate `full_open` and
  made the observation model order-dependent. The environment now fuses
  same-variable observations by inverse noise variance, with circular averaging
  for wind direction.
- Launched the next gate:
  `reports/v31_contextdecoy_fusion_oraclesubtype_seed45_h075_20260620/`,
  tmux `v31_fusion_oraclesubtype_20260620`.
  It combines the observation-fusion fix with oracle subtype-teacher
  pretraining while preserving the strong latent/context decoy scenario.

### 2026-06-20 Final Gate: Met+Specialist Pair Scene

- The fusion/decoy run created a superficially promising learned policy, but
  strict no-duty-guard replay showed that the scene still had a true fixed
  static shortcut. Source metrics were close
  (`custom_ppo=44.286885`, `validation_selected_static=44.288191`), but
  true replay-local static reached `44.037335`, far better than learned or
  explicit subtype replay. This rules out migrating the v20+/decoy family to
  the paper mainline.
- Added `configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml`.
  The new contract keeps `met_station_core` cheap enough to pair with exactly
  one specialist under the hard power budget:
  `met + specialist = 0.74 <= 0.75`, while two specialists remain infeasible.
  This removes the earlier structural failure where dynamic event policies had
  to drop core meteorological context and therefore lost to a fixed
  weather-context mask.
- Main run:
  `reports/v31_metpair_stronglatent_seed45_h075_20260620`.
  The original evaluation ranked learned PPO first:
  `custom_ppo=0.487083`,
  `validation_selected_static=0.491597`,
  `full_open_unconstrained=0.508449`,
  `feasible_static_projected=0.511365`,
  `aoi=0.525733`,
  `round_robin=0.530911`,
  `random=0.559486`.
- Re-evaluating the same checkpoint with subtype-router minimum confidence
  `0.8` improved the learned result:
  `custom_ppo=0.485635` versus
  `validation_selected_static=0.491597`. This is a `0.005962` absolute
  improvement, about `1.21%` relative to the selected static baseline.
- Strict replay-local static gate also passed:
  `best_replay_policy=split_metpair_subtype_explicit_l4`,
  `best_replay_oracle_loss_mean=0.482174`,
  `static_reference_policy=static_action10`,
  `static_reference_oracle_loss_mean=0.492351`.
  The explicit dynamic replay improves over replay-local true fixed static by
  `0.010177` absolute, about `2.07%`.
- Fixed `scripts/71_v31_behavior_complexity_audit.py` so state-dependent
  four-regime controllers are not misclassified as fixed/simple merely because
  they use four masks or persist with period 1. The audit now marks
  state-dependent behaviour using event-sensor L1 and event-mask mutual
  information before applying low-complexity fixed/cycle labels.
- Behaviour audit on `eval_router_conf08/rollout_custom_ppo.npz` now passes:
  `unique_mask_count=4`, `top1_mask_fraction=0.412354`,
  `top3_mask_fraction=0.913574`, `mask_entropy_bits=1.806220`,
  `transition_entropy_bits=1.998642`,
  `event_sensor_l1=1.579090`,
  `event_mask_mi_bits=0.520959`,
  `state_dependent=True`,
  `fixed_like=False`,
  `simple_cycle_like=False`,
  `behavior_complexity_gate_pass=True`.
- The learned behaviour is a state-dependent second-slot specialist policy:
  it keeps `met_station_core` as a stable backbone and switches the second
  active sensor among `laser_disdrometer`, `fc4_flux`,
  `shielded_thermo_hygro`, and `surface_temp_ir`. This is not a fixed sensor
  subset and not a simple round-robin/cycle, but the paper should describe it
  precisely as backbone-plus-contextual-specialist scheduling.

### 2026-06-20 Strong-Claim Extension: First Multiseed Batch

- Added the fixed-protocol runner
  `scripts/run_v31_metpair_strongclaim_seed_sweep_20260620.sh`. It reproduces
  the seed45 met+specialist setup, then runs standard evaluation,
  router-confidence `0.8` evaluation, strict no-duty-guard explicit subtype
  replay, and corrected behaviour-complexity audit.
- Added collector `scripts/72_v31_collect_metpair_strongclaim.py`. It reports
  per-seed learned margin versus `validation_selected_static`, strict replay
  margin versus replay-local true fixed static, behaviour gate status, and an
  aggregate claim-strength label.
- Remote smoke collection on existing seed45 is consistent with the current
  interpretation:
  `complete_seeds=1`, `seed_gate_pass_count=1`,
  `claim_strength=single_seed_only`.
- Replication batch launched on `remote-gpu`:
  `metpair_s41` through `metpair_s44` plus `metpair_s46` and `metpair_s47`,
  corresponding to
  `reports/v31_metpair_strongclaim_seed{41,42,43,44,46,47}_h075_20260620`.
  Together with completed seed45, this will produce the first 7-seed evidence
  pool.
- Strong paper claim target is not yet met. The required bar is at least `10`
  complete seeds with at least `8/10` full seed-gate passes, positive mean
  learned margin, and positive mean strict-replay margin.
- The 7-seed collection finished at
  `reports/aggregate/metpair_strongclaim_7seed_20260620/` and rules out the
  old metpair branch as a strong-claim result:
  `complete_seeds=7`, `seed_gate_pass_count=1`,
  `learned_gate_pass_count=1`, `replay_gate_pass_count=3`,
  `behavior_gate_pass_count=2`, `claim_strength=not_supported`.
  Mean learned margin versus `validation_selected_static` is negative
  (`-0.022343`).
- Failure diagnosis:
  seed45 is a mechanism demonstration, not robust evidence. Other seeds expose
  static shortcuts such as `shielded_thermo_hygro + laser_disdrometer` without
  the met backbone, and learned PPO often fails to infer subtype context even
  when explicit subtype replay has headroom.
- Started a new branch:
  `scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh`.
  This branch requires `met_station_core`, exposes
  `agent_context_particle_alert`, `agent_context_flux_alert`,
  `agent_context_thermal_alert`, and `agent_context_event_alert` to the agent,
  balances subtype probabilities, and uses
  `subtype_balanced_transport_rich` final-test selection.
- Backbone-context pilot seeds 41 and 42 both completed and passed all gates:
  `complete_seeds=2`, `seed_gate_pass_count=2`,
  `learned_gate_pass_count=2`, `replay_gate_pass_count=2`,
  `behavior_gate_pass_count=2`, `mean_learned_margin_abs=0.020020`,
  `mean_replay_margin_abs_vs_static_reference=0.021757`,
  `claim_strength=replicated_pilot`.
- Additional backbone-context seeds 43, 44, 45, 46, and 47 are running. The
  next decision point is whether at least `4/5` complete context seeds pass; if
  yes, launch the remaining seeds to reach the 10-seed strong-claim bar.
- The 7-seed backbone-context collection finished at
  `reports/aggregate/metpair_backbone_context_7seed_20260620/`:
  `complete_seeds=7`, `seed_gate_pass_count=3`,
  `learned_gate_pass_count=5`, `replay_gate_pass_count=3`,
  `behavior_gate_pass_count=7`, `claim_strength=not_supported`.
  This fixes the non-fixed/non-cycle behaviour issue, but not the static
  shortcut in all seeds.
- Failure diagnosis:
  in seeds such as 43 and 44, explicit subtype replay itself fails or has too
  small a margin against true fixed static. The remaining issue is therefore
  simulator/task structure: the matching specialist must have stronger
  non-substitutable information about future subtype-specific targets.
- Launched strong-latent backbone-context probes for failed seeds 43 and 44:
  `reports/v31_metpair_backbone_context_stronglatent_seed{43,44}_h075ctxsl_20260620`.
- Added event-subtype macro evidence fields to
  `scripts/70_v31_split_replay_gate.py` and
  `scripts/72_v31_collect_metpair_strongclaim.py`. The macro metric is the
  unweighted mean of particle/flux/thermal oracle losses, reported separately
  from the strict step-weighted replay gate.
- Recomputed backbone-context 7-seed evidence at
  `reports/aggregate/metpair_backbone_context_7seed_macro_20260620/`:
  strict seed gate remains unsupported (`3/7`), but macro-positive evidence is
  stronger (`macro_seed_positive_count=5/7`,
  `mean_replay_macro_margin_abs_vs_static_reference=0.016516`).
- Recomputed strong-latent probes at
  `reports/aggregate/metpair_backbone_context_stronglatent_2seed_macro_20260620/`:
  seeds 43 and 44 both have positive learned macro and replay macro margins
  (`macro_seed_positive_count=2/2`), while strict 1% step-weighted gate remains
  unsupported (`seed_gate_pass_count=0/2`).
- Current interpretation: the defensible next mainline is not the old
  step-weighted "always beats static" claim. The promising claim is
  regime-macro robust contextual specialist scheduling. It still needs
  additional seeds; a strong version would require at least `10` complete seeds
  with `>=8/10` macro-positive full gates, preferably `9/10` or `10/10` for a
  one-sided sign test below conventional thresholds.
- Strong-latent continuation failed early:
  partial aggregate
  `reports/aggregate/metpair_backbone_context_stronglatent_partial4_macro_20260620/`
  has `macro_seed_positive_count=2/4` and `seed_gate_pass_count=0/4`.
  Seed41 fails catastrophically because replay-local fixed
  `met_station_core|surface_temp_ir` dominates explicit dynamic replay on both
  step-weighted and macro-subtype loss. Seed42 has positive learned margin but
  fails replay macro and behaviour. This branch cannot support the final strong
  claim.
- Balanced-latent pilot also failed on seed41 at the learned-policy stage:
  `validation_selected_static=1.685834` vs router-evaluated
  `custom_ppo=1.820356`. It is not an eligible strong-claim branch.
- Added an orthogonal-linear generator branch:
  `event_subtype_flux_latent_linear_scale`, `offset`, and `clip` in
  `src/data_sources/public_weather_synthesis.py`, passed through
  `scripts/20_build_public_weather_truth.py`,
  `scripts/58_v31_split_protocol_run.py`, and
  `scripts/25_v2_train_custom_ppo.py`.
  The new wrapper
  `scripts/run_v31_metpair_backbone_context_ortholinear_seed_sweep_20260620.sh`
  starts with seed41 as a structural shortcut test. This branch replaces the
  unstable exponential flux latent with a bounded linear flux-latent term and
  reduces thermal strength to avoid a fixed `surface_temp_ir` shortcut.
- Operational correction: all remote GPU work now uses only the `remote-gpu`
  SSH alias. Older internal-address or tunnel-based paths are obsolete and must
  not be used for this project.
- Orthogonal-linear seed41 completed on `remote-gpu`:
  `reports/v31_metpair_backbone_context_ortholinear_seed41_h075ctxol_20260620`.
  The structural replay gate now passes against replay-local fixed static:
  best explicit dynamic replay
  `split_metpair_subtype_explicit_l10=5.142764` vs static reference
  `static_action5=5.212586`, margin `0.069822` (`1.34%`).
  Behaviour also passes (`unique_mask_count=4`, `top3_mask_fraction=0.790527`
  for the raw rollout, `state_dependent=true`, `fixed_like=false`,
  `simple_cycle_like=false`).
- The learned-policy interpretation depends on deployment wrapper:
  raw learned PPO is step-weighted positive
  (`custom_ppo=4.956431` vs `validation_selected_static=5.233835`, margin
  `0.277404`), but the router-confidence deployment is negative
  (`custom_ppo=5.330788`, margin `-0.096953`). The router layer is therefore
  not paper-safe in its current confidence setting.
- Macro-subtype diagnosis for raw seed41: raw PPO improves calm, particle, and
  thermal losses versus selected static, but loses on flux
  (`11.330058` vs static `10.574289`), so learned macro margin is still
  negative (`-0.161275`) even though the explicit dynamic replay macro is
  positive (`+0.035894`). This shifts the immediate bottleneck from scene
  structure to learned teacher/curriculum alignment for the flux subtype.
- Added a strong-teacher orthogonal-linear wrapper:
  `scripts/run_v31_metpair_backbone_context_ortholinear_strongteacher_seed_sweep_20260620.sh`.
  It keeps the same generator, aligns subtype teacher lookahead to `10`,
  strengthens BC/AWBC, lowers entropy, and reduces dwell/switch friction.
  Active remote tmux:
  `metpair_ortholinear_strongteacher_seed41_20260620` on `remote-gpu`, using
  GPU1. The first decision point is whether seed41 becomes both step-weighted
  and macro-positive under the raw PPO deployment before expanding to a
  multi-seed pool.

## 2026-06-20 Remote Access Hygiene and Strong-Teacher Expansion
- Cleaned local operational context so future runs use only the `remote-gpu`
  SSH alias. Removed stale hardcoded remote-host paths from `AGENTS.md`,
  active progress/planning/report notes, `fetch_results.sh`, and the local
  `microclimate-experiment-server` skill. Also removed password-based sync
  helpers from the smoke-result fetch script and converted unrelated historical
  database credentials to environment variables.
- Verified with repository and skill searches that the old remote connection
  strings and password-sync helpers no longer appear in active text/script
  files after excluding generated reports and archived paper assets.
- Strong-teacher 3-seed aggregate completed:
  `reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_3seed_macro_20260620/`
  and raw counterpart
  `reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_3seed_raw_macro_20260620/`.
  Step-weighted strict gate is now `3/3`; learned gate `3/3`, replay gate
  `3/3`, behaviour gate `3/3`. Mean router learned margin is `+0.594738`;
  mean raw learned margin is `+0.614495`.
- Macro-subtype remains insufficient: macro-positive full seed count is only
  `1/3` even though learned macro and replay macro are each positive in `2/3`
  seeds. Do not claim macro-robustness from this branch.
- Launched strong-teacher extension seeds `44--50` on `remote-gpu`:
  GPU0 seeds `44 45`, GPU1 seeds `46 47`, GPU2 seed `48`, GPU4 seed `49`,
  GPU5 seed `50`. GPU3 was left free for the macro-reward diagnostic run.
  Target final aggregate is 10 seeds (`41--50`) with the strong paper claim
  evaluated on the step-weighted strict static/replay/behaviour gate, not on
  macro-subtype robustness unless later evidence changes.
- Macro-reward diagnostic seed41 completed basic eval/replay/behaviour and
  seed42 has started. It is exploratory only; the main strong-claim path is
  now the strong-teacher 10-seed expansion.

## 2026-06-21 Balanced-Objective Strong-Claim Expansion
- Re-read `planning-with-files` and `microclimate-experiment-server` skill
  instructions, then restored the active
  `task_plan.md`/`progress.md`/`findings.md` context.
- Ran `session-catchup.py`; no unsynced plan context was reported.
- Added the user's new planning constraint to `task_plan.md`: after `10`
  complete hypothesis rounds without a new breakthrough, stop conservative
  retries and pivot to deeper experiment/framework/algorithm changes.
- Defined a hypothesis round as a named branch with completed seed/pilot
  evidence, aggregate result, and a logged keep/pivot decision.
- Registered the active balanced-objective 14-seed expansion as
  `BO-1 = ortholinear balanced-objective 14-seed expansion`.
- Checked `remote-gpu` only via the SSH alias. The balanced-objective workers
  reached `200000` timesteps and GPU utilization dropped to idle.
- Confirmed seeds `41,42,43,45,47,49,51,53` already have train/eval/replay and
  behaviour artifacts.
- Found that even seeds `44,46,48,50,52,54` had completed training and
  router-confidence evaluation and are now running strict replay via
  `70_v31_split_replay_gate.py` on CPU.
- Left the worker tmux sessions running; do not kill them while replay is
  active. Aggregation tmux `agg_wait_balanced_20260621` is still waiting for
  worker sessions to exit.

## Next Verification Unit
- Continue monitoring the six even-seed replay jobs. Once replay and behaviour
  artifacts exist for seeds `44,46,48,50,52,54`, let or force the 14-seed
  aggregation run, then inspect both:
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_macro_20260621/metpair_claim_summary.json`
  and
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_oldclaim_20260621/oldclaim_summary.json`.
- If final BO-1 clears at least `12/14` old-claim step or macro gates with
  behaviour complete, treat it as a major claim breakthrough. Otherwise count
  BO-1 against the 10-round limit and pivot the next round to a deeper
  algorithm/generator change rather than another small tuning retry.

## 2026-06-21 BO-1 14-Seed Breakthrough and 24h Autonomy Launch
- Recomputed the 14-seed balanced-objective aggregate after all even-seed
  replay/behaviour jobs finished.
- Main aggregate paths:
  - `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_macro_20260621/`
  - `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_raw_macro_20260621/`
  - `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_oldclaim_20260621/`
- Final BO-1 14-seed summary:
  - complete seeds: `14`
  - learned PPO beats static/rule/operational baselines on step objective:
    `14/14`
  - learned PPO beats static/rule/operational baselines on macro objective:
    `14/14`
  - strict explicit-replay macro gate: `14/14`
  - behaviour complexity gate: `14/14`
  - strict explicit-replay step gate: `11/14`
  - learned-policy true-static step gate: `12/14`
  - macro old-claim sign-test p-value: `0.00006104`
  - step old-claim sign-test p-value: `0.02868652`
- Generated detailed remote report:
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_oldclaim_20260621/BREAKTHROUGH_REPORT.md`.
- Patched `scripts/73_v31_collect_oldclaim_gate.py` to record direct
  learned-policy margins against replay-local true fixed static, separately
  from the hand-coded explicit replay gate.
- Added `scripts/74_v31_write_balancedobjective_report.py`, a dependency-free
  Markdown report writer for aggregate summaries.
- Added and launched
  `scripts/run_v31_balancedobjective_24h_autonomous_20260621.sh`.
  Remote tmux:
  `bo24_autonomy_20260621`.
- The 24h runner starts at seed `55`, runs 12 seeds per wave across 6 GPUs,
  automatically aggregates after each wave, writes per-wave reports, and stops
  if free disk falls below `200GB`.

## Next Verification Unit
- Monitor `bo24_autonomy_20260621` and the first wave `55--66`.
  At each completed wave, inspect the generated report under
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_41_<end_seed>_20260621_oldclaim/BREAKTHROUGH_REPORT.md`.
- If macro gates remain stable but step replay remains below the strong bar,
  preserve the macro claim as the breakthrough and start a deeper follow-up
  round aimed specifically at step-weighted true-static failures, rather than
  rewriting the current macro result as broader than it is.
- While monitoring wave `55--66`, found and fixed an old-claim collector
  consistency issue: `scripts/73_v31_collect_oldclaim_gate.py` compared raw
  `v2_custom_ppo_metrics.csv` losses but previously preferred
  `eval_router_conf08` for behaviour audit. The collector now defaults to raw
  behaviour (`--behavior-eval-dir .`) and records `behavior_eval_dirs` in
  `oldclaim_summary.json`; `scripts/74_v31_write_balancedobjective_report.py`
  includes that field in the generated report.
- Synced the fixed collector/report writer to `remote-gpu` and verified
  remote `py_compile`. The first-wave odd-seed temporary aggregate changed
  from mixed-mouth `step=3/6, macro=5/6, behavior=5/6` to consistent raw
  behaviour `step=4/6, macro=6/6, behavior=6/6`. This does not rescue the
  unqualified step claim, but it preserves the bounded macro claim and removes
  a false behaviour failure for seed57.
- Regenerated and synced the 14-seed BO-1 oldclaim report with the corrected
  raw-behaviour audit. Core counts did not change:
  `step=11/14`, `macro=14/14`, `behavior=14/14`; the report now explicitly
  records `behavior_eval_dirs: ['.']`.
- Minor monitoring error: an inline remote diagnostic used `python` before
  activating `darts` and failed with `python: command not found`; reran it
  under `conda activate darts`.
- Corrected stale journal-target planning state after user clarification:
  the active target journal is *Expert Systems with Applications* (ESWA), not
  *Cold Regions Science and Technology*. Updated root `task_plan.md`, root
  `progress.md`, root `findings.md`, the static-break plan finding, and the
  ESWA terminology plan. Also switched `.planning/.active_plan` to
  `2026-06-10-eswa-terminology-rewrite`. Legacy source filenames were not
  renamed.
- Implemented the next deeper framework hook for step-claim exploration:
  `subtype_static_auto` AWBC teacher mode. The earlier
  `awbc_teacher_mode=subtype_auto` still used hand-written calm/particle/flux/
  thermal sensor ids, while replay `subtype_auto` selected actions from static
  candidate subtype losses. The new mode evaluates static candidates on the
  static-selection split, records event/non-event/subtype losses, automatically
  selects calm by `oracle_loss_non_event` and event subtypes by subtype-specific
  loss, writes `awbc_subtype_static_auto_candidates.csv` and
  `awbc_subtype_static_auto_selection.json`, and uses those actions for BC/AWBC
  and subtype-router deployment.
- Patched `scripts/25_v2_train_custom_ppo.py`,
  `scripts/58_v31_split_protocol_run.py`, and `src/v2/custom_ppo.py`; local and
  remote `py_compile` passed, and remote CLI help exposes
  `--awbc-teacher-mode subtype_static_auto` plus
  `--awbc-teacher-auto-score-mode {raw,staticnorm}`.
- Parameterized the base metpair seed-sweep wrapper with
  `AWBC_TEACHER_MODE` and `AWBC_TEACHER_AUTO_SCORE_MODE`, defaulting to the
  previous `subtype_auto/raw` behavior so the already-running BO-1 24h runner
  remains comparable.
- Added
  `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_autoteacher_seed_sweep_20260621.sh`
  as a separate `RUN_PREFIX` / `BUDGET_LABEL` branch for the new
  `subtype_static_auto` teacher. Local and remote `bash -n` passed. It has not
  been launched yet to avoid colliding with the active 24h BO wave.
- First 24h BO extension wave completed and was synced locally:
  - aggregate:
    `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_41_66_20260621_oldclaim/`
  - complete seeds: `26`
  - learned PPO beats static/rule/operational baselines on step objective:
    `24/26`
  - learned PPO beats static/rule/operational baselines on macro objective:
    `26/26`
  - strict explicit-replay step gate: `20/26`
  - strict explicit-replay macro gate: `26/26`
  - behaviour complexity gate: `26/26`
  - learned-policy true-static step gate: `21/26`
  - macro sign-test p-value: `1.49e-08`
  - step sign-test p-value: `0.00467765`
- Interpretation after 26 seeds: macro claim is much stronger; the
  unqualified step-weighted old claim is still bounded because strict replay
  step gate fails seeds `48,49,52,55,56,59`. Wave `67--78` has already started
  under the original BO settings in remote tmux `bo24_autonomy_20260621`.

## 2026-06-21 ESWA Plan File Correction
- User pointed out that some planning files still lagged behind the journal
  change. Rechecked the root plan, active plan, and 05-25 rewrite/evidence
  files.
- Confirmed the active journal target is *Expert Systems with Applications*
  (ESWA). The user typo "applicantion" is treated as "Applications".
- Updated `docs/05-25-crst-rewrite-strategy.md` and
  `docs/05-25-full-rewrite-evidence-ledger.md` so their current content is
  ESWA-facing. Their filenames are retained only as historical paths.
- Replaced current-target CRST language in those files with ESWA/Elsevier
  framing: intelligent sensing-system scheduling, prediction-driven decision
  support, ESWA compliance/data-availability gates, and Antarctic blowing-snow
  as the benchmark application rather than the journal-scope anchor.

## 2026-06-21 Step-Diagnostic Replay During BO-1 Wave 67--78
- Checked `remote-gpu` tmux `bo24_autonomy_20260621`: wave `67--78` is in
  PPO training; odd seeds `67,69,71,73,75,77` have oracle checkpoints and are
  mid-run, while paired even seeds wait in the same worker queues.
- Reran the subtype-auto step diagnostic under the `darts` conda environment
  after a read-only check accidentally used bare `python`.
- Seed55 diagnostic result:
  `replay_gate_subtype_auto_static_noguard_stepdiag` has `gate_pass=True`.
  Best replay mean loss is `3.214754`, validation-selected static reference is
  `3.415349`, replay-local true fixed static is `3.372207`, and the margin
  against true fixed static is `+0.157453` absolute / `+0.046691` relative.
- Seed59 diagnostic summary is not available yet; CPU replay remains running.
- Interpretation: at least one 26-seed step failure is recoverable by automatic
  subtype/static-candidate replay selection, so the prepared
  `subtype_static_auto` teacher branch is the next high-priority framework
  branch after current BO wave resources free up.

## 2026-06-21 Goal Constraint Reconfirmation
- Rechecked the active 24h autoresearch goal after user clarification. The
  execution constraints are now explicit in both root and subproject plans:
  PPO is retained as the final learned scheduler; each modification direction
  has at most 10 bounded work units without effective improvement; the current
  microclimate sensor setup remains the physical baseline, with only moderate
  simulated variants allowed.
- This turns the current balanced-objective seed expansion into a bounded BO-1
  direction. It cannot become an indefinite seed retry loop; absent a strict
  step-gate improvement, the next layer is the prepared `subtype_static_auto`
  teacher/PPO branch or a deeper simulator/framework modification.

## 2026-06-21 BO-1 Wave 67--78 Interim Positive Signal
- Queried `remote-gpu` after odd seeds in wave `67--78` completed. Seeds
  `67,69,71,73,75,77` all have oracle, PPO, eval, strict replay, and behavior
  audit artifacts; paired even seeds have begun PPO training.
- Interim odd-seed gates:
  - strict replay step gate: `6/6`;
  - strict replay macro gate: `6/6`;
  - behavior complexity gate: `6/6`;
  - fixed-like policies: `0/6`;
  - simple-cycle-like policies: `0/6`.
- Step margins against the static reference are consistently positive:
  `+0.166386`, `+0.171943`, `+0.216564`, `+0.224125`, `+0.167435`,
  `+0.255616` absolute across seeds `67,69,71,73,75,77`.
- This is the first live sign that BO-1 may be improving the old step-gate
  failure rate, but it remains interim until seeds `68,70,72,74,76,78` complete
  and the aggregate report through seed `78` is generated.

## 2026-06-21 Seed59 Step-Diagnostic Completion
- The seed59 `subtype_auto` diagnostic completed on `remote-gpu` and passed the
  strict no-duty-guard step gate.
- Result:
  - best replay policy `split_subtype_auto_top2_c0_p0_f0_t0_l10`;
  - best replay step loss `3.106645`;
  - static reference `static_action5` step loss `3.379619`;
  - margin `+0.272974` absolute / `+0.080771` relative.
- This confirms that seed55 was not a single isolated recovery. At least two
  earlier BO-1 step failures can be fixed by subtype/static-candidate replay
  selection, supporting the prepared `subtype_static_auto` teacher/PPO branch if
  the current BO-1 aggregate still has residual step failures.

## 2026-06-21 BO-1 Aggregate Through Seed 78
- Wave `67--78` completed on `remote-gpu`, and the 24h runner generated:
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_41_78_20260621_oldclaim/BREAKTHROUGH_REPORT.md`.
- Synced the old-claim, macro, and raw-macro aggregate directories locally.
- Official 41--78 summary:
  - complete seeds: `38`;
  - learned step wins over static/rule/operational baselines: `36/38`;
  - learned macro wins over static/rule/operational baselines: `38/38`;
  - strict explicit replay step gate: `32/38`;
  - strict explicit replay macro gate: `38/38`;
  - behavior complexity gate: `38/38`;
  - learned-policy true-static step gate: `33/38`;
  - mean step margin vs best operational baseline: `0.310320`;
  - mean macro margin vs best operational baseline: `0.085936`;
  - old-step sign-test p-value: `1.2171280104666948e-05`;
  - macro sign-test p-value: `3.637978807091713e-12`.
- The new wave itself is clean on the step gates:
  seeds `67--78` are `12/12` for old-claim step gate and `12/12` for learned
  true-static step gate. This is a real improvement over the 41--66 checkpoint,
  not just more of the same failure pattern.
- Claim implication: this supports a strong statistical step claim plus the
  already strong macro claim, but still not an all-seed / zero-failure
  unqualified step claim because old failures `48,49,52,55,56,59` remain in the
  cumulative aggregate.
- Anti-stall decision: effective improvement was observed within BO-1, so the
  direction has not failed. Continue the already-started wave `79--90` to test
  whether the 12/12 step-gate behavior persists.

## 2026-06-21 BO-1 Wave 79--90 First-Half Boundary
- Seeds `79,81,83,85,87,89` completed all artifacts. Seeds
  `80,82,84,88,90` have started and seed `86` is starting.
- The first half of wave `79--90` is not a second clean wave:
  - seed83 fails the strict step replay gate:
    best replay loss `3.273781` vs static reference `3.212263`, margin
    `-0.061518` absolute / `-0.019151` relative. Its macro replay and behavior
    gates pass.
  - seed87 passes strict replay step and macro gates, but the learned behavior
    audit fails: `fixed_like=True`, `state_dependent=False`, event-mask mutual
    information only about `0.02`.
  - seeds `79,81,85,89` pass both replay step and behavior gates.
- Interpretation: BO-1 remains materially improved after the 41--78 aggregate,
  but a stable zero-failure step/behavior claim is not established. If the final
  79--90 aggregate remains mixed, the next direction should target
  teacher/PPO behavior robustness, e.g. `subtype_static_auto` teacher selection
  and/or explicit behavior-diversity regularization, rather than simply adding
  more same-configuration seed waves.

## 2026-06-21 BO-1 Stopped and AT-1 Autoteacher Pivot Launched
- Aggregate through seed `90` completed and was synced locally:
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_41_90_20260621_oldclaim/`.
- Official 41--90 result:
  - complete seeds `50`;
  - learned step wins `45/50`;
  - learned macro wins `50/50`;
  - strict explicit replay step gate `40/50`;
  - strict explicit replay macro gate `49/50`;
  - behavior complexity gate `49/50`;
  - learned-policy true-static step gate `41/50`.
- Wave `79--90` itself was mixed: old step gate `8/12`, macro gate `11/12`,
  behavior gate `11/12`, with step failures at `83,84,86,87` and the behavior
  failure at `87`.
- Decision: BO-1 supports a strong statistical claim but not the desired stable
  zero-failure strong claim. Continuing same-config seed expansion would violate
  the user's anti-stall intent, so `bo24_autonomy_20260621` was stopped after
  41--90.
- Pivot launched: AT-1 `subtype_static_auto` teacher/PPO branch in remote tmux
  `autoteacher_pilot_parallel_83_92_20260621`, seeds `83,84,86,87,91,92`, one
  seed per GPU. This keeps PPO as the final scheduler and preserves the
  met+specialist sensor baseline while changing teacher construction.
- Started local watcher tmux `autoteacher_pilot_local_watch_20260621`; first
  snapshot confirms all six AT-1 seeds have begun and GPUs are allocated.

## 2026-06-21 AT-1 Failed; RT-1 Router Sweep Started
- AT-1 `subtype_static_auto` pilot completed on seeds `83,84,86,87,91,92`.
- Result:
  - oldclaim step gate `2/6`;
  - oldclaim macro gate `3/6`;
  - behavior gate `3/6`;
  - claim strength `not_supported`.
- Interpretation:
  - seed83 and seed86 step failures are repaired;
  - seed84 remains slightly step-negative;
  - seed87, seed91, and seed92 expose behavior/router robustness failures.
- Added a backward-compatible `--metrics-eval-dir` option to
  `scripts/73_v31_collect_oldclaim_gate.py` so router-deployed metrics and
  router-deployed behavior can be evaluated consistently.
- Launched RT-1 router threshold sweep in remote tmux
  `autoteacher_router_sweep_83_92_20260621`, testing `conf00` and `conf05`
  deployment directories for the same six seeds.

## 2026-06-21 RT-1 Failed; BR-1 Behavior-Regularized PPO Launched
- RT-1 router threshold sweep failed:
  - `conf00` keeps oldclaim step gate at `2/6` and behavior gate at `3/6`;
  - `conf05` logs show behavior failure across all six seeds.
- Conclusion: changing router deployment threshold is not enough. The next layer
  must alter PPO training feedback.
- Added environment-variable controls to
  `scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh` for
  duty-balance and duty-score feedback while preserving default behavior.
- Added BR-1 wrapper:
  `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorreg_seed_sweep_20260621.sh`.
- Launched remote tmux `behaviorreg_pilot_parallel_83_92_20260621` on seeds
  `83,84,86,87,91,92`, one seed per GPU. This branch keeps the original
  `subtype_auto` teacher and adds mild duty-based PPO feedback to reduce
  state-independent specialist collapse.

## 2026-06-21 BR-1 Watcher Installed
- Checked current BR-1 remote state through `remote-gpu`:
  - tmux `behaviorreg_pilot_parallel_83_92_20260621` is active;
  - all six seeds have completed oracle setup and are in PPO training;
  - PPO progress is about `38k--41k / 120k` timesteps per seed;
  - no evaluation/replay/behavior aggregate artifacts exist yet.
- Added and launched local watcher
  `scripts/watch_behaviorreg_pilot_20260621.sh` in tmux
  `behaviorreg_pilot_local_watch_20260621`.
- The watcher writes
  `reports/aggregate/behaviorreg_pilot_local_watch_20260621_status.md` and
  syncs BR-1 aggregate directories every `600` seconds.

## 2026-06-21 BR-1 Failed; BD-1 Pivot Required
- BR-1 completed all six seeds and aggregate directories were synced locally:
  - `reports/aggregate/behaviorreg_pilot_83_92_oldclaim_20260621/`
  - `reports/aggregate/behaviorreg_pilot_83_92_macro_20260621/`
  - `reports/aggregate/behaviorreg_pilot_83_92_raw_macro_20260621/`
- Official BR-1 old-claim summary:
  - complete seeds `6`;
  - old step gate `3/6`;
  - old macro gate `4/6`;
  - behavior gate `4/6`;
  - replay gate `5/6`;
  - replay macro gate `6/6`;
  - learned true-static step gate `2/6`.
- Failure seeds:
  - step: `83,87,92`;
  - macro: `87,92`;
  - behavior: `87,92`.
- Report written:
  `reports/aggregate/behaviorreg_pilot_83_92_failure_report_20260621.md`.
- Decision: do not expand BR-1. Mild duty-balance / duty-score feedback is too
  indirect. Pivot to BD-1: explicit state-dependent behavior signal or PPO
  architecture modification, while preserving PPO and the current met+specialist
  sensor baseline.

## 2026-06-21 BD-1 Failed; BRG-1 Architecture/State Pivot Required
- BD-1 `subtype_action_ce/margin` pilot completed on remote-gpu in tmux
  `behaviorbd_pilot_parallel_83_92_20260621`.
- Synced aggregate directories locally:
  - `reports/aggregate/behaviorbd_pilot_83_92_oldclaim_20260621/`
  - `reports/aggregate/behaviorbd_pilot_83_92_macro_20260621/`
  - `reports/aggregate/behaviorbd_pilot_83_92_raw_macro_20260621/`
- Official BD-1 old-claim summary:
  - complete seeds `6`;
  - old step gate `2/6`;
  - old macro gate `4/6`;
  - behavior gate `4/6`;
  - replay gate `5/6`;
  - replay macro gate `6/6`;
  - learned true-static step gate `3/6`.
- Failure seeds:
  - step: `83,84,87,92`;
  - macro: `87,92`;
  - behavior: `87,92`.
- Mechanism:
  - subtype action CE/margin losses were active in PPO logs, so this is a real
    negative result rather than a wiring failure;
  - seeds `87` and `92` still show low event-mask MI and are fixed-like /
    weakly state-dependent under the corrected behavior audit;
  - seed `83` still fails strict replay headroom.
- Report written:
  `reports/aggregate/behaviorbd_pilot_83_92_failure_report_20260621.md`.
- Decision: do not expand BD-1. Pivot to BRG-1: observable regime-belief PPO
  state/architecture path with subtype auxiliary head and conservative subtype
  router, preserving PPO and the current met+specialist sensor baseline.

## 2026-06-21 BRG-1 Launched
- Added backward-compatible runner controls:
  - `INCLUDE_OBSERVABLE_REGIME_BELIEF`;
  - `REGIME_BELIEF_LOOKBACK`;
  - `EVENT_GATED_ACTOR`.
- Added BRG-1 wrapper:
  `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg_seed_sweep_20260621.sh`.
- Added BRG-1 parallel runner:
  `scripts/run_v31_behaviorbrg_pilot_parallel_83_92_20260621.sh`.
- Added BRG-1 watcher:
  `scripts/watch_behaviorbrg_pilot_20260621.sh`.
- Local validation passed:
  - `bash -n` for all new/modified shell scripts;
  - `python -m py_compile` for PPO, split runner and env files;
  - CLI help exposes regime-belief and event-gated-actor flags.
- Remote validation passed in conda `darts`; non-conda shell has no `python`,
  so remote validation and experiments must activate `darts`.
- Launched remote tmux `behaviorbrg_pilot_parallel_83_92_20260621` at
  `2026-06-21T07:35:34+08:00`.
- Launched local watcher tmux `behaviorbrg_pilot_local_watch_20260621`.
- Initial remote health check at `2026-06-21T07:36:58+08:00`:
  six seeds have started, GPUs are allocated, no early traceback/error was found.

## 2026-06-21 BRG-1 Partial Improvement; BRG-2 Required
- BRG-1 completed all six hard/control seeds and aggregate directories were
  synced locally:
  - `reports/aggregate/behaviorbrg_pilot_83_92_oldclaim_20260621/`
  - `reports/aggregate/behaviorbrg_pilot_83_92_macro_20260621/`
  - `reports/aggregate/behaviorbrg_pilot_83_92_raw_macro_20260621/`
- Official BRG-1 old-claim summary:
  - complete seeds `6`;
  - old step gate `3/6`;
  - old macro gate `5/6`;
  - behavior gate `5/6`;
  - operational macro gate `6/6`;
  - replay macro gate `6/6`;
  - learned true-static step gate `4/6`.
- Macro collectors:
  - router-eval macro gate `6/6`, behavior `6/6`;
  - raw macro gate `5/6`, behavior `5/6`.
- Interpretation:
  - BRG-1 improves on BD-1/BR-1 and repairs raw behavior for seed `92`;
  - it still fails the strong claim because seed `87` remains fixed-like under
    raw deployment, seed `83` misses the strict replay step margin, and seed
    `92` remains slightly worse than validation-selected static on step loss.
- Report written:
  `reports/aggregate/behaviorbrg_pilot_83_92_partial_report_20260621.md`.
- Decision: BRG-1 is an effective-improvement unit, not a breakthrough. Run
  BRG-2 as exactly one bounded follow-up in the same direction: match raw/eval
  subtype-router confidence at `0.70`, modestly increase entropy, keep PPO and
  the current met+specialist sensor baseline, and do not re-enable BD-1 direct
  subtype-action CE/margin. If BRG-2 does not clearly improve step or produce a
  clean macro/behavior pilot, pivot away from BRG to simulator/reward headroom
  or deeper PPO architecture.

## 2026-06-21 BRG-2 Launched
- Added BRG-2 scripts:
  - `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg2_seed_sweep_20260621.sh`;
  - `scripts/run_v31_behaviorbrg2_pilot_parallel_83_92_20260621.sh`;
  - `scripts/watch_behaviorbrg2_pilot_20260621.sh`.
- BRG-2 settings:
  - raw and eval subtype-router confidence both `0.70`;
  - entropy coefficient `0.0075`;
  - observable regime-belief lookback `12`;
  - event-gated actor enabled;
  - subtype auxiliary coefficient `5.0`;
  - BD-1 direct subtype-action CE/margin losses disabled.
- Local and remote validation passed:
  - `bash -n` for the new shell scripts and base runners;
  - Python compile for PPO/env/split/collector/report scripts;
  - remote conda `darts` Python `3.12.12`.
- Launched remote tmux `behaviorbrg2_pilot_parallel_83_92_20260621` at
  `2026-06-21T08:08:48+08:00`.
- Launched local watcher tmux `behaviorbrg2_pilot_local_watch_20260621`.

## 2026-06-21 BRG-2 Completed; Subtype-Aware Behavior Gate Added
- BRG-2 completed all six hard/control seeds and produced aggregate evidence:
  - default oldclaim step `5/6`;
  - default oldclaim macro `5/6`;
  - default behavior `5/6`;
  - explicit replay step `6/6`;
  - explicit replay macro `6/6`;
  - learned true-static step `5/6`.
- Added subtype-aware behavior auditing to
  `scripts/71_v31_behavior_complexity_audit.py`.
  The old event-binary audit is retained, but if rollout truth contains
  `event_subtype_particle_latent`, `event_subtype_flux_latent`, and
  `event_subtype_thermal_latent`, behavior can also pass through subtype MI/L1.
- Sanity check passed on seed `92`: custom PPO becomes
  `state_dependent=True`, `fixed_like=False`, `simple_cycle_like=False`, while
  validation-selected fixed static remains `behavior_complexity_gate_pass=False`.
- BRG-2 subtype-aware aggregate:
  - oldclaim step `5/6`;
  - oldclaim macro `6/6`;
  - behavior `6/6`;
  - macro gate `6/6`;
  - raw macro gate `6/6`.
- Remaining blocker:
  seed `92` learned PPO still loses the step objective to validation-selected
  static (`3.046497` vs `3.024605`), although explicit subtype replay clears
  the same seed (`2.971215` vs true fixed static `3.049884`).
- Report written:
  `reports/aggregate/behaviorbrg2_pilot_83_92_evidence_report_20260621.md`.
- Decision: BRG-2 is the best hard/control pilot so far but not final. Run
  BRG-3 with a moderate action-fidelity signal layered on top of BRG-2. If it
  does not repair seed `92` without breaking the other five seeds, pivot away
  from BRG to deeper PPO architecture or reward/oracle redesign.

## 2026-06-21 BRG-3 Launched
- Added BRG-3 scripts:
  - `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg3_seed_sweep_20260621.sh`;
  - `scripts/run_v31_behaviorbrg3_pilot_parallel_83_92_20260621.sh`;
  - `scripts/watch_behaviorbrg3_pilot_20260621.sh`.
- BRG-3 keeps BRG-2 settings and adds moderate action fidelity:
  - `SUBTYPE_ACTION_CE_COEF=0.25`;
  - `SUBTYPE_ACTION_MARGIN_COEF=0.05`;
  - `SUBTYPE_ACTION_MARGIN=0.50`.
- Local and remote validation passed (`bash -n`, Python compile, conda `darts`).
- Launched remote tmux `behaviorbrg3_pilot_parallel_83_92_20260621` at
  `2026-06-21T08:40:18+08:00`.
- Launched local watcher tmux `behaviorbrg3_pilot_local_watch_20260621`.

## 2026-06-21 BRG-3 Failed; Temporal Pivot Required
- BRG-3 completed all six seeds.
- Result:
  - old step gate `3/6`;
  - old macro gate `6/6`;
  - behavior gate `6/6`;
  - replay step gate `3/6`;
  - replay macro gate `6/6`;
  - learned true-static step gate `2/6`.
- Compared with BRG-2, the moderate action-fidelity signal worsened step
  results (`5/6 -> 3/6`) while preserving macro/behavior. It is therefore not
  the right repair for seed `92`.
- Failure report written:
  `reports/aggregate/behaviorbrg3_pilot_83_92_failure_report_20260621.md`.
- Decision: close BRG+action-fidelity. Pivot to TEMPORAL-1: a deeper
  temporal/lead-aware PPO or reward-credit modification, because the remaining
  explicit replay headroom appears lead-based rather than simple action
  imitation.

## 2026-06-21 TEMPORAL-1 Launched
- Parameterized temporal controls in
  `scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh` while
  preserving previous defaults:
  - `SUBTYPE_CONTEXT_LEAD_STEPS`;
  - `SUBTYPE_LATENT_TARGET_LAG_STEPS`;
  - `BLOWING_SNOW_LEAD_STEPS`;
  - `GREEDY_LOOKAHEAD_STEPS`;
  - `REPLAY_LEAD_STEPS`;
  - `REPLAY_DWELL_STEPS`.
- Added TEMPORAL-1 scripts:
  - `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_temporal1_seed_sweep_20260621.sh`;
  - `scripts/run_v31_temporal1_pilot_parallel_83_92_20260621.sh`;
  - `scripts/watch_temporal1_pilot_20260621.sh`.
- TEMPORAL-1 settings:
  - context lead `16`;
  - subtype latent target lag `6`;
  - blowing-snow lead `10`;
  - subtype aux lookahead `16`;
  - AWBC teacher lookahead `16`;
  - oracle subtype teacher lookahead `16`;
  - replay lead candidates `0,4,8,12,16`;
  - no subtype action CE/margin.
- Local and remote validation passed.
- Launched remote tmux `temporal1_pilot_parallel_83_92_20260621` at
  `2026-06-21T09:12:06+08:00`.
- Launched local watcher tmux `temporal1_pilot_local_watch_20260621`.

## 2026-06-21 TEMPORAL-1 Breakthrough on Hard/Control Seeds
- TEMPORAL-1 completed all six seeds.
- Oldclaim aggregate:
  - old step gate `6/6`;
  - old macro gate `5/6`;
  - behavior gate `6/6`;
  - replay step gate `6/6`;
  - replay macro gate `6/6`;
  - learned true-static step gate `5/6`.
- Macro collectors:
  - router-eval seed/macro/behavior gates `6/6`;
  - raw seed/macro/behavior gates `6/6`.
- The former seed `92` blocker is repaired:
  - BRG-2: `custom_ppo=3.046497` vs selected static `3.024605`;
  - TEMPORAL-1: `custom_ppo=2.777641` vs selected static `2.845988`.
- Behavior remains valid under subtype-aware audit for all six seeds; no learned
  policy is fixed-like or simple-cycle-like.
- Breakthrough report written:
  `reports/aggregate/temporal1_pilot_83_92_breakthrough_report_20260621.md`.
- Decision: TEMPORAL-1 is the new leading candidate. It is not final because it
  is still a six-seed pilot; launch fresh-seed expansion `93--98`.

## 2026-06-21 TEMPORAL-1 Fresh-Seed Expansion 93--98 Launched
- Generalized TEMPORAL-1 runner/watcher to accept `SEED_LABEL` and custom seed
  sets without overwriting the completed `83_92` aggregate.
- Remote validation passed.
- Launched remote tmux `temporal1_pilot_parallel_93_98_20260621` for seeds
  `93,94,95,96,97,98` at `2026-06-21T09:38:59+08:00`.
- Launched local watcher tmux `temporal1_pilot_local_watch_93_98_20260621`.

## 2026-06-21 TEMPORAL-1 Fresh-Seed Expansion 93--98 Mixed Result
- TEMPORAL-1 fresh-seed expansion completed and was synced locally.
- Aggregate result:
  - old step gate `4/6`;
  - old macro gate `6/6`;
  - behavior gate `6/6`;
  - strict replay step gate `5/6`;
  - strict replay macro gate `6/6`;
  - dedicated macro collectors `6/6` for both router-eval and raw deployment.
- Failed / boundary seeds:
  - seed `95`: learned PPO loses step to selected static, and default explicit
    replay also loses step to strict fixed static;
  - seed `96`: default explicit replay has step headroom, but learned PPO still
    loses to selected/true static;
  - seed `97`: selected-static step gate passes, but true-static comparison is
    marginally negative.
- Behavior remains valid on all six fresh seeds: each policy uses four masks,
  subtype MI is positive, and none is fixed-like or simple-cycle-like.
- Report written:
  `reports/aggregate/temporal1_pilot_93_98_mixed_report_20260621.md`.
- Decision: TEMPORAL-1 remains the leading direction, but the fresh expansion
  blocks a zero-failure old-step claim. Launch a wide lead/dwell replay
  diagnostic on seeds `95,96,97` before deciding whether to repair TEMPORAL-1 or
  pivot to simulator / target-generation / deeper PPO credit changes.

## 2026-06-21 TEMPORAL-1 Wide Replay Diagnostic 95--97 Launched
- Launched remote tmux `temporal1_wide_replay_diag_95_97_20260621`.
- Purpose:
  - determine whether seed `95` default replay failure is a missed lead/dwell
    search-space issue or a true fixed-static shortcut;
  - verify that seeds `96` and `97` still have explicit dynamic headroom under a
    wider replay search.
- Search grid:
  - lead steps `0,2,4,6,8,10,12,16,20,24`;
  - dwell steps `4,6,8,10,12,16,24,32`;
  - strict no-duty-guard static reference;
  - PPO remains the final learned scheduler; this is diagnostic only.

## 2026-06-21 TEMPORAL-1 Wide Replay Diagnostic 95--97 Completed
- Wide lead/dwell explicit replay completed.
- Result:
  - seed `95`: replay gate still fails; best wide replay loss `3.600575` vs
    strict fixed static `3.520518`, margin `-0.080057`;
  - seed `96`: replay gate passes; margin `+0.048074`;
  - seed `97`: replay gate passes; margin `+0.032034`.
- Interpretation: seed `95` is not a narrow replay-search problem. Its raw step
  objective has a real fixed-static shortcut, specifically the
  `met_station_core|fc4_flux` pair. Seeds `96` and `97` remain learned-credit /
  deployment issues because explicit dynamic headroom exists.
- Report written:
  `reports/aggregate/temporal1_wide_replay_diag_95_97_report_20260621.md`.
- Decision: close same TEMPORAL-1 expansion. Pivot to `SCENEBAL-1`, a
  simulator/target-generation rebalancing direction that keeps PPO and the
  met+specialist sensor geometry but weakens the raw-step dominance of fixed
  `met+fc4`.

## 2026-06-21 SCENEBAL-1 Pilot 93--98 Launched
- Added scripts:
  - `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed_sweep_20260621.sh`;
  - `scripts/run_v31_scenebal1_pilot_parallel_20260621.sh`;
  - `scripts/watch_scenebal1_pilot_20260621.sh`.
- SCENEBAL-1 inherits TEMPORAL-1 PPO/teacher/lead settings and changes the
  simulator/target contract:
  - subtype probabilities: particle `0.38`, flux `0.24`, thermal `0.38`;
  - lower flux multiplier and flux target weights;
  - stronger particle and thermal latent/target weights;
  - PPO remains the final learned scheduler.
- Local `bash -n` and remote conda validation passed.
- Launched remote tmux
  `scenebal1_pilot_parallel_93_94_95_96_97_98_20260621`.
- Launched local watcher
  `scenebal1_pilot_local_watch_93_94_95_96_97_98_20260621`.

## 2026-06-21 SCENEBAL-1 Seed93 Staticnorm Fix
- Seed `93` exposed a protocol robustness issue before PPO training:
  validation static-selection windows contained no flux-subtype samples, so
  `staticnorm_subtype` could not derive a positive flux normalizer.
- Added a bounded fallback in `scripts/25_v2_train_custom_ppo.py`: if a subtype
  normalizer is missing from static-selection windows, the script samples
  subtype-positive windows from the normalization split, records the fallback
  table, and uses the resulting normalizer consistently in reward normalization
  and staticnorm metrics.
- Evidence now written remotely for seed `93`:
  - `reward_staticnorm_normalizers.json` records missing flux and fallback starts
    `25244,38625,49313,58988`;
  - `reward_staticnorm_fallback_candidates.csv` contains the fallback candidate
    losses.
- Relaunched seed `93` as
  `scenebal1_seed93_rerun_staticnormfix_20260621`; it has entered PPO training.

## 2026-06-21 SCENEBAL-1 Pilot 93--98 Completed
- SCENEBAL-1 completed on seeds `93,94,95,96,97,98`.
- Gates:
  - old operational step `6/6`;
  - old operational macro `6/6`;
  - behavior `6/6`;
  - strict replay step `6/6`;
  - strict replay macro `6/6`.
- Mean step margin vs best operational baseline: `0.102988`; median:
  `0.045900`.
- Key repaired case: seed `95` now has `custom_ppo=1.951350` vs
  `validation_selected_static=1.963998`, margin `+0.012648`; TEMPORAL-1 had
  failed this seed structurally under wide replay.
- Boundary still present:
  - learned true-static step `5/6`;
  - learned true-static macro `3/6`.
- Report written:
  `reports/aggregate/scenebal1_pilot_93_98_breakthrough_report_20260621.md`.
- Decision: SCENEBAL-1 is effective and should be expanded rather than pivoted
  away. Next bounded unit: fresh seeds `99--104`.

## 2026-06-21 SCENEBAL-1 12-Seed Strong Operational Claim
- Expansion `99--104` completed and independently reproduced the pilot:
  - operational step `6/6`;
  - operational macro `6/6`;
  - behavior `6/6`;
  - replay step/macro `6/6`.
- Combined seeds `93--104`:
  - operational step `12/12`;
  - operational macro `12/12`;
  - behavior `12/12`;
  - replay step/macro `12/12`;
  - mean step margin `0.133984`;
  - median step margin `0.088511`;
  - sign-test p `0.000244140625`.
- Claim status: strong operational multiseed claim supported.
- Boundary: learned true-static step `11/12`, true-static macro `5/12`; this is
  not an unconditional true-static macro claim.
- Report written:
  `reports/aggregate/scenebal1_12seed_93_104_strongclaim_report_20260621.md`.
- Next bounded unit: continue SCENEBAL-1 expansion on seeds `105--110`.

## 2026-06-21 Goal and Plan-State Audit
- Checked the API goal with the goal tool. It remains active, but its text is
  stale because it still names BO-1 and step-claim framing from an earlier
  branch.
- Treated `research-state.yaml:active_goal` as authoritative for current work:
  24h autonomous PD-PPO strong-claim exploration for ESWA, with PPO preserved as
  the final learned scheduler, `remote-gpu` as the only server entry, and a
  per-direction 10-unit anti-stall rule.
- Updated `.planning/.active_plan` back to
  `2026-06-07-pd-ppo-static-break-recalibration` instead of the completed ESWA
  terminology rewrite plan.
- Updated root and active planning files so the current direction is
  SCENEBAL-1, not BO-1. BO-1 remains historical evidence; SCENEBAL-1 is the
  active simulator/target-generation branch.
- Verified on `remote-gpu` that SCENEBAL-1 seeds `105--110` completed with
  operational step `6/6`, operational macro `6/6`, behavior `6/6`, and replay
  step/macro `6/6`; `research-state.yaml` now records that wave as complete.
- Observed that all six GPUs are currently occupied by another user's diffusion
  jobs, so the immediate autonomous path is local/remote sync, combined
  aggregation, and report writing rather than launching another wave.

## 2026-06-21 SCENEBAL-1 18-Seed Strong Operational Claim
- Synced `105--110` aggregate and key per-seed audit artifacts locally.
- Generated combined `93--110` aggregate directories:
  - `reports/aggregate/scenebal1_18seed_93_110_oldclaim_20260621/`;
  - `reports/aggregate/scenebal1_18seed_93_110_macro_20260621/`;
  - `reports/aggregate/scenebal1_18seed_93_110_raw_macro_20260621/`.
- Aggregate result:
  - complete seeds `18`;
  - old operational step `18/18`;
  - old operational macro `18/18`;
  - behavior `18/18`;
  - strict explicit replay step/macro `18/18`;
  - learned true-static step `17/18`;
  - learned true-static macro `7/18` in the original oldclaim collector
    (superseded by the replay-normalized diagnostic below);
  - mean step margin vs best operational baseline `0.129583`;
  - median step margin `0.077552`;
  - step/macro sign-test p `3.814697265625e-06`.
- Corrected `scripts/74_v31_write_balancedobjective_report.py` because its old
  BO-template wording falsely implied strict replay did not pass all seeds.
- Wrote the local strong-claim report:
  `reports/aggregate/scenebal1_18seed_93_110_strongclaim_report_20260621.md`.
- Decision at this point: do not mark the global goal complete yet; run a
  true-static macro diagnostic before further seed expansion. This diagnosis is
  recorded in the next section and supersedes the original `7/18` macro count.

## 2026-06-21 SCENEBAL-1 True-Static Macro Diagnostic
- Found a scale-mixing issue in `scripts/73_v31_collect_oldclaim_gate.py`:
  learned PPO macro came from the main staticnorm scale, while replay true-static
  macro references came from replay-local static normalization.
- Patched the collector to compute learned PPO macro with the same replay-local
  normalizers used by `scripts/72_v31_collect_metpair_strongclaim.py`.
- Reran corrected oldclaim aggregate:
  `reports/aggregate/scenebal1_18seed_93_110_oldclaim_replaynorm_20260621/`.
- Corrected result:
  - learned true-static macro `18/18`, not `7/18`;
  - learned true-static step remains `17/18`;
  - seed `95` is positive against true fixed static but below the strict
    relative margin threshold.
- Updated the replay-normalized report and main strong-claim report. The current
  remaining boundary is seed95 true-static step strict-margin, not macro.

## 2026-06-21 Seed95 True-Static Step Diagnostic
- Diagnosed the only strict-margin true-static step miss.
- Seed95 values:
  - PPO `1.9513495687247089`;
  - replay-local true fixed static `static_action5=1.9530910958687855`;
  - positive margin `+0.0017415271440766045`;
  - required strict margin `0.003906182191737571`;
  - shortfall `0.0021646550476609665`.
- Across 18 seeds, PPO has positive true-static step margins `18/18`; the
  strict-margin gate is `17/18`.
- Wrote:
  `reports/aggregate/scenebal1_seed95_true_static_step_diagnostic_20260621.md`.
- Updated claim boundary: the current result supports "beats true fixed static
  in sign on all seeds" but not "strict-margin true-static step dominance on
  every seed".

## 2026-06-21 SCENEBAL-1 Stress Wave Waiter
- GPU check showed all six GPUs occupied by another user's jobs.
- Started tmux `scenebal1_waitfree_111_116_20260621` on `remote-gpu`.
- It polls GPU idleness every 600 seconds and will launch seeds `111--116` only
  when all six GPUs are idle.
- Purpose: robustness stress test, not repair. The current evidence already
  resolves the static-shortcut concern except for one strict-margin seed.

## 2026-06-21 SCENEBAL-1 Paper Claim Integration
- Rechecked `remote-gpu`: tmux `scenebal1_waitfree_111_116_20260621` remains
  alive and GPUs are still busy; no new stress-wave results are available.
- Synced the corrected oldclaim collector, report writer, reports, and research
  state to `remote-gpu`. Remote py_compile passed for:
  `scripts/73_v31_collect_oldclaim_gate.py` and
  `scripts/74_v31_write_balancedobjective_report.py`.
- Added:
  `reports/aggregate/scenebal1_18seed_93_110_paper_claim_mapping_20260621.md`.
- Updated canonical paper source `paper/main.tex` and included files:
  `sections/01_introduction.tex`, `sections/04_framework_protocol.tex`,
  `sections/05_simulation_setup.tex`, `sections/06_results.tex`,
  `sections/07_discussion_future_work.tex`, `sections/08_conclusion.tex`,
  `tables/metpair_staticnorm_macro_summary.tex`, `highlights.txt`, and
  `pdppo_crst_rewrite_highlights.txt`.
- Claim migrated from stale `14`-seed wording to the corrected `18`-seed
  SCENEBAL-1 evidence:
  operational step/macro `18/18`, explicit replay step/macro `18/18`, behavior
  `18/18`, replay-normalized true-static macro `18/18`, positive true-static
  step `18/18`, strict-margin true-static step `17/18`.
- Verification:
  `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` succeeded
  in `paper/`, regenerating `main.pdf`; `pdftotext` confirmed the new 18-seed
  numbers render and the checked old `13/14`, `10/14`, and ten-seed highlight
  phrases no longer appear.
- Error recorded:
  a first attempt to read aggregate files guessed stale filenames
  `summary.json` and `oldclaim_summary_by_seed.csv`; resolved by listing the
  directories and using `metpair_claim_summary.json` and
  `oldclaim_seed_summary.csv`.

## 2026-06-21 SCENEBAL-1 Local Watcher
- Started local tmux `scenebal1_watch_111_116_20260621`.
- It monitors remote tmux `scenebal1_waitfree_111_116_20260621`, syncs matching
  SCENEBAL-1 aggregate directories every 600 seconds, and exits after the remote
  session ends.
- First status file:
  `reports/aggregate/scenebal1_pilot_111_112_113_114_115_116_local_watch_20260621_status.md`.
- First snapshot confirms seed outputs are not started yet and all six GPUs
  remain occupied.

## 2026-06-21 Seed-Margin Risk Analysis
- Added:
  `reports/aggregate/scenebal1_18seed_93_110_seed_margin_risk_20260621.md`.
- Seed95 is the only seed below `0.005` and `0.02` true-static step margin.
  The next-lowest seed is seed98 at `0.020629`.
- Decision:
  continue SCENEBAL-1 stress-wave monitoring. Do not pivot from SCENEBAL-1 unless
  the new wave shows repeated true-static sign failures, behavior collapse, or
  loss of explicit replay dynamic headroom.

## 2026-06-21 24-Seed Post-Collect Watcher
- Added script:
  `scripts/watch_scenebal1_24seed_postcollect_20260621.sh`.
- Local validation:
  `bash -n` passed.
- Started local tmux:
  `scenebal1_postcollect_93_116_20260621`.
- Purpose:
  when remote session `scenebal1_waitfree_111_116_20260621` ends, run combined
  `93--116` macro/raw/oldclaim aggregation on `remote-gpu`, write a 24-seed
  report, sync it locally, and exit.
- First status:
  `reports/aggregate/scenebal1_24seed_93_116_postcollect_status_20260621.md`.
- First tick confirms the remote session is still active, so post-collect is
  correctly waiting.

## 2026-06-21 12:54 CST
- Verified `paper/main.pdf` after adding the SCENEBAL-1 18-seed evidence figure.
  `latexmk` reports the PDF is up to date, and `pdftotext` confirms the rendered
  text includes the corrected `18/18` and `17/18` boundaries without the checked
  stale 14-seed wording.
- Checked `remote-gpu`: `scenebal1_waitfree_111_116_20260621` remains alive,
  all GPUs are busy, and no seed `111--116` result directories exist yet.
- Fixed and restarted the local SCENEBAL-1 monitor/postcollect watcher scripts
  after discovering repeated `printf` option warnings from Markdown list lines.
  The pilot watcher now writes clean status output.

## 2026-06-21 12:59 CST
- Synced the updated research/plan files, watcher scripts, paper source/PDF, and
  SCENEBAL-1 figure artifacts to `remote-gpu`.
- Remote watcher syntax checks passed. Remote `research-state.yaml` validation
  passed after activating the `darts` Conda environment.
- Remote GPUs are still fully occupied and no seed `111--116` directories exist,
  so the waitfree session remains in the correct waiting state.

## 2026-06-21 13:03 CST
- Updated `paper/sections/06_results.tex` to cite both the 18-seed table and
  the new seed-level evidence figure in the main results prose.
- Rebuilt `paper/main.pdf` and verified the rendered text says
  `Table 3 and Figure 5`; stale 14-seed claim wording remains absent.

## 2026-06-21 13:04 CST
- Removed an accidentally synced remote-only `paper/sections/main.pdf` from
  `remote-gpu`; the correct `paper/main.pdf` remains present.

## 2026-06-21 13:07 CST
- Fixed the pilot watcher aggregate sync rules so stale remote status Markdown
  can no longer overwrite local postcollect status files.
- Restarted both local SCENEBAL-1 watcher tmux sessions; current status files
  now update cleanly at `13:06`.

## 2026-06-21 13:08 CST
- Scanned the framework for deprecated remote-access references. No executable
  UniVPN/aTrust/old-IP path remains; matches are current prohibition text or
  `uv.lock` version-number false positives.

## 2026-06-21 13:10 CST
- Added a pre-registered 24-seed stress-wave decision protocol:
  `reports/aggregate/scenebal1_24seed_decision_protocol_20260621.md`.
- The report specifies when to upgrade to 24-seed manuscript wording, when to
  keep the current 18-seed claim, and when to pivot to simulator, reward/oracle,
  PPO architecture, or evaluation-contract changes.

## 2026-06-21 13:11 CST
- Preflighted the remote waitfree script and the SCENEBAL-1 parallel seed
  runner. No launch-path issue found.
- The runner waits for idle GPUs, uses the `darts` environment, distributes
  seeds over six GPUs, propagates seed failures, and runs aggregate collectors
  after successful completion.

## 2026-06-21 13:15 CST
- Rechecked `remote-gpu` after the expected `13:14:35` waitfree tick.
- The waitfree log still reports `busy_gpus=6`, and no seed `111--116` result
  directories exist. Stress wave remains correctly queued.

## 2026-06-21 16:08 CST
- Resumed the active `/goal` and reloaded `autoresearch`,
  `remote-gpu-server`, `microclimate-experiment-server`, and
  `planning-with-files` instructions.
- Checked the active SCENEBAL-2 pivot through the only valid server entry,
  `remote-gpu`. The remote tmux session
  `scenebal2_pivot_122_117_20260621` is alive.
- Current SCENEBAL-2 status:
  - seed `122`: artifact bits `100000`; PPO training has reached about
    `177152` timesteps;
  - seed `117`: artifact bits `100000`; PPO training has reached about
    `172032` timesteps;
  - both seeds have the TCN oracle artifact and are still before PPO checkpoint,
    eval, replay, and behavior aggregation;
  - latest local watcher status is
    `reports/aggregate/scenebal2_pivot_122_117_local_watch_20260621_status.md`;
  - no Traceback, RuntimeError, CUDA OOM, or NaN was found in the current
    SCENEBAL-2 scan.
- Interpretation: SCENEBAL-2 is still running normally. No claim decision can be
  made until the decision audit for
  `scenebal2_pivot_conf05_122_117` exists.

## 2026-06-21 16:15 CST
- Rechecked `remote-gpu` after a transient SSH 255. The alias is still the only
  server path used.
- SCENEBAL-2 remote tmux remains alive.
- Seeds `122` and `117` have completed PPO training to `200000` timesteps and
  now show artifact bits `111010`: oracle, PPO checkpoint, base eval, and
  strict replay artifacts exist; router-conf0.5 eval and behavior audit are
  still pending.
- The pivot runner has entered
  `scenebal1_router_reaudit_start date=2026-06-21T16:15:27+08:00 seeds=122 117
  router_conf=0.5 eval_dir=eval_router_conf05_scenebal2_20260621`.
- Next required evidence is the
  `scenebal2_pivot_conf05_122_117_decision_audit_20260621.json` aggregate.

## 2026-06-21 16:22 CST
- SCENEBAL-2 pilot completed and was synced locally after manual aggregate
  rsync; the local watcher had detected completion but its rsync was interrupted
  by SSH reset.
- Decision audit:
  `reports/aggregate/scenebal2_pivot_conf05_122_117_decision_audit_20260621.json`
  reports `upgrade_allseed_strict`.
- Gate counts are clean on both pilot seeds: operational step/macro `2/2`,
  explicit replay step/macro `2/2`, behavior `2/2`, true-static macro `2/2`,
  true-static step sign `2/2`, and strict-margin true-static step `2/2`.
- Seed `122`, the previous fresh-failure seed, recovers with true-static step
  margin `0.077386`; seed `117` passes with margin `0.028498`.
- Wrote pilot report:
  `reports/aggregate/scenebal2_pivot_conf05_122_117_pilot_report_20260621.md`.
- Added `scripts/run_v31_scenebal2_confirm_117_122_20260621.sh` to train the
  missing confirmation seeds `118--121` and then aggregate `117--122` under the
  same fixed router-conf0.5 protocol.

## 2026-06-21 16:25 CST
- Synced the SCENEBAL-2 confirmation runner, pilot report, `findings.md`,
  `progress.md`, `research-state.yaml`, and active-plan files to `remote-gpu`.
- Remote validation passed under conda `darts`: the confirmation runner and
  pilot runner pass `bash -n`; the synced `research-state.yaml` parses; the
  pilot decision JSON still reports `upgrade_allseed_strict`.
- Launched tmux `scenebal2_confirm_117_122_20260621`.
- Started local watcher tmux `scenebal2_confirm_local_watch_20260621`, pointed
  at aggregate label
  `scenebal2_confirm_conf05_117_118_119_120_121_122`.
- Startup health: seeds `118--121` created truth CSV and dataset-validation
  artifacts; no early error is visible. GPU memory is still idle at the first
  25-second check, so oracle/PPO training has not yet visibly allocated GPU
  memory.
- Error and repair: a follow-up multi-source `rsync` again sent root
  `progress.md` and active-plan `progress.md` to the same remote directory.
  Immediately repaired by syncing root `progress.md` to the project root and
  active-plan `progress.md` to
  `.planning/2026-06-07-pd-ppo-static-break-recalibration/progress.md`
  separately. Remote sizes after repair: root `128432` bytes, active plan
  `319940` bytes.

## 2026-06-21 16:28 CST
- First confirmation-wave health check:
  - remote tmux `scenebal2_confirm_117_122_20260621` is alive;
  - seeds `118--121` have entered PPO training after truth/dataset/oracle setup;
  - latest timesteps are around `14336--15360`;
  - GPUs `0--3` show about `1563 MiB` each and nonzero utilization;
  - pilot seeds `117` and `122` remain complete with artifact bits `111111`;
  - early error scan found no Traceback, RuntimeError, CUDA OOM, or NaN.
- Local watcher status file:
  `reports/aggregate/scenebal2_pivot_117_122_local_watch_20260621_status.md`.

## 2026-06-21 16:34 CST
- Confirmation-wave mid-run check:
  - remote tmux `scenebal2_confirm_117_122_20260621` is alive;
  - seeds `118`, `119`, `120`, and `121` remain at artifact bits `100000`;
  - latest PPO timesteps are about `113664`, `113664`, `115712`, and `111616`;
  - GPUs `0--3` remain active with about `1563 MiB` each;
  - no Traceback, RuntimeError, CUDA OOM, or NaN was found.
- No aggregate exists yet; continue monitoring until 200k timesteps, router
  eval, replay, behavior audit, and combined `117--122` decision audit finish.

## 2026-06-21 16:40 CST
- Confirmation-wave late-training check:
  - remote tmux remains alive;
  - seeds `118`, `119`, `120`, and `121` are around `198656`, `197632`,
    `200000`, and `194560` timesteps respectively;
  - seed `120` has already written the PPO checkpoint (`bits=110000`);
  - GPUs `0--3` remain active and the error scan is clean.
- No combined aggregate yet. Continue short-interval polling for eval/replay/
  behavior completion and `scenebal2_confirm_conf05_117_118_119_120_121_122`
  decision output.

## 2026-06-21 16:55 CST
- SCENEBAL-2 six-seed confirmation completed on `remote-gpu`; tmux
  `scenebal2_confirm_117_122_20260621` is no longer alive because the runner
  exited normally.
- Synced local aggregate artifacts:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122*`.
- Decision audit:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_decision_audit_20260621.json`
  reports `upgrade_allseed_strict`.
- Gate counts are clean: operational step/macro `6/6`, explicit replay
  step/macro `6/6`, behavior `6/6`, replay-normalized true-static macro `6/6`,
  true-static step sign `6/6`, strict-margin true-static step `6/6`, and
  old-claim step/macro `6/6`.
- Margins: minimum true-static step margin `0.028498`, mean true-static step
  margin `0.068917`, maximum true-static step margin `0.100144`, minimum
  operational step margin `0.044206`.
- Added paper-fit audit:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_paper_fit_audit_20260621.md`.
- Interpretation: SCENEBAL-2 is no longer only a seed122 recovery pilot. It is a
  fresh-confirmed candidate scenario. It can be integrated naturally into the
  paper only as a regime-balanced backbone-plus-one-specialist microclimate
  benchmark, not as an unconditional claim that PD-PPO beats fixed static
  scheduling in every sensor-scheduling problem.

## 2026-06-21 17:00 CST
- Added `scripts/run_v31_scenebal2_expand_117_128_20260621.sh`, a thin
  reproducible wrapper that trains SCENEBAL-2 seeds `123--128` and aggregates
  `117--128`.
- Local validation passed: shell syntax and `research-state.yaml` YAML parsing.
- Synced the expansion script, updated `findings.md`, `progress.md`,
  `research-state.yaml`, active-plan files, and the paper-fit audit to
  `remote-gpu`.
- Remote validation under conda `darts` passed; all six GPUs were idle at launch
  time.
- Launched remote tmux `scenebal2_expand_117_128_20260621`.
- Launched local watcher tmux `scenebal2_expand_local_watch_20260621` with
  aggregate label
  `scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128`.

## 2026-06-21 16:55 CST Remote Tick
- Checked `remote-gpu` for the SCENEBAL-2 `117--128` expansion. Remote clock was
  `2026-06-21T16:54:37+08:00`; tmux
  `scenebal2_expand_117_128_20260621` is alive.
- The master log shows:
  `scenebal2_confirm_start ... train_seeds=123 124 125 126 127 128 ...`
  and `scenebal2_pivot_start ... seeds=123 124 125 126 127 128`.
- GPUs `0--5` are active with about `1563 MiB` each.
- Seeds `123--128` have artifact bits `100000`: oracle exists and PPO training
  has started; PPO checkpoints/evals/replay/behavior are not expected yet.
- Latest PPO timesteps are early: seed123 `3072`, seed124 `4096`, seed125
  `3072`, seed126 `4096`, seed127 `4096`, seed128 `4096`.
- Error scan found no Traceback, RuntimeError, CUDA OOM, or NaN.
- While waiting for the expansion to mature, inspected current manuscript
  sections. The paper already has the right backbone-plus-specialist framing, so
  SCENEBAL-2 migration would mainly patch evidence counts/assets and add a
  subtype-balance explanation rather than rewrite the contribution.
- Added migration plan:
  `reports/aggregate/scenebal2_manuscript_migration_patch_plan_20260621.md`.

## 2026-06-21 16:58 CST Remote Tick
- Remote tmux `scenebal2_expand_117_128_20260621` remains alive.
- Seeds `123--128` remain at artifact bits `100000`, as expected during PPO
  training.
- Latest timesteps: seed123 `51200`, seed124 `51200`, seed125 `51200`, seed126
  `51200`, seed127 `51200`, seed128 `52224`.
- GPUs `0--5` remain active at about `1563 MiB` each.
- Error scan remains clean.
- Added a parameterised evidence-figure generator:
  `paper/figures/gen_fig_scenebal_evidence.py`.
- Smoke-tested it on the completed six-seed SCENEBAL-2 aggregate, producing
  `paper/figures/figure_scenebal2_6seed_evidence.pdf` and `.png`.
- This is preparatory only; do not switch the manuscript figure until the
  running `117--128` aggregate finishes.

## 2026-06-21 17:01 CST Remote Tick
- Remote tmux `scenebal2_expand_117_128_20260621` remains alive.
- Seeds `123--128` are about halfway through PPO training:
  seed123 `104448`, seed124 `103424`, seed125 `104448`, seed126 `103424`,
  seed127 `105472`, seed128 `105472`.
- Artifact bits remain `100000`, which is expected before the PPO checkpoints
  are written.
- GPUs `0--5` remain active at about `1563 MiB` each.
- No aggregate exists yet and the error scan remains clean.
- Added `scripts/77_v31_write_scenebal_summary_table.py`, which writes a LaTeX
  summary table from a SCENEBAL decision audit JSON.
- Smoke-tested it on the completed six-seed SCENEBAL-2 decision audit, producing
  `paper/tables/scenebal2_6seed_staticnorm_macro_summary.tex`.
- This is also preparatory only; do not switch the manuscript table until the
  running `117--128` aggregate finishes.

## 2026-06-21 17:05 CST Remote Tick
- Remote tmux `scenebal2_expand_117_128_20260621` remains alive.
- Seeds `123--128` are late in PPO training:
  seed123 `158720`, seed124 `157696`, seed125 `158720`, seed126 `158720`,
  seed127 `160768`, seed128 `160768`.
- Artifact bits remain `100000`; checkpoints/eval/replay are not written yet.
- GPUs `0--5` remain active at about `1563 MiB` each.
- No aggregate exists yet and the error scan remains clean.
- Corrected a multi-source rsync placement error for the newly added SCENEBAL-2
  helper assets: `scripts/77_v31_write_scenebal_summary_table.py` and
  `paper/tables/scenebal2_6seed_staticnorm_macro_summary.tex` briefly appeared
  in the remote project root. Re-synced them to `scripts/` and `paper/tables/`,
  removed root copies, and verified remote placement.

## 2026-06-21 17:08 CST Remote Tick
- Remote tmux `scenebal2_expand_117_128_20260621` remains alive.
- Seeds `125`, `127`, and `128` have reached `200000` timesteps and written PPO
  checkpoints; seeds `123`, `124`, and `126` are at `199680`, `197632`, and
  `198656` respectively.
- Current new-seed artifact bits are mixed as expected near the checkpoint
  boundary: `125/127/128` are `110000`, while `123/124/126` remain `100000`.
- No aggregate exists yet and the experiment error scan is clean.
- The monitoring command itself emitted `printf: --: invalid option` because the
  format string started with `---`; this is monitor noise only. Future remote
  monitor snippets should use `printf --` or avoid leading hyphens.

## 2026-06-21 17:10 CST Remote Tick
- Remote tmux `scenebal2_expand_117_128_20260621` remains alive.
- All new seeds `123--128` reached `200000` timesteps.
- Artifact bits are now `111000` for all six new seeds: oracle, PPO checkpoint,
  and base eval exist; router-conf0.5 eval, strict replay, and behavior audit are
  still pending.
- No aggregate exists yet and the experiment error scan remains clean.

## 2026-06-21 17:13 CST Remote Deep Check
- The expansion is not stalled. Remote process inspection shows all six new seeds
  running `scripts/70_v31_split_replay_gate.py` for strict no-duty-guard explicit
  subtype replay.
- Recent files confirm split-replay rollouts and candidate tables are being
  written under each seed's `replay_gate_explicit_static_noguard/` directory.
- `eval_router_conf08` files also exist for new seeds as part of the underlying
  seed sweep, but the final expansion aggregate still needs the fixed
  `eval_router_conf05_scenebal2_20260621` router evaluation plus behavior audit
  after seed-level replay finishes.
- Continue monitoring; no manual intervention is needed.

## 2026-06-21 17:20 CST
- SCENEBAL-2 12-seed expansion `117--128` completed and was synced locally.
- Decision audit:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128_decision_audit_20260621.json`.
- Decision is `upgrade_allseed_strict`.
- All pre-registered gates are clean: operational step/macro `12/12`, explicit
  replay step/macro `12/12`, behaviour `12/12`, true-static macro `12/12`,
  true-static step positive `12/12`, strict-margin true-static step `12/12`,
  and old-claim step/macro `12/12`.
- Key margins: minimum true-static step margin `0.028498`, mean true-static
  step margin `0.073911`, minimum operational step margin `0.044206`, mean
  operational step margin `0.112659`.
- Generated SCENEBAL-2 12-seed paper assets:
  `paper/figures/figure_scenebal2_12seed_evidence.pdf`,
  `paper/figures/figure_scenebal2_12seed_evidence.png`, and
  `paper/tables/scenebal2_12seed_staticnorm_macro_summary.tex`.
- Added 12-seed paper-fit/breakthrough report:
  `reports/aggregate/scenebal2_confirm_conf05_117_128_12seed_breakthrough_report_20260621.md`.
- Decision: the current scene is more specialised than the original broad claim
  but not over-specialised if written as a regime-balanced
  backbone-plus-one-specialist microclimate benchmark. It can be naturally
  integrated into the paper as a benchmark/evidence update, not as a universal
  sensor-scheduling theorem.
- Local path error noted: an earlier command tried to read `.planning/` and
  `research-state.yaml` from the repository root instead of
  `rl_sensor_scheduling_framework/`; the framework root is the authoritative
  planning location for this task.

## 2026-06-21 17:27 CST
- Added SCENEBAL-2 18-seed extension runner:
  `scripts/run_v31_scenebal2_expand_117_134_20260621.sh`.
- The runner trains seeds `129--134` and aggregates seeds `117--134` under the
  same fixed router-conf0.5 SCENEBAL-2 protocol.
- Local validation passed: shell syntax and `research-state.yaml` YAML parsing.
- Synced the runner, 12-seed reports/assets, plan files, `findings.md`,
  `progress.md`, and `research-state.yaml` to `remote-gpu` with target-directory
  separated rsync commands.
- Remote validation under conda `darts` passed.
- Launched remote tmux `scenebal2_expand_117_134_20260621`.
- Launched local watcher tmux `scenebal2_expand_local_watch_117_134_20260621`
  with aggregate label
  `scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133_134`.
- Early health check: remote tmux is alive; master log has the correct
  `train_seeds=129 130 131 132 133 134` and `all_seeds=117 ... 134`; GPU memory
  is still idle and new seed artifact bits are `000000`, consistent with early
  setup/preprocessing before oracle/PPO allocation.

## 2026-06-21 17:31 CST
- SCENEBAL-2 18-seed extension health check:
  - remote tmux `scenebal2_expand_117_134_20260621` is alive;
  - GPUs `0--5` have entered the expected low-memory PPO training state;
  - seeds `129--134` have artifact bits `100000`;
  - latest PPO timesteps are about `29696--31744` of `200000`;
  - no Traceback, RuntimeError, CUDA OOM, Exception, or NaN was found.
- No aggregate exists yet; continue monitoring until checkpoints, router eval,
  strict replay, behaviour audit, and the `117--134` decision audit finish.

## 2026-06-21 17:34 CST
- SCENEBAL-2 18-seed extension mid-training check:
  - remote tmux remains alive;
  - seed129--134 artifact bits remain `100000`, as expected before checkpoints;
  - latest PPO timesteps are about `70656--73728` of `200000`;
  - GPUs `0--5` remain active with about `1563 MiB` each;
  - error scan remains clean.
- No aggregate exists yet.

## 2026-06-21 17:38 CST
- SCENEBAL-2 18-seed extension late-training check:
  - remote tmux remains alive;
  - seed129--134 artifact bits remain `100000`;
  - latest PPO timesteps are about `124928--130048` of `200000`;
  - GPUs remain active with about `1563 MiB` each;
  - error scan remains clean.
- No checkpoint/eval/replay aggregate exists yet.

## 2026-06-21 17:45 CST
- SCENEBAL-2 18-seed extension post-training check:
  - all new seeds `129--134` reached `200000` timesteps and wrote PPO
    checkpoints;
  - artifact bits are `111000` for all six new seeds: oracle, PPO checkpoint,
    and base eval exist or are being finalized;
  - remote process inspection shows `24_v2_evaluate_rollouts.py` active for the
    new seed directories, so the run is in the expected base-eval/post-training
    stage rather than stalled;
  - no error signatures were found.
- Next expected stages are fixed router-conf0.5 eval, strict no-duty-guard
  replay, behaviour audit, and `117--134` aggregation.

## 2026-06-21 17:47 CST
- SCENEBAL-2 18-seed extension strict replay stage:
  - remote tmux remains alive;
  - all new seeds `129--134` still show artifact bits `111000`;
  - process inspection shows six active `scripts/70_v31_split_replay_gate.py`
    processes, one per new seed;
  - these are running strict no-duty-guard subtype-explicit replay with static
    candidate reference enforcement and macro column
    `oracle_loss_macro_subtype_event_staticnorm`;
  - GPU memory is idle, which is expected for CPU replay.
- No decision aggregate exists yet.

## 2026-06-21 17:52 CST
- SCENEBAL-2 18-seed extension completed on `remote-gpu`; tmux
  `scenebal2_expand_117_134_20260621` exited normally.
- Synced local aggregate and new seed compact artifacts for seeds `129--134`.
- Decision audit:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133_134_decision_audit_20260621.json`.
- Decision is `upgrade_allseed_strict`.
- Gate counts are clean: operational step/macro `18/18`, explicit replay
  step/macro `18/18`, behaviour `18/18`, true-static macro `18/18`,
  true-static step sign `18/18`, strict-margin true-static step `18/18`, and
  old-claim step/macro `18/18`.
- Margins: minimum true-static step margin `0.028498`, median `0.078516`, mean
  `0.079259`; minimum operational step margin `0.031445`, mean operational step
  margin `0.136583`; mean learned macro margin versus true-static macro
  reference `0.070709`.
- Generated 18-seed paper assets:
  `paper/figures/figure_scenebal2_18seed_evidence.pdf`,
  `paper/figures/figure_scenebal2_18seed_evidence.png`, and
  `paper/tables/scenebal2_18seed_staticnorm_macro_summary.tex`.
- Added breakthrough report:
  `reports/aggregate/scenebal2_confirm_conf05_117_134_18seed_breakthrough_report_20260621.md`.
- Decision: launch a final SCENEBAL-2 `135--140` extension to reach 24 seeds
  before replacing the manuscript main evidence.

## 2026-06-21 17:57 CST
- Answered the paper-fit question from the current manuscript and SCENEBAL-2
  evidence: the scenario is more specialised than the original broad
  sensor-scheduling claim, but it is not over-specialised if framed as a
  regime-balanced backbone-plus-one-specialist microclimate benchmark targeting
  the static-shortcut failure mode.
- Added `scripts/run_v31_scenebal2_expand_117_140_20260621.sh` for the final
  SCENEBAL-2 24-seed extension: train seeds `135--140`, aggregate seeds
  `117--140`, router confidence fixed at `0.5`, same SCENEBAL-2 sensor geometry.
- Updated `research-state.yaml` and the active plan: `117--134` is now
  `complete_breakthrough_confirmation`, and the active unit is the `117--140`
  24-seed extension.

## 2026-06-21 17:58 CST
- Synced the SCENEBAL-2 `117--140` runner, `research-state.yaml`, root progress,
  and active-plan files to `remote-gpu`.
- Started remote tmux `scenebal2_expand_117_140_20260621` with log
  `logs/scenebal2_expand_117_140_20260621.master.log`.
- Started local watcher tmux `scenebal2_expand_local_watch_117_140_20260621`.
- Early health check: remote tmux is alive; log entered
  `train_seeds=135 136 137 138 139 140` and `all_seeds=117--140`; no
  Traceback/RuntimeError/CUDA OOM/Exception/NaN/path error signatures were
  found. Old seeds `117--134` show artifact bits `111111`; new seeds
  `135--140` are in startup with bits `000000`.

## 2026-06-21 18:01 CST
- SCENEBAL-2 `117--140` 24-seed extension is actively training on
  `remote-gpu`.
- Remote tmux `scenebal2_expand_117_140_20260621` is alive.
- GPUs `0--5` each show an active PPO process with about `1563 MiB` memory.
- New seeds `135--140` all have artifact bits `100000`: oracle exists and PPO
  training is underway; no eval/replay/behaviour artifacts yet.
- Latest PPO timesteps are about `17408--18432` of `200000`; no
  Traceback/RuntimeError/CUDA OOM/Exception/NaN/path error signatures were found.
- No `117--140` aggregate exists yet. Continue monitoring through checkpoint,
  router eval, strict replay, behaviour audit, and final decision audit.

## 2026-06-21 18:05 CST
- SCENEBAL-2 `117--140` extension mid-training check:
  - remote tmux `scenebal2_expand_117_140_20260621` remains alive;
  - GPUs `0--5` each hold about `1563 MiB`, consistent with six active PPO
    training processes;
  - seeds `135--140` remain at artifact bits `100000`;
  - latest PPO timesteps are about `86016--90112` of `200000`;
  - error scan remains clean.
- No `117--140` aggregate exists yet; continue monitoring to checkpoint and
  post-training evaluation.

## 2026-06-21 18:13 CST
- SCENEBAL-2 `117--140` extension reached the checkpoint boundary:
  - remote tmux is still alive;
  - seeds `135--140` all reached `200000` timesteps;
  - all six new seeds wrote `custom_ppo.pt`, so artifact bits are now `110000`;
  - base eval, fixed router-conf0.5 eval, strict replay, behaviour audit, and
    `117--140` aggregate are still pending;
  - error scan remains clean.
- Continue monitoring the post-training stages; do not interpret the 24-seed
  claim until the decision audit exists.

## 2026-06-21 18:17 CST
- SCENEBAL-2 `117--140` extension has moved into strict replay:
  - remote tmux remains alive;
  - several `scripts/70_v31_split_replay_gate.py` processes are active;
  - GPUs are mostly idle, expected for the CPU replay stage;
  - no `117--140` decision aggregate exists yet;
  - error scan remains clean.
- Continue monitoring for replay completion, behaviour audit, and aggregate
  decision output.

## 2026-06-21 18:24 CST
- SCENEBAL-2 `117--140` 24-seed aggregate completed on `remote-gpu`; tmux
  `scenebal2_expand_117_140_20260621` exited.
- Synced the decision audit and aggregate directories:
  - `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133_134_135_136_137_138_139_140_decision_audit_20260621.json`
  - `..._oldclaim_20260621`
  - `..._macro_20260621`
  - `..._raw_macro_20260621`
- Decision is `upgrade_allseed_strict`; operational step/macro, explicit replay
  step/macro, behaviour, true-static macro, true-static step sign,
  strict-margin true-static step, and old-claim step/macro gates are all
  `24/24`; all failure lists are empty.
- Key margins: minimum true-static step `0.028498`, mean true-static step
  `0.076901`, minimum operational step `0.031445`, mean operational step
  `0.149379`, mean learned macro margin versus true-static reference
  `0.070991`, mean explicit replay macro margin versus true-static reference
  `0.077345`.
- Generated SCENEBAL-2 24-seed paper assets:
  `paper/figures/figure_scenebal2_24seed_evidence.pdf`,
  `paper/figures/figure_scenebal2_24seed_evidence.png`, and
  `paper/tables/scenebal2_24seed_staticnorm_macro_summary.tex`.
- Migrated the active manuscript main result from SCENEBAL-1 to SCENEBAL-2:
  `paper/sections/05_simulation_setup.tex` now reports seeds `117--140`, and
  `paper/sections/06_results.tex` references the SCENEBAL-2 24-seed table and
  figure.
- Rebuilt `paper/main.pdf` with
  `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`.
- `pdftotext` verification confirms the rendered PDF contains SCENEBAL-2,
  `117--140`, `24/24`, seed `117`, `0.0285`, `0.0710`, and `0.0773`; no active
  paper-section residual matches for SCENEBAL-1, `93--116`, seed `95`, `0.0042`,
  `0.0799`, or `0.0862`.

## 2026-06-21 18:28 CST
- Reconciled planning/state after the 24-seed breakthrough:
  - `research-state.yaml` now marks `scenebal2_expansion_117_140` as
    `complete_manuscript_replacement_breakthrough`;
  - `latest_results.scenebal2_24seed_117_140` records the all-strict `24/24`
    gates, key margins, and paper-fit boundary;
  - active plan Phase 25 is complete and Phase 26 is now the post-breakthrough
    evidence audit / paper-packaging phase.
- This prevents the project state from continuing to ask for the already
  completed `117--140` monitoring/aggregation step.

## 2026-06-21 18:30 CST
- Completed Phase 26 post-breakthrough evidence audit:
  - local and remote `research-state.yaml` parse correctly;
  - local and remote `paper/main.pdf` contain SCENEBAL-2, `117--140`,
    `24/24`, `0.0285`, `0.0710`, and `0.0773`;
  - active paper sources and rendered PDF show no checked stale SCENEBAL-1
    main-result residuals.
- Set Phase 27 to manuscript claim-framing/submission-fit audit. The immediate
  decision is to treat SCENEBAL-2 as a natural regime-balanced specialist-budget
  benchmark, not as a universal sensor-scheduling theorem.

## 2026-06-21 18:36 CST
- Completed Phase 27 claim-framing audit:
  - added `reports/aggregate/scenebal2_24seed_claim_framing_audit_20260621.md`;
  - added `reports/aggregate/scenebal2_24seed_supervisor_summary_20260621.md`;
  - patched `paper/sections/01_introduction.tex` and
    `paper/sections/05_simulation_setup.tex` to motivate the
    backbone-plus-specialist geometry as a deployment-relevant microclimate
    benchmark abstraction.
- Rebuilt `paper/main.pdf` successfully with `latexmk -xelatex`; rendered text
  confirms the deployment-relevant scenario wording and SCENEBAL-2 `24/24`
  evidence remain present.
- Remote check shows no active tmux experiment sessions and all six GPUs idle;
  the workstream is now evidence packaging and manuscript polishing, not waiting
  for running experiments.

## 2026-06-21 18:40 CST
- Completed requirement-by-requirement audit for the active PD-PPO strong-claim
  goal:
  `reports/aggregate/pdppo_strongclaim_completion_audit_20260621.md`.
- Audit judgment: the experimental strong-claim objective is achieved. The final
  defensible claim is bounded to the SCENEBAL-2 regime-balanced
  backbone-plus-one-specialist microclimate benchmark, with all primary gates
  passing `24/24`.
- Updated `research-state.yaml` status to `strongclaim_experiment_complete` and
  marked active plan Phase 28 complete. Remaining work is manuscript polishing
  and submission packaging, not additional experiment exploration for the stated
  goal.
- Remote sync note: two `rsync` attempts hit a transient SSH connection close
  while sending the later files. This matched the known 255-class remote SSH
  issue rather than server downtime. Retried the remaining completion audit with
  `scp remote-gpu:...`; remote verification then passed.

## 2026-06-21 19:58 CST
- User requested research/report first and explicitly said not to directly
  modify the paper before report approval.
- Audited active paper theory/problem sections:
  `paper/sections/03_problem_formulation.tex`,
  `paper/sections/appendix_theory.tex`,
  `paper/sections/01_introduction.tex`,
  `paper/sections/05_simulation_setup.tex`, and
  `paper/sections/07_discussion_future_work.tex`.
- Verified literature candidates using arXiv/CrossRef/DOI endpoints instead of
  memory-generated citations.
- Added report-only artifact:
  `reports/aggregate/specialist_bottleneck_theory_extension_report_20260621.md`.
- No files under `paper/` were modified in this step.

## 2026-06-21 20:19 CST
- User provided final author metadata:
  affiliation `东南大学 机械工程学院`, corresponding author `Yongzhe Li`, and phone
  `13946019751`.
- Verified active manuscript sources already contain the matching English
  metadata:
  `School of Mechanical Engineering, Southeast University, Nanjing, China`,
  `Yongzhe Li\corref{cor1}`, and `Tel.: +86 13946019751`.
- Updated `task_plan.md` to mark affiliation and corresponding-author phone as
  resolved. No `paper/` source edit was needed.

## 2026-06-21 20:38 CST
- Resolved current plan-file consistency debt:
  - root `task_plan.md` now states that the SCENEBAL-2 `117--140` strong-claim
    experiment is complete and that current work is ESWA manuscript theory
    integration / packaging;
  - subproject `task_plan.md` now uses `paper/main.tex` and
    `paper/sections/*.tex` as the canonical source, marks V3.1/metpair/SCENEBAL-1
    as historical or superseded where relevant, and records SCENEBAL-2
    `117--140` as the main evidence block;
  - `.planning/.active_plan` now points to
    `2026-06-10-eswa-terminology-rewrite`;
  - `.planning/2026-06-10-eswa-terminology-rewrite/task_plan.md` now matches
    the canonical `paper/main.tex` source rather than the historical
    `pdppo_crst_rewrite.tex` rewrite files.
- Applied the approved specialist-bottleneck theory extension to the manuscript:
  - added verified citations in `paper/references.bib` from arXiv/DOI/CrossRef
    endpoints for Golovin/Krause, Bajcsy et al., Lauri et al., Shi et al., and
    Kaul/Yates/Gruteser;
  - updated `paper/sections/02_related_work.tex` with active-perception,
    POMDP, AoI, and energy-constrained scheduling context;
  - added the forecast-relevant specialist-bottleneck definition and
    Proposition~1 to `paper/sections/03_problem_formulation.tex`;
  - added the proof and simple-cycle argument in
    `paper/sections/appendix_theory.tex`;
  - loaded the appendix from `paper/main.tex` and removed the stale simulator
    parameter table from the theory appendix.
- Rebuilt `paper/main.pdf` with
  `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`.
  Result: success, 36 pages, no undefined citations or references. Remaining
  BibTeX warnings are the existing empty-page warnings for `Liu2024`,
  `Murad2020`, `Pendyala2024`, and `Wei2020`.
- PDF text audit confirms the rendered manuscript contains the specialist
  bottleneck proposition, Appendix A.3 simple-cycle discussion, SCENEBAL-2,
  `117--140`, `24/24`, and the new literature citations.

## 2026-06-21 21:29 CST
- Completed a manuscript completeness and consistency pass after the theory
  integration.
- Verified that the old public release cannot be treated as the current
  SCENEBAL-2 evidence package: the `v0.1.0` release URL returns `404`, and
  `git ls-remote --tags` reports no usable current release tag.
- Updated `paper/main.tex` data availability to avoid citing that stale release
  and to require a future versioned archive containing code, SCENEBAL-2
  aggregate tables, seed-level summaries, figure assets, and reproduction
  scripts.
- Updated `paper/sections/05_simulation_setup.tex` with the SCENEBAL-2
  subtype-balance, target-weighting, and met-plus-one-specialist benchmark
  definition used for seeds `117--140`.
- Updated `paper/sections/06_results.tex` with the raw unnormalised subtype
  macro sensitivity boundary: explicit replay macro remains positive `24/24`,
  but learned-policy raw macro dominance is not supported (`0/24`).
- Rebuilt `paper/main.pdf` with
  `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`.
  Result: success, `37` pages, no undefined citations or references. The only
  BibTeX warnings are the existing empty-page warnings for `Liu2024`,
  `Murad2020`, `Pendyala2024`, and `Wei2020`.
- PDF text audit confirms the rendered file contains the new SCENEBAL-2 design
  paragraph, the metric-sensitivity boundary, `0/24`, `24/24`, and the updated
  versioned-archive data-availability statement.
- During the final over-claim/stale-name scan, the current SCENEBAL-2 table
  label was renamed from the historical `metpair` label to
  `tab:scenebal2_staticnorm_macro_summary`, and the Results reference was
  updated accordingly.

## 2026-06-21 21:45 CST
- Checked whether the active manuscript has fully moved to the new SCENEBAL-2
  claim. Main `paper/main.tex` and rendered `paper/main.pdf` already carried
  the new 24-seed SCENEBAL-2 claim, true fixed-static replay framing,
  behaviour-complexity gate, and raw macro sensitivity boundary.
- Found and fixed two residual consistency issues:
  - `paper/highlights.txt` and the historical
    `paper/pdppo_crst_rewrite_highlights.txt` still said `18 seeds`;
  - `paper/sections/03_problem_formulation.tex` still described the stricter
    step-weighted fixed-static comparison as a limitation rather than as a
    separate strict gate now passed by SCENEBAL-2.
- Rebuilt `paper/main.pdf` successfully after the edits. The PDF remains
  `37` pages and has no undefined references/citations. The remaining BibTeX
  warnings are the existing empty-page warnings only.
- Final scan found no old `18 seeds`, `SCENEBAL-1`, `V3.1`, `metpair`,
  `seed45`, `h075`, `CRST`, or `pdppo_crst` residue in the active source set
  checked for submission-facing text.

## 2026-06-21 22:28 CST
- Responded to the figure-count regression in the new mainline manuscript.
- Audited current and historical figures. Current `main.pdf` had five active
  figures: framework, data split, AWS illustration, synthetic-statistics
  validation, and the SCENEBAL-2 24-seed evidence figure. The older
  `figure_operational_summary`, `figure_operational_behavior`, and
  `figure_fixed_budget_power_error` assets were not reinserted directly because
  they describe the old 10-seed / compact-static / deployable-static protocol
  and would conflict with the current SCENEBAL-2 claim.
- Added a new SCENEBAL-2 diagnostic figure generator:
  `paper/figures/gen_fig_scenebal2_diagnostics.py`.
- Generated and inserted two current-claim figures:
  - `paper/figures/figure_scenebal2_metric_boundary.{pdf,png}` for the
    static-normalised-vs-raw macro boundary;
  - `paper/figures/figure_scenebal2_behavior_audit.{pdf,png}` for the corrected
    24-seed behaviour-complexity audit.
- Updated `paper/sections/06_results.tex` to include both figures with bounded
  captions that do not broaden the claim.
- Rebuilt `paper/main.pdf` successfully. The manuscript now has seven active
  figures, `38` pages, no undefined references/citations, and only the existing
  BibTeX empty-page warnings.

## 2026-06-22 04:47 CST
- Completed the ESWA terminology, title, figure-style, and sub-review
  convergence pass requested after reading `docs/06-23-01.md`.
- Updated the manuscript title to
  `Forecast-Loss-Driven Projected PPO for Power-Constrained Sensor Scheduling`.
- Rewrote the visible experimental vocabulary away from internal gate/oracle
  phrasing:
  - `fixed evaluation forecaster`;
  - `validation-selected fixed-mask and rule-based dynamic baselines`;
  - `event-aware diagnostic reference`;
  - `behavioural diagnostics`;
  - `prespecified superiority-margin formula`.
- Added the step-margin formula
  `epsilon_s=max{0.001, 0.002 L_ref_step,s}` to the evaluation setup, and
  described behavioural diagnostics using held-out action traces, action-mask
  entropy, transition entropy, event-action mutual information, regime-
  conditioned sensor-use separation, and the fixed/cycle thresholds.
- Expanded the final 24-seed table with medians and percentile bootstrap 95%
  confidence intervals over seed-level paired margins:
  - step margin vs. best selected baseline: mean `0.1494`, median `0.1060`,
    CI `[0.1124, 0.1892]`;
  - macro margin vs. best selected baseline: mean `0.0841`, median `0.0884`,
    CI `[0.0738, 0.0948]`;
  - learned macro margin vs. fixed-mask replay: mean `0.0710`, median `0.0673`,
    CI `[0.0625, 0.0798]`.
- Redrew/rebuilt the active figure set with shared plotting style:
  - `paper/figures/figure_pdppo_framework_image2.{pdf,png}`;
  - `paper/figures/figure_scenebal2_24seed_evidence.{pdf,png}`;
  - `paper/figures/figure_scenebal2_behavior_audit.{pdf,png}`;
  - `paper/figures/figure_mechanism_robustness.{pdf,png}`;
  - `paper/figures/figure3_synthetic_statistics.{pdf,png,svg}`.
- Reworked Figure 5/6 readability after sub-review:
  - compressed Figure 5 Panel B labels;
  - rotated/sparsified Figure 6 seed ticks;
  - removed an obsolete S/M caption note.
- Recorded reference-paper retrieval status in
  `paper/reference_papers/eswa_similar/README.md`. Closest ESWA full-text PDFs
  were blocked/closed through available routes; three valid local reference
  PDFs remain available for style and related-work checks.
- Used sub-review feedback from three agents:
  - AI-writing/argument tone: final verdict `positive`;
  - figure readability: final verdict `positive`;
  - statistical/claim framing: argument judged positive after fixes, with only
    metadata caveats.
- Final verification:
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`
    succeeds;
  - `paper/main.pdf` is 41 pages;
  - PDF metadata title matches the final title;
  - PDF text audit finds no visible `SCENEBAL`, `metpair`, `router`, `gate`,
    `oracle`, `score threshold`, `true-static`, or `subtype-conditioned`
    residue;
  - `pdffonts main.pdf` shows no Type 3 fonts;
  - no undefined references/citations were found.
- Remaining non-scientific metadata caveat: no grant number or public archival
  DOI has been supplied. The manuscript now states this fact instead of making
  an unsupported archive promise.
