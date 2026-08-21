# Task Plan: PD-PPO Paper Closure and ESWA Evidence Alignment

> Current authoritative plan: `.planning/2026-06-10-eswa-terminology-rewrite/`.
> This root-level file is a historical snapshot retained for continuity; the
> 2026-06-23-02 PPO-LEMMA closure status is recorded in the active `.planning`
> directory.

## Goal
Bring the PD-PPO manuscript and experiment package to a coherent submission-ready state for *Expert Systems with Applications* (ESWA): V3.1 S2 and later corrected evidence are treated according to their protocol validity, V3.1-aligned A1/A2/H1 ablations are integrated as diagnostics where still relevant, missing reproducibility/hardware details are filled, and all paper claims are conservative, traceable, internally consistent, and compile-clean.

## Current Phase
Phase 13 (ESWA Terminology, Title, and Figure-Style Refinement)

## Operating Rules
- Use `paper/main.tex` and `paper/sections/*.tex` as the formal manuscript
  source; `paper/raw.tex` is not authoritative. Historical rewrite files such
  as `paper/pdppo_crst_rewrite.tex` are retained as provenance only.
- Do not expose internal benchmark codenames or engineering labels in the
  manuscript. Use reader-facing terms such as one-specialist benchmark,
  fixed-mask baseline, rule-based dynamic baseline, fixed evaluation forecaster,
  event-aware diagnostic reference, and behavioural diagnostics.
- Use SCENEBAL-2 seeds `117--140` as the current main performance evidence.
  V3.1 S2 and earlier metpair evidence are retained as historical or diagnostic
  context unless explicitly reintroduced.
- Treat V3.1-aligned A1/A2/H1 as the current ablation diagnostics; keep V2 ablations only as historical/development context if explicitly needed.
- Do not claim uniform statistical significance against round-robin; current V3.1 S2 supports mean improvement over round-robin and Bonferroni-significant gains over AoI/random.
- Do not imply ActionEmbedding, EventAwareCritic, or action mask individually have statistically proven performance gains; A1 only supports the AWBC + oracle-prior stabilisation pair and full stack versus MaskedActor-only.
- Do not let the normalized scheduling cost table contradict physical watt/current text. If normalized cost is not absolute wattage, state the calibration logic explicitly and avoid claiming it preserves a false wattage ranking.
- Long experiments run only on the GPU server in tmux; sync partial results back before risky interruptions.
- Remote execution uses only the `remote-gpu` SSH alias. Do not use hardcoded
  host addresses, historical internal network paths, old tunnel/client scripts,
  or password-based sync helpers.
- Static-break recalibration is no longer limited to scenario parameter edits.
  If pure scene tuning cannot produce a stable learned policy, framework-level
  changes are allowed: new policy/critic layers, richer simulator regimes,
  modified data generation, temporal abstractions, memory/belief modules, and
  stricter anti-shortcut baselines. The final requirement is not "a plausible
  dynamic schedule" but an RL framework that is forecast-optimal under the tested
  protocol and whose learned behaviour is genuinely state-dependent rather than
  fixed sensors or a simple two/three-combination cycle.
- The 2026-06-21 SCENEBAL-2 strong-claim experiment is complete for the stated
  objective. Any future strong-claim exploration keeps the hard anti-stall rule:
  after `10` complete hypothesis rounds without a new breakthrough, stop
  conservative retries and pivot to a deeper experiment/framework/algorithm
  change. A round means a named branch with a complete seed set or pilot,
  aggregate evidence, and a logged keep/pivot decision. `BO-1` and
  `SCENEBAL-1` are superseded by `SCENEBAL-2`, which now provides the active
  24-seed evidence block.
- The anti-stall rule applies per modification direction, not only globally.
  If a scene/simulator/teacher/framework direction shows no effective
  improvement after at most `10` complete work units, or earlier evidence marks
  it as failed or likely failed, abandon that direction and move to another
  layer of change. A work unit may be a full seed wave, a locked pilot, a
  completed diagnostic branch, or another bounded experiment unit with a
  written keep/pivot decision.
- PPO must remain the learned scheduling algorithm being validated. Deeper
  modifications may change PPO inputs, auxiliary heads, policy architecture,
  reward shaping, teacher construction, replay/evaluation protocols, simulator
  dynamics, or sensor/noise calibration, but they must not replace PPO with a
  different final scheduler.
- The current sensor configuration should be preserved as the baseline physical
  design. Moderate sensor-config changes are allowed because this is a simulated
  scenario, but they must remain explainable as variants of the existing
  microclimate sensing setup rather than an arbitrary new sensor system.

## Phases

### Phase 1: Consolidate Existing State
- [x] Read the historical TODO in `docs/05-13/00_TODO_goal_required_experiments_and_paper_fixes.md`.
- [x] Read V3.1 rerun plan and S2 completion report.
- [x] Record the local V3.1-aligned ablation snapshot after rsync.
- [x] Separate blocking manuscript tasks from optional experiment upgrades.
- **Status:** complete

### Phase 2: Paper Closure Pass
- [x] Compile `paper/paper.tex` after recent figure/table/text edits.
- [x] Search for stale V2/future-work language that contradicts V3.1-as-mainline.
- [x] Search for over-strong claims: all-baseline significance, strict within-3% static gap, extreme-storm validation without event-fraction wording.
- [x] Fix author/affiliation placeholders if user provides final metadata; resolved
  from the later user metadata recorded under "Unresolved Metadata".
- [x] Produce a short paper closure report with remaining caveats.
- **Status:** complete

### Phase 3: V3.1-Aligned Ablation Monitoring
- [x] Remote tmux/result status checked after server recovery.
- [x] Full result snapshot synced locally under `reports/v31_ablation_aligned/`.
- [x] Completion accepted: A1 `80/80`, A2 `40/40`, H1 `45/45`.
- [x] Collector CSVs available:
  - `reports/v31_ablation_aligned/v31_aligned_a1_stats.csv`
  - `reports/v31_ablation_aligned/v31_aligned_a2_stats.csv`
  - `reports/v31_ablation_aligned/v31_aligned_h1_stats.csv`
  - `reports/v31_ablation_aligned/v31_aligned_completion_check.csv`
- **Status:** complete

### Phase 4: Decide Whether to Upgrade Ablation Narrative
- [x] Compare V3.1-aligned A1/A2/H1 against existing V2/development diagnostics.
- [x] Upgrade manuscript wording from “development diagnostics” to “V3.1-aligned diagnostics”.
- [x] Replace A2 table, A1 table, and H1 heatmap with V3.1-aligned assets.
- [x] Keep interpretation conservative:
  - AWBC + oracle prior jointly significant.
  - MaskedActor-only significantly worse than full PD-PPO.
  - ActionEmbedding, EventAwareCritic, and action mask are architectural supports, not individually proven performance drivers in this batch.
- **Status:** complete

### Phase 5: Final Verification and Backup
- [x] Compile manuscript with XeLaTeX.
- [x] Save updated `paper/paper.pdf`.
- [x] Confirm A2/A1 tables and H1 heatmap are present and readable by rendering pages 47--52.
- [x] Commit paper subrepo changes as a submission-candidate checkpoint (`7defe35`).
- [x] Commit planning/report changes in main repo (`e92bec6`).
- [x] `reports/v31_ablation_aligned/` kept untracked (generated heavy artifacts).
- **Status:** complete

### Phase 6: Critical Manuscript Gap Repair From Review
- [x] Audit and fill reproducibility/data details:
  - [x] Data availability statement in paper §Code and Data Availability.
  - [x] Code repository link at default GitHub URL
        (historical package; not sufficient for the current SCENEBAL-2 archive).
  - [x] Frozen TCN oracle architecture: 3 layers, dilations 1/2/4, ReLU, dropout 0.05, AdamW.
  - [x] Target set $\mathcal{U}$, $H=8$, uniform $\alpha_h=1$, target weights $\beta_v$ listed in §3.5.
  - [x] AoI: freshness-age scoring + feasibility projector. Round-robin: cyclic order proportional to power draw.
  - [x] 5 physical instrument families → 8 logical scheduling channels (§3.1).
  - [x] Normalized deployment cost units; no watt-calibrated power traces claimed.
  - [x] External field-data validation: not claimed; RS485 logs not publicly distributed.
- [x] RS485 reader evidence integrated:
  - [x] README.md and SENSOR_DATA_SPEC.md read; sequence shapes summarized in findings.md.
  - [x] Decision: no appendix/table; raw logs not publicly distributed.
- [x] P1 FC4/power-cost: scheduling cost ≠ datasheet wattage; explicit wording in sensor_specs, §3.2, §4.1.
- [x] P2 round-robin claims: “lower mean FW-MAE” language; Bonferroni non-significance noted.
- [x] P3 innovation narrative: architectural supports vs statistically supported drivers (AWBC+prior pair).
- [x] Recompile and check affected pages after Phase 6 edits
      (superseded by the later `paper/main.pdf` rebuild and text audit).
- **Status:** complete

### Phase 7: Final Checkpoint
- [x] Re-run stale-language scans: no over-claim language remaining.
- [x] Compile final manuscript with XeLaTeX (67pp, 0 undefined refs/citations, 22 overfull hbox).
- [x] Save final PDF (`paper/paper.pdf`).
- [x] Commit paper subrepo as submission-ready checkpoint (`c02a0bc`).
- [x] Update planning files with final caveats.
- **Status:** complete

### Phase 8: ESWA Full Rewrite Takeover and Evidence Reconciliation
- [x] Recover prior planning state and read the 2026-05-25 rewrite strategy/evidence ledger.
- [x] Recognize that the Phase 7 "submission-ready" manuscript is superseded by the
  later ESWA rewrite and evidence-reconciliation decision; its results remain evidence only.
- [x] Reconcile the V3.1 static-comparator gap raised at handoff:
  superseded by the current SCENEBAL-2 fixed-mask replay protocol and active
  ESWA subproject plan.
- [x] If justified by code and stored artifacts, produce a locked summary for the
  trained-selected candidate static comparator; otherwise constrain manuscript claims
  to the existing fixed-priority `feasible_static_projected` comparator.
  Superseded by the later true fixed-mask replay and static-reference-normalised
  evidence package.
- [x] Inspect the current rewritten manuscript source under `paper/` and resume
  evidence-led drafting/editing from the first incomplete section. Superseded by
  the active `.planning/2026-06-10-eswa-terminology-rewrite/` manuscript pass.
- [x] Compile and run claim/compliance checks after the next coherent drafting unit.
  Completed repeatedly in the active ESWA subproject plan.
- **Status:** superseded_complete

### Phase 9: PD-PPO Static-Break Scenario Recalibration
- [x] Accept that the PD-PPO scene is manually calibrated and may be redefined;
  the goal is to find a reproducible scenario where static shortcuts are not
  structurally dominant.
- [x] Migrate the v1 v6 complex static-break sensor calibration into a PD-PPO
  sensor config.
- [x] Extend oracle-lift schedule diagnostics with v6-specific dynamic schedules
  based on SPC/fc4/context complementarity rather than laser-only switching.
- [x] Supersede the early local linear-oracle smoke calibration path with the
  later remote V25--V31 structural gates and final met+specialist TCN/PPO gate.
- [x] Launch and complete the final remote TCN-oracle gate:
  `reports/v31_metpair_stronglatent_seed45_h075_20260620/v2_tcn_oracle.pt`.
- [x] Append the recalibration chronology and final result to root
  `CHANGELOG.md`.
- [x] After structure gate passed, run the reduced PPO split-protocol gate:
  `custom_ppo.pt` completed to `120000` timesteps.
- [x] Back up the current supervisor-draft PDF before changing scenario evidence:
  `paper/_archive/pdppo_crst_rewrite_20260608_213436_pre_static_break_recalibration.pdf`.
- [x] Upgrade the scenario gate from "dynamic scheduling is meaningful" to a
  stricter static-shortcut gate for the current recalibration objective:
  - learned PPO beats source selected static on final-test windows;
  - explicit dynamic replay beats replay-local true fixed static with static
    duty guard disabled;
  - behaviour audit rejects fixed/simple-cycle deployment;
  - zero warm-up aborts.
  The earlier 8/10-seed gate is retained as optional final-submission
  robustness, not as the 20-hour recalibration gate.
- [x] Clarify the old `deployable static` ambiguity by using a true fixed-static
  no-duty-guard replay reference for the final met+specialist gate.
- [x] Close the V24 dual-flux phase24 learned candidate after locked seeds
  `41--45` failed strict replication against static-priority and duty baselines.
- [x] Launch a V25 low-budget static-squeeze TCN gate using the V24
  event-selective-laser sensor model at B=`1.03/1.05/1.08`.
- [x] Record the first V25 structural pass:
  `particle_heavy_flux_v7`, B=`1.03`, dynamic margin `+0.039284` against
  deployable static and event margin `+0.030887`, with diverse mid-duty dynamic
  behaviour and laser shortcut broken.
- [x] Run split-replay/PPO follow-up for the V25 low-budget structural pass;
  it failed and was superseded by later V31 met+specialist evidence.
- [x] Launch split-replay follow-up for
  `particle_heavy_flux_v7`, B=`1.03` in tmux
  `pdppo_v25_lowbudget_splitreplay_particle_b1p03_seed45_h082_20260620`.
- [x] Prepare a V25 low-budget learned-PPO runner with automatic behaviour
  complexity audit and a non-cycle default teacher path.
- [x] After V25/V26/V27 failed learned or strict replay gates, escalate from
  scenario-only tuning to framework/generator redesign with explicit anti-fixed
  and anti-simple-cycle behaviour diagnostics.
- [x] Add the first framework-level fallback mechanism for V26: calm/non-event
  observation reliability fields in the sensor runtime plus a calm-selective
  event-instrument sensor config.
- [x] Close V25 `particle_heavy_flux_v7 @ B=1.03` at split-replay:
  best replay `0.415506` loses to validation static `0.398729` and raw static
  `0.396226`.
- [x] Launch V26 calm-selective strict raw-static TCN gate in tmux
  `pdppo_v26_calm_selective_lowbudget_gate_seed45_h082_20260620`.
- [x] After the first V26 row failed strict raw-static headroom, implement and
  launch V27 latent-event-subtype structural diagnostics:
  particle-dominant, flux-dominant, and thermal-boundary regimes have different
  generated target structure, and oracle-lift now includes subtype-aware dynamic
  schedules.
- [x] Close V25 `particle_heavy_flux_v7 @ B=1.08` at split replay despite its
  earlier structural headroom:
  best replay `split_top3_l6_dwell6` loss `0.440489` loses to both AOI
  `0.437016` and replay-local raw static `static_action0=0.433427`.
- [x] Stop the V27 latent-subtype strict scan for the current recalibration
  objective after learned V27 variants failed strict static replay; the branch
  is superseded by the V31 met+specialist candidate.
- [x] Record the first V26 strict raw-static structural pass:
  `particle_heavy_flux_v7 @ B=1.08`, dynamic loss `0.362321`,
  raw static `0.385357`, raw-static margin `+0.059779`, event margin
  `+0.051663`, diverse dynamic behaviour, `gate_pass=True`.
- [x] Launch V26 B=`1.08` split-replay follow-up in tmux
  `pdppo_v26_splitreplay_particle_b1p08_seed45_h082_20260620`.
- [x] Close V26 `particle_heavy_flux_v7 @ B=1.08` at split replay:
  best replay `split_top2_l6_dwell12` loss `0.425502` loses to replay-local
  raw static `static_action1=0.409637`, so no learned PPO is launched.
- [x] Prepare a V26 learned-PPO wrapper for the exact passing configuration,
  gated on split-replay success:
  `run_pdppo_static_break_v26_calm_selective_low_budget_learned_ppo_seed45_h082_20260620.sh`.
- [x] Make the split-protocol path V27-ready by propagating latent event-subtype
  truth-generation parameters through `59_v31_split_protocol_grid.py`,
  `58_v31_split_protocol_run.py`, and `25_v2_train_custom_ppo.py`.
- [x] Add a stronger subtype diagnostic family, `subtype_auto`, that chooses
  calm/particle/flux/thermal masks from subtype-specific static-candidate losses
  rather than relying only on hand-written subtype schedules.
- [x] Launch a single-combo V27 `subtype_auto` probe for
  `particle_heavy_flux_v7 @ B=1.05` in tmux
  `pdppo_v27_subtype_auto_probe_particle_b1p05_seed45_h082_20260620`.
- [x] Record the V27 `subtype_auto` structural pass:
  `particle_heavy_flux_v7 @ B=1.05`, best dynamic
  `dynamic:subtype_auto_c1_p0_f1_t0_lead6`, loss `0.419422`, raw-static margin
  `+0.046764`, event dynamic margin `+0.065354`, with nontrivial mid-duty and
  switch diagnostics.
- [x] Extend `70_v31_split_replay_gate.py` with
  `--replay-family subtype_auto` so subtype-conditioned schedules can be tested
  under the same split-replay/raw-static gate.
- [x] Add and remotely validate the V27 subtype-auto split replay runner:
  `run_pdppo_static_break_v27_subtype_auto_low_budget_split_replay_gate_seed45_h082_20260620.sh`.
- [x] Launch V27 `subtype_auto` split replay in tmux
  `pdppo_v27_subtypeauto_splitreplay_particle_b1p05_seed45_h082_20260620`.
- [x] Decide V27 `subtype_auto` promotion only after split replay completes:
  pass requires beating both the source reference and replay-local raw
  fixed-static reference; if it passes, learned PPO still needs an observable
  risk/subtype-belief or memory mechanism because the diagnostic schedule uses
  privileged subtype labels.
- [x] Record V27 `subtype_auto` split-replay pass:
  best replay `split_subtype_auto_top2_c0_p1_f1_t0_l0`, loss `0.501525`,
  source-reference margin `+0.017509`, raw-static margin `+0.011146`,
  `gate_pass=True`.
- [x] Run behaviour-complexity audit on the best V27 subtype-auto replay:
  `unique_mask_count=9`, `top3_mask_fraction=0.794189`,
  `mask_entropy_bits=2.335018`, `transition_entropy_bits=2.771129`,
  `event_sensor_l1=2.582324`, `event_mask_mi_bits=0.344734`,
  `behavior_complexity_gate_pass=True`.
- [x] Implement the learned-policy bridge for V27 subtype-auto:
  add observable regime inference through memory or a learned
  risk/subtype-belief head, and add a non-privileged training/evaluation path
  that does not expose generated subtype labels at deployment time.
- [x] Add `awbc_teacher_mode=subtype_auto` and propagate subtype teacher masks
  through `25_v2_train_custom_ppo.py`, `58_v31_split_protocol_run.py`, and
  `59_v31_split_protocol_grid.py`.
- [x] Add and launch the V27 subtype-auto learned PPO runner:
  `run_pdppo_static_break_v27_subtype_auto_low_budget_learned_ppo_seed45_h082_20260620.sh`,
  tmux `pdppo_v27_subtypeauto_learnedppo_particle_b1p05_seed45_h082_20260620`.
- [x] Evaluate the first learned V27 subtype-auto PPO under the stricter
  final-test raw-static gate:
  learned `custom_ppo=0.516547` loses to replay-local raw static around
  `0.514839`, so it is not paper-mainline evidence despite passing behaviour
  complexity.
- [x] Add actor BC warm-start as the next framework-level learning fix:
  `src/v2/custom_ppo.py` now supports configurable supervised teacher
  pretraining before PPO, with CLI plumbing through `25`, `58`, and `59`.
- [x] Launch the BC warm-start V27 subtype-auto PPO runner:
  `run_pdppo_static_break_v27_subtype_auto_low_budget_bc_warmstart_ppo_seed45_h082_20260620.sh`,
  tmux `pdppo_v27_subtypeauto_bcwarm_ppo_particle_b1p05_seed45_h082_20260620`.
- [x] Evaluate the BC warm-start run:
  it still loses to same-run `feasible_static_projected`
  (`custom_ppo=0.517628`, static `0.517380`) while passing behaviour
  complexity, so it is not paper-mainline evidence.
- [x] Add optional observable regime-belief features derived from observation
  history/mask coverage and propagate
  `--include-observable-regime-belief` / `--regime-belief-lookback` through
  `25`, `58`, `59`, metadata recovery, and custom PPO train/eval env copies.
- [x] Launch the belief+BC+PPO runner:
  `run_pdppo_static_break_v27_subtype_auto_low_budget_belief_bc_ppo_seed45_h082_20260620.sh`,
  tmux `pdppo_v27_subtypeauto_belief_bc_ppo_particle_b1p05_seed45_h082_20260620`.
- [x] Evaluate the belief+BC+PPO run against the same strict raw-static and
  behaviour-complexity gates before considering paper-mainline migration.
- [x] Evaluate the belief+BC+PPO run:
  failed same-run static (`custom_ppo=0.519019` vs
  `feasible_static_projected=0.518138`) while passing behaviour complexity.
- [x] Add a supervised observable subtype auxiliary head to the custom PPO
  actor. The head uses `event_subtype_id` only as a training target and keeps
  inference inputs observable.
- [x] Launch subtype-auxiliary PPO in tmux
  `pdppo_v27_subtypeauto_subtypeaux_ppo_particle_b1p05_seed45_h082_20260620`.
- [x] Evaluate the subtype-auxiliary PPO run against same-run static,
  behaviour complexity, and strict raw-static replay.
- [x] Evaluate subtype-auxiliary PPO:
  it beat same-run `feasible_static_projected` (`0.506899` vs `0.515707`) and
  passed behaviour complexity, but lost to replay-local raw static
  (`0.505811`) by `0.001088`.
- [x] Run stronger subtype auxiliary / imitation variants and accept only if
  learned PPO beats replay-local raw static with nontrivial behaviour; the V27
  branch produced only a near-threshold single-seed candidate and was
  superseded by the V31 met+specialist result.
- [x] Identify first single-seed learned candidate:
  strongbc2 seed45 beats replay-local raw static by `+0.005198` over required
  `0.005140` and passes behaviour complexity.
- [x] Do not reproduce V27 strongbc2 on seeds 46/47 for paper-mainline
  migration; this branch is superseded by the stronger V31 met+specialist
  candidate.
- [x] Resolve the V27 collapse path by moving to the met+specialist-pair
  simulator/sensor contract plus corrected behaviour-complexity audit.
- [x] Supersede the V27/energy-account branch for the current 20-hour
  recalibration objective after the met+specialist-pair scene passed the
  strict static replay gate and learned-PPO behaviour gate.
- [x] Add `configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml`,
  allowing `met_station_core + one specialist` under `budget=0.75` while
  keeping two specialists infeasible.
- [x] Run the final TCN-oracle and reduced-PPO gate:
  `reports/v31_metpair_stronglatent_seed45_h075_20260620`, with
  `v2_tcn_oracle.pt`, `custom_ppo.pt`, and PPO training completed to
  `120000` timesteps.
- [x] Verify learned forecast gate:
  router-confidence re-evaluation gives `custom_ppo=0.485635` versus
  `validation_selected_static=0.491597`.
- [x] Verify strict no-duty-guard replay-local static gate:
  best explicit dynamic `split_metpair_subtype_explicit_l4=0.482174` beats
  true fixed static `static_action10=0.492351`, `gate_pass=True`.
- [x] Verify non-fixed, non-simple-cycle behaviour with the corrected audit:
  `unique_mask_count=4`, `event_mask_mi_bits=0.520959`,
  `event_sensor_l1=1.579090`, `state_dependent=True`,
  `behavior_complexity_gate_pass=True`.
- [x] Add root `CHANGELOG.md` with the recalibration chronology, failed
  branches, final candidate evidence, and paper-mainline decision.
- [~] Robustness extension for a strong claim: reproduce the met+specialist
  candidate on additional seeds before paper rewrite.
  - Strong claim target: at least `10` complete seeds, at least `8/10` full
    seed-gate passes, positive mean learned margin, and positive mean strict
    replay margin.
  - First replication batch launched on `remote-gpu`:
    `metpair_s41` through `metpair_s44`, plus `metpair_s46` and
    `metpair_s47`.
  - Automation added:
    `scripts/run_v31_metpair_strongclaim_seed_sweep_20260620.sh` and
    `scripts/72_v31_collect_metpair_strongclaim.py`.
  - Result: old metpair branch failed the 7-seed robustness test:
    only `1/7` full seed-gate passes and negative mean learned margin.
  - Pivot: launch `v31_metpair_backbone_context` with required met backbone,
    explicit agent context alerts, balanced subtype probabilities, and
    subtype-balanced final-test windows.
- [~] Round BO-1, ortholinear balanced-objective 14-seed expansion:
  - Purpose: test whether switching the experiment contract to the same
    regime-balanced event-subtype objective used for static selection, PPO
    evaluation, replay macro gate, and claim collection restores the older
    strong claim without relying on posthoc macro reinterpretation.
  - Completed pilot evidence on seeds `41` and `42` was positive under the new
    strict old-claim collector.
  - Partial 8-seed aggregate (`41,42,43,45,47,49,51,53`) is a real
    breakthrough direction: learned PD-PPO beats static/rule/operational
    baselines on all 8 seeds; old-claim step gate is `7/8` only because seed49
    fails strict replay; old-claim macro gate is `8/8`.
  - Final 14-seed checkpoint completed:
    `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_oldclaim_20260621/`.
    It is a major breakthrough for the regime-balanced macro claim:
    old-claim macro gate `14/14`, behaviour gate `14/14`, learned PPO beats
    static/rule/operational baselines on the step and macro objectives
    `14/14`, one-sided sign test for macro gate `p=0.00006104`.
  - Evidence boundary remains explicit: unqualified step-weighted replay claim
    is not yet supported because strict explicit-replay step gate is `11/14`;
    failing seeds are `48,49,52`. Learned-policy true-static step gate is
    `12/14`, failing seeds `41,48`.
  - Breakthrough report written remotely:
    `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_14seed_oldclaim_20260621/BREAKTHROUGH_REPORT.md`.
  - User requested a 24-hour autonomous continuation. A guarded tmux runner is
    active on `remote-gpu`:
    `bo24_autonomy_20260621`, starting from seeds `55--66` and then continuing
    in 12-seed waves until the 24-hour window ends or disk free space drops
    below `200GB`. Each wave automatically aggregates and writes a Markdown
    report.
- **Status:** superseded experiment chronology; the later SCENEBAL-2
  `117--140` result completed the strong-claim objective and is now the
  manuscript mainline.

### Phase 10: Specialist-Bottleneck Theory Report Before Manuscript Application
- [x] Follow the user constraint: research and report first; do not directly
  modify the manuscript under `paper/`.
- [x] Audit the active theory/problem sections and identify the missing bridge
  from SCENEBAL-2 to a broader problem class.
- [x] Verify literature candidates through arXiv/CrossRef/DOI endpoints rather
  than writing citations from memory.
- [x] Produce the pre-application report:
  `reports/aggregate/specialist_bottleneck_theory_extension_report_20260621.md`.
- [x] User approval received to apply the theory extension after report review.
- [x] Apply edits to
  `paper/sections/*.tex` or `paper/references.bib`.
- **Status:** complete; manuscript application moved to Phase 11

### Phase 11: Specialist-Bottleneck Theory Application and ESWA Packaging
- [x] Resolve plan-file consistency debt across root plan, subproject plan, and
  `.planning/.active_plan`.
- [x] Apply the specialist-bottleneck definition, proposition, and appendix
  proof to the canonical manuscript source.
- [x] Add verified primary citations to `paper/references.bib` and use them in
  related work/problem framing without over-claiming adaptive-submodularity or
  POMDP theory.
- [x] Rebuild `paper/main.pdf` and check for undefined references/citations.
- [x] Record the applied change in `progress.md` and `findings.md`.
- **Status:** complete

### Phase 12: Manuscript Completeness and Metric-Boundary Consistency
- [x] Verify that the old public GitHub release cannot be used as the current
  SCENEBAL-2 evidence archive. The `v0.1.0` release URL returns `404`, and
  `git ls-remote --tags` reports no current release tag.
- [x] Keep the manuscript data-availability statement forward-looking and
  specify the exact required archive contents: code, SCENEBAL-2 aggregate
  tables, seed-level summaries, figure assets, and reproduction scripts.
- [x] Add the SCENEBAL-2 subtype-balance and target-weighting design statement
  to `paper/sections/05_simulation_setup.tex`.
- [x] Add the raw unnormalised subtype-macro sensitivity boundary to
  `paper/sections/06_results.tex`: explicit replay macro remains positive
  `24/24`, but the learned-policy raw macro gate is `0/24`.
- [x] Rebuild `paper/main.pdf` and check that the PDF text contains the new
  SCENEBAL-2 design, metric-boundary, and data-availability statements.
- [x] Record this consistency pass in the plan, progress, and findings files.
- **Status:** complete

### Phase 13: ESWA Terminology, Title, and Figure-Style Refinement
- [x] Read `docs/06-23-01.md` and apply its terminology guidance to the active
  manuscript source rather than the historical rewrite files.
- [x] Optimise the title to
  `Forecast-Loss-Driven Projected PPO for Power-Constrained Sensor Scheduling`.
- [x] Replace manuscript-facing `gate`, `oracle`, `true fixed-static`, and
  internal benchmark language with ESWA-style terms: fixed-mask baseline,
  paired comparison, fixed evaluation forecaster, event-aware diagnostic
  reference, and behavioural diagnostics.
- [x] Redraw the main framework asset from the image2 exploration into a
  deterministic repository-local PDF/PNG figure.
- [x] Rework the mechanism/robustness figure to use the same plotting style and
  seed-count labels instead of dense margin annotations.
- [x] Record the reference-paper download status under
  `paper/reference_papers/eswa_similar/README.md`, including blocked ESWA/HAL
  retrievals and the three valid local PDF references.
- [x] Rebuild `paper/main.pdf`, run a PDF text audit, and request a final
  sub-review pass.
- [x] Fix final sub-review findings: clarify the superiority-margin formula,
  add median/bootstrap CI rows, define behavioural diagnostics, improve Figure
  5/6 readability, remove Type 3 fonts, and replace residual internal wording.
- **Status:** complete

### Superseded Round: SCENEBAL-1 Balanced Scene/Target Generation
- [~] Round SCENEBAL-1, balanced scene/target generation:
  - Purpose: preserve PPO and the met+one-specialist sensor geometry while
    rebalancing subtype probabilities, latent strengths, and target weights so
    fixed `met_station_core|fc4_flux` no longer dominates the step objective.
  - Pilot seeds `93--98` completed with operational step `6/6`, operational
    macro `6/6`, behavior `6/6`, and strict replay step/macro `6/6`.
  - Expansion seeds `99--104` reproduced the same operational and behavior
    gates. Combined `93--104` gives `12/12` operational step/macro, behavior,
    and replay gates with sign-test p `0.000244140625`.
  - Expansion seeds `105--110` are verified complete on `remote-gpu` with
    operational step `6/6`, operational macro `6/6`, behavior `6/6`, and replay
    step/macro `6/6`.
  - Evidence boundary remains explicit: learned true-static macro dominance is
    not yet established, so unconditional "beats every true fixed static under
    every macro score" is not allowed.
  - Next action: sync/aggregate the `105--110` wave locally, produce combined
    `93--110` reports, then decide whether to continue seed expansion or run a
    targeted true-static macro diagnostic.
- **Status:** superseded by SCENEBAL-2 24-seed evidence (`117--140`)

## Superseded Experiment: Energy-Account Specialist Pilot

- [x] Add subtype-router low-confidence fallback to train/split/grid entry
  points.
- [x] Record target weights/scales in split protocol manifests.
- [x] Confirm current best low-confidence replay is not acceptable:
  `1.005365` vs raw static `1.012736`, but behaviour gate fails with only
  three masks and simple-cycle/fixed-like flags.
- [x] Confirm behaviour-valid replay exists but is not forecast-optimal:
  `router_contract_scan_c4_fb13_t086`, `1.045930`, behaviour gate pass.
- [x] Launch energy-account pilot:
  `v31_energy_pilot_20260620`; stopped because full oracle-greedy BC was too
  slow for an interactive pilot.
- [x] Launch reduced-cost replacement:
  `v31_energy_fast_20260620`.
- [x] Monitor fast pilot to completion; failed forecast and behaviour gates.
- [x] Launch subtype-router strongBC energy pilot:
  `v31_energy_router_20260620`.
- [x] Stop treating the energy-account branch as active for the 2026-06-20
  recalibration objective. The met+specialist-pair scene cleared the required
  forecast/static/behaviour gates first.
- [x] Defer any remaining energy-account replay/audit work to a future
  robustness or second-scenario study.
- **Status:** superseded by met+specialist-pair result

## Unresolved Metadata (Pre-Submission)
- [x] Affiliation resolved from user metadata:
  `School of Mechanical Engineering, Southeast University, Nanjing, China`
  (`东南大学机械工程学院`).
- [x] Corresponding author resolved from user metadata:
  Yongzhe Li, Tel.: `+86 13946019751`.
- [ ] Code repo commit hash not pinned (dynamic main branch).
- [ ] Versioned SCENEBAL-2 archive / DOI not pinned. The old GitHub release is
      historical and should not be cited as the current evidence package.

## Current Evidence Inventory

## Current 2026-06-21 PD-PPO Mainline Claim Status

| Candidate | Evidence | Status |
|---|---:|---|
| SCENEBAL-2 seeds `117--140` | operational, explicit replay, behaviour, true-static macro, true-static step sign, strict-margin true-static step, old-claim step/macro gates all `24/24` | Current mainline evidence |
| SCENEBAL-1 seeds `93--116` | earlier 24-seed result | Superseded by SCENEBAL-2 because SCENEBAL-2 gives a cleaner specialist-bottleneck mechanism |
| Old metpair seed45 / 7-seed replications | single-seed or weak multi-seed gates | Historical diagnostics only |

Supported manuscript boundary: PD-PPO learns a non-fixed, non-cyclic,
state-dependent specialist scheduler in the SCENEBAL-2 regime-balanced
backbone-plus-one-specialist microclimate benchmark. This is not a universal
optimality theorem for arbitrary sensor scheduling systems.

Metric boundary added on 2026-06-21: the strong manuscript claim is tied to the
ordinary step objective and the static-normalised event-regime macro objective.
The raw unnormalised subtype-macro diagnostic does not support learned-policy
macro dominance (`0/24`), although explicit replay macro remains positive
`24/24`. Do not phrase the result as dominance under every aggregation.

## Historical 2026-06-20 PD-PPO Claim Status

| Candidate | Evidence | Status |
|---|---:|---|
| Old metpair seed45 | 1/1 seed gate pass | Mechanism demo only; not robust |
| Old metpair 7 seeds | 1/7 strict seed gate pass | Not supported |
| Backbone-context 7 seeds | 3/7 strict seed gate pass; 5/7 macro-positive | Behaviour solved, static shortcut remains |
| Strong-latent probes 43/44 | 0/2 strict seed gate pass; 2/2 macro-positive | Promising for regime-macro claim only |
| Strong-latent partial 4 seeds | 0/4 strict seed gate pass; 2/4 macro-positive | Failed; seed41 exposes fixed static shortcut |
| Ortholinear seed41 raw PPO | 1/1 strict raw seed gate pass; learned macro negative | Structural replay fixed; learned flux subtype still weak |
| Ortholinear strong-teacher seed41 | superseded before paper migration | Replaced by SCENEBAL-2 evidence |

Historical next-step note at the time: finish ortholinear strong-teacher seed41
and judge it under both raw learned PPO and router-confidence deployment. This
note is now superseded by the completed SCENEBAL-2 `117--140` evidence block.

Current narrative boundary: seed45 can illustrate mechanism, but the paper
mainline should not claim broad optimality from the old single-seed result. If
the ortholinear/strong-teacher branch holds, the claim can be framed as
forecast-oriented contextual specialist scheduling under a fixed meteorological
backbone. If only macro holds, the wording must explicitly say
regime-balanced/event-subtype macro rather than broad step-weighted optimality.

| Evidence block | Status | Role in paper |
|---|---|---|
| V3.1 S2 main experiment, 3 budgets x 10 seeds | complete | Main performance result |
| V3.1 event-fraction condition table | complete | Event-heavy/calm/mixed stratified analysis |
| G1 V3.1 generator validation | complete | Simulation credibility |
| V2 A1/A2/H1 ablations | complete | Historical/development diagnostics only |
| V3.1-aligned A1/A2/H1 rerun | complete | Current component/path/hyperparameter diagnostics |
| Hardware/manual sensor evidence | integrated, needs wording discipline | Motivation for normalized costs |
| Real RS485 reader/data shape | available locally, not yet integrated | Supports hardware/data availability and raw sequence format |

## V3.1-Aligned Ablation Snapshot
Last local sync: 2026-05-16.

| Experiment | Expected | Local eval files | Notes |
|---|---:|---:|---|
| A2 staged | 40 | 40 | Complete; full D4 is best in staged path |
| A1 remove-one | 80 rows | 80 completion rows after reference reuse | No AWBC/prior and masked-only significantly worse |
| H1 hyperparameter | 45 rows | 45 completion rows after default reference reuse | All tested cells within 2.5% of default |

Local backup path: `reports/v31_ablation_aligned/`.

Key V3.1-aligned ablation numbers:

| Experiment | Result |
|---|---|
| A2 full PD-PPO (D4) | `0.1629 ± 0.0137` FW-MAE |
| A1 full PD-PPO | `0.1629 ± 0.0137` FW-MAE |
| A1 no AWBC/prior | `0.1853 ± 0.0209`, significant vs full |
| A1 masked-only | `0.1828 ± 0.0156`, significant vs full |
| H1 default `(AWBC=0.1, KL=1.0)` | `0.1616 ± 0.0138` |
| H1 best mean `(AWBC=0.1, KL=0.5)` | `0.1599 ± 0.0123`, descriptive only |

## Key Questions
1. Is the current manuscript already submission-close after V3.1-aligned ablations? Yes, subject to final metadata/commit decisions and any desired polish.
2. Should V2 A1/A2/H1 remain in the main text? No; V3.1-aligned diagnostics now supersede them.
3. What is the most important ablation conclusion? The AWBC + oracle prior pair is the statistically supported stabilisation mechanism.
4. What is the main risk now? Accidentally over-claiming individual component effects that are not significant in A1.
5. What is the newest blocking manuscript risk? Internal inconsistency between FC4 low measured average draw and high normalized scheduling cost, plus missing reproducibility/data details.

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep V3.1 S2 as mainline | It is complete, synced, and supports the intended high-level claim. |
| Upgrade A1/A2/H1 to V3.1-aligned diagnostics | The rerun completed and passed completion/sanity checks. |
| Keep individual component claims conservative | A1 does not show significant standalone degradation for ActionEmbedding, EventAwareCritic, or action mask. |
| Treat the reviewer gap list as a new critical repair phase | These are manuscript credibility issues, not optional polish. |
| Use `/home/horeb/_Data/SEUAWS/rs485-reader` as the real-sensor evidence source | It contains the serial reader, data specification, and sample JSONL/CSV logs needed to document raw sequence shape. |
| Use file-based planning in repo root | The task spans paper, reports, scripts, and server state; persistent planning reduces context loss. |
| Preserve historical TODO as archive | It contains useful chronology but is too long and mixed to be the active working plan. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Local `python` command unavailable | 1 | Use `python3` or environment-specific Python where needed. |
| Local pandas unavailable under system `python3` | 1 | Avoid pandas for light checks or use the correct conda env when present. |
| `rg` command pattern interpreted backticks in shell | 1 | Use quoted/simpler searches or Python string scans for literal Markdown snippets. |

## Notes
- Re-read this file before changing paper claims or deciding whether to merge new experiments.
- Update `progress.md` after each paper or experiment phase.
- Update `findings.md` whenever new experimental results or manuscript caveats are discovered.
- The 2026-05-25 rewrite authority is `docs/05-25-crst-rewrite-strategy.md` together
  with `docs/05-25-full-rewrite-evidence-ledger.md`; earlier Phase 1--7 closure
  records describe the superseded algorithm-first manuscript checkpoint.
