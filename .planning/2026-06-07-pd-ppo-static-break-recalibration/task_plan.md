# Task Plan: PD-PPO Static-Break Scenario Recalibration

## Goal
Find and validate a PD-PPO scene where static shortcuts are structurally broken
and adaptive scheduling has real forecast-value headroom, while the learned
schedule keeps a nontrivial dynamic duty profile instead of collapsing to
multiple always-on or always-off sensors.

Clarified hard target: the accepted scheduler must show real dynamic scheduling. A low-loss policy is invalid if it relies on several sensors being permanently on/off, a single nearly-static sensor shortcut, or high-frequency thrashing that would be operationally unrealistic.

Current hard target, updated 2026-06-21: the final evidence must support a
strong ESWA claim that PPO/PD-PPO is forecast-optimal under the tested protocol
and produces genuinely state-dependent scheduling. The scheduler cannot be a
fixed sensor subset, a simple back-and-forth rotation, or a renamed rule policy.
If scene-only tuning stops improving, deeper changes are allowed: simulator/data
generation, teacher/oracle construction, PPO observation features, auxiliary
heads, memory/lead context, reward shaping, replay/evaluation gates, and
moderate explainable sensor/noise calibration. PPO must remain the final learned
scheduler being validated.

## Scope
- This plan is isolated from v1 and from manuscript editing.
- v1 is an archived reference, not part of this fork. Its records, failed
  routes, and diagnostic lessons may be read to avoid repeating unproductive
  directions, but v1 code/methods/results must not be merged into the active
  PD-PPO implementation, main evidence tables, or first-paper claim chain.
- Do not run a full PPO grid until a structural gate shows that the scene is worth training on.
- Use only `remote-gpu` for remote execution. Do not use historical IPs,
  UniVPN/aTrust paths, old tunnels, or hardcoded host addresses.
- Anti-stall rule: for each modification direction, allow at most `10` bounded
  work units without effective improvement. A unit may be a seed wave, locked
  pilot, diagnostic branch, or another bounded experiment with aggregate
  evidence and a written keep/pivot decision. If failed or likely failed before
  10 units, pivot immediately to another layer.
- Preserve the current met+one-specialist microclimate sensing geometry as the
  baseline. Moderate sensor-config changes are allowed only when explainable as
  simulated variants of the same sensing setup.

## Current Phase
Flexible-subset Stage-B scene screening; no PPO launch permitted until the
forecast-geometry and online-transfer gates pass on the same frozen assets.

## Direction update (2026-09-10)
- The arbitrary-subset objective remains active. The immediate target is not
  to force a positive PPO result, but to establish a physically traceable,
  state-dependent resource/quality chain that creates downstream subset-value
  separation.
- The SOC route is not currently admissible as an entity-system model. The
  existing audit uses a non-controllable 600 W external heater profile and
  only about 25 hourly truth rows; its output is an envelope diagnostic, not
  evidence for a battery-powered deployment.
- V609--V612 dynamic heater/frequency/quality variants are closed before PPO.
  Their resource frontiers changed, but operating forecast gaps were
  approximately zero on most development seeds.
- V613 quality-coupled screening is also not a pass: online transfer is
  partially usable, but complete-subset opportunity is not consistently above
  the predeclared threshold. No V613 PPO result is promoted.
- V614 truth/resource screening now uses an independent cold-availability
  relation from the SEUAWS Modbus test and a separate physical-watt heater
  manifest. It passed the truth/resource gate across four seeds, with
  state-dependent quality, four joint heater states, and 55 W feasibility.
- The first V614 output mixed physical watts with the legacy normalized 2.15
  budget and was discarded before geometry. The corrected rerun is the only
  V614 evidence. V615 frozen-asset preparation is now running from it.
- V615 completed its four-seed operating geometry audit but failed materiality;
  the strongest static candidate remained the same in every seed. A bounded
  V616 fidelity correction is now running: the measured availability relation
  is used directly with `availability_floor=0`, while all scene, resource,
  budget, split, and candidate settings remain unchanged.
- Next route: implement/validate a persistent operating-factor chain driven by
  deployable nowcast features, then rerun truth-only resource geometry,
  downstream subset geometry, and chronological online transfer. Do not
  retune heater thresholds, budgets, target weights, or SOC values to recover a
  margin.

## Phase 14: Observable Physical Resource Chain

- [ ] Generate a resource trace from deployable weather inputs and the declared
  GMX500/Parsivel heater rules; do not use generator modes or future targets.
- [ ] Run the remote V627 occupancy and 32-subset feasibility screen on four
  development seeds.
- [ ] Require state-dependent heater support and a changing feasible frontier
  before refitting any forecaster or launching online transfer.
- [ ] If V627 passes, prepare matched frozen assets and audit downstream subset
  forecast geometry. If it fails, close the heater route with evidence and
  select the next entity-grounded resource mechanism.
- [x] V627 passed the physical occupancy/frontier screen.
- [x] V628/V629 completed the matched frozen-asset and downstream geometry
  audit without loss clipping.
- **Status:** pivot required before online transfer or PPO

### V629 decision

The nowcast-driven heater controller produced real resource-state variation,
but the downstream operating opportunity gaps were `0`, `0`, `0`, and `0` for
the resource-conditioned audit; ordinary condition gaps were
`0.00053`, `0`, `0.07485`, and `0.00529`. The effect is therefore not
consistent across development seeds. V629 is closed as a physical-resource
screen, not promoted to policy evidence. The next route must couple the
observable operating factors to both target innovation and measurement
quality before another frozen forecaster is fitted.

## Phases

### Phase 1: Context Isolation
- [x] Create an isolated planning directory under `rl_sensor_scheduling_framework/.planning/`.
- [x] Set `.planning/.active_plan` to this recalibration task.
- [x] Keep root/v1 planning files out of this execution loop.
- **Status:** complete

### Phase 2: Static-Break Scene Implementation
- [x] Port an archived v6-style sensor-calibration idea into PD-PPO as a
  historical starting point only.
- [x] Add v6-specific SPC/fc4/context dynamic schedule diagnostics.
- [x] Add a calibration gate script to sweep budgets and target-weight profiles.
- [x] Compile new scripts.
- [x] Run local linear-oracle smoke calibration.
- **Status:** complete

### Phase 3: Structure Gate
- [x] Check feasible subset diversity.
- [x] Check whether best/static top-K no longer collapses to laser shortcut.
- [x] Check whether dynamic event-conditioned schedules beat best static by at least the configured margin.
- [x] Append result to `../CHANGELOG.md`.
- **Status:** complete

### Phase 4: Remote TCN Gate
- [x] Sync modified config/scripts to `remote-gpu`.
- [x] Launch TCN-oracle calibration in tmux.
- [x] Monitor logs and sync results back.
- [x] Append result to `../CHANGELOG.md`.
- **Status:** complete

### Phase 5: Duty-Aware PPO Protocol
- [x] Add rollout-level duty metrics to all evaluated policies.
- [x] Add a training-time duty-balance shaping term for PPO.
- [x] Select the best scene from gate results under the dynamic-duty criterion.
- [x] Run a reduced PPO split-protocol grid.
- [ ] Compare PD-PPO, validation-selected static, event-conditioned dynamic diagnostics, and heuristics.
- [ ] Decide whether to expand to full 3-budget x 10-seed evidence.
- **Status:** pending

## Active Experiment Override (2026-09-10)

The current active objective is the PD-PPO flexible-subset redesign, not the
historical ESWA manuscript phase retained above.  The next route is V580
Stage-B truth-only screening.  It combines persistent specialist-separated
target dynamics and observation quality derived from noisy nowcasts with the
independent-nowcast heater/resource trace.  The scheduler may observe only
causal nowcast/alert context and executable runtime state; it must not receive
latent generator states, simulator event labels, or future targets.  No asset
preparation, online transfer, or PPO training is permitted until the
predeclared truth-support, persistence, resource, and subset-geometry gates
pass across all four development seeds.

### Phase 6: Energy-Account Structural Gate
- [x] Expose energy-account parameters through split/grid PPO wrappers.
- [x] Add coverage-group support to oracle-lift calibration gate.
- [x] Launch short TCN gates with coverage groups, energy account, and strict dynamic-duty filters.
- [x] Sync and analyze microstructure / decorrelation / high-amplitude structural gates.
- [x] Conclude that no strict structural gate passed under the current scene family.
- **Status:** complete

### Phase 7: Operational Baseline Constraint
- [x] Add evaluation-only duty-constrained heuristic baselines.
- [x] Forward the constrained-baseline options through split/grid wrappers.
- [x] Verify by syntax/dry-run only locally, then compile on the server.
- [x] Run a reduced server evaluation pass on the strongest existing PD-PPO settings.
- [x] Compare PD-PPO against both original and operationally constrained heuristics.
- **Status:** complete

### Phase 8: No-Warmup Hard-Duty Probe
- [x] Sync partial no-warmup metrics from the server without pulling heavy rollouts.
- [x] Aggregate the first 14 completed no-warmup runs.
- [x] Launch a reduced no-warmup + hard-duty server run.
- [x] Monitor the reduced run and sync results.
- [x] Decide whether no-warmup can be promoted as a valid static-break scene.
- **Status:** complete

### Phase 9: Switch-Limited Operational Baselines
- [x] Add evaluation-only minimum-dwell / switch-limited baseline variants.
- [x] Preserve original heuristic and static rows unchanged.
- [x] Run saved-run replay on the completed hard-duty reduced seeds.
- [x] Compare PD-PPO against static, original dynamic, duty-constrained, and dwell-constrained operational baselines.
- [x] Decide whether operational switching constraints produce a usable secondary claim.
- **Status:** complete

### Phase 10: Residual Static Shortcut Resolution
- [x] Audit whether mainline env-dwell12 PD-PPO truly beats baselines.
- [x] Record that current answer is qualified positive rather than full static dominance.
- [x] Inspect seed42 validation-selected static shortcut and identify why it survives.
- [x] Choose the next minimal fair correction: static deployability constraint, scene micro-calibration, or additional seed replication.
- [x] Run or launch only the highest-ROI follow-up experiment on the server.
- [x] Decide whether the remaining seed42 deployable-static gap requires stricter duty-high retraining.
- **Status:** complete

### Phase 11: Evidence Lock And Minimal Expansion
- [x] Aggregate h75 reduced 3-seed evidence.
- [x] Record that h75 is the strongest operational branch.
- [x] Decide whether to lock the 3-seed supervisor-draft evidence or launch a small 2-seed expansion.
- [x] Launch the locked-parameter seeds 44--45 extension.
- [x] Aggregate the 5-seed h75 evidence after completion.
- [x] Prepare the exact table rows/claim text needed for the English manuscript without editing `raw.tex`.
- **Status:** complete

### Phase 12: Full Static-Shortcut Break Gate
- [x] Audit the existing h75 10-seed result against deployable selected static.
- [x] Record that h75 is insufficient for the user's stronger requirement.
- [x] Add a boundary-switch v14 sensor calibration that makes the seed42 compact SPC shortcut infeasible.
- [x] Add a server runner for B=0.60/0.65 x seeds 41--43 with symmetric h75/dwell constraints.
- [x] Launch the reduced v14 gate on `remote-gpu`.
- [x] Stop the v14 budget gate after partial results showed no full static-break path.
- [x] Diagnose that fixed-priority duty replay remains too strong and PD-PPO is not event-conditioned.
- [x] Expose event-gated actor, event-start sampling, event reward multiplier, and SOC auxiliary controls through the split-protocol runners.
- [x] Add a v14 event-gated mechanism probe runner.
- [x] Launch the v14 event-gated mechanism probe on `remote-gpu`.
- [ ] Monitor completion and sync compact metrics.
- [ ] Audit whether event/non-event sensor duty actually diverges.
- [ ] Decide whether to keep algorithm-side event gating or move to v15 scene physics.
- [x] Add v15 event-complementarity sensor calibration.
- [x] Launch a v15 structural oracle-lift gate before any PPO retraining.
- [x] Align structural gate with deployable-static reference.
- [x] Audit v15/v16 structural gate dynamic headroom and static-shortcut failure modes.
- [x] Select `particle_heavy_flux_v7` as the strongest current profile family.
- **Status:** complete

### Phase 13: Independent Particle-Heavy PD-PPO Route
- [x] Write an explicit route file separating this fork from v1:
  `pdppo_independent_particle_heavy_route.md`.
- [x] Lock the active route to `rl_sensor_scheduling_framework` only.
- [x] Record that v1 is historical background only, not active method/evidence.
- [x] Promote `particle_heavy_flux_v7` plus `oracle_greedy` AWBC as the next
  learned-policy branch.
- [x] Monitor seed45 particle-heavy oracle-greedy PPO completion.
- [x] Sync metrics and audit raw per-policy losses, deployable-static wins,
  event/calm losses, and duty behaviour.
- [ ] If seed45 passes, launch locked seeds 41--45 with the same settings.
- [x] If seed45 fails, run v17 structural gate before additional PPO retries.
- [x] Audit v17 particle-heavy seed45 structural gate under h0.82 and env dwell 12.
- [x] Run a targeted v17 particle-heavy budget scan at B=1.05/1.10/1.20
  before any further PPO training.
- [x] Launch one v17 particle-heavy learned PPO probe at the strongest
  structural point: B=1.10, seed45, h0.82, oracle-greedy AWBC.
- [x] Monitor the v17 B=1.10 seed45 PPO probe, sync compact metrics, and
  decide whether to replicate seeds 41--45 or redesign again.
- [x] Audit event-pair replay under the split-run oracle after B=1.10 partial
  failure.
- [x] Launch focused B=1.05/B=1.20 budget-bracket learned probes on seed45.
- [x] Monitor the B=1.05/B=1.20 budget-bracket run and decide whether any
  budget clears deployable static plus dynamic baseline gates.
- [x] Launch B=1.10 balanced-training probe after budget bracket showed the
  dynamic-baseline gap is not budget-position-specific.
- [x] Monitor B=1.10 balanced-training result and decide whether it closes the
  non-event loss gap without losing deployable-static wins.
- [x] Add a weak-candidate-prior B=1.10 probe after balanced training failed to
  close the non-event loss gap.
- [x] Launch the weak-candidate-prior B=1.10 probe.
- [x] Launch paired weak-candidate-prior balanced-training probe as a minimal
  2x2 diagnostic.
- [x] Monitor both weak-candidate-prior B=1.10 probes and decide whether either
  repairs calm-window loss without reviving static collapse.
- [x] Add exactly one stronger balanced-prior probe after the weak balanced
  prior improved non-event loss but still missed dynamic baselines.
- [x] Launch stronger balanced-prior probe.
- [x] Monitor stronger balanced-prior probe.
- [x] Run event-density / event-weight threshold analysis from existing
  rollouts before any v18 scenario change.
- [x] Launch v18 event-dominant structural gate after threshold analysis.
- [x] Monitor v18 event-dominant structural gate and decide whether to train PPO.
- [x] Add `event_fraction` support through the split/grid PPO wrappers and fix
  the non-overlapping start selector failure found on first launch.
- [x] Launch the v18 event-dominant single-seed PPO probe on `remote-gpu`.
- [x] Monitor the v18 PPO probe to completion, sync compact metrics, and decide
  whether to replicate seeds 41--45 or redesign again.
- [x] Audit v18 event/calm loss and identify event-window loss as the remaining
  small dynamic-baseline gap.
- [x] Launch one medium event-emphasis v18 PPO probe before any multi-seed
  expansion.
- [x] Monitor the medium event-emphasis v18 PPO probe before any multi-seed
  expansion.
- [x] Run fixed event-pair replay on the v18 balanced source run before
  launching any event-pair teacher.
- [x] Launch one v18 balanced80k seed45 probe before deciding whether v18 is
  optimization-limited or needs another scenario redesign.
- [x] Monitor the v18 balanced80k seed45 probe before deciding
  whether v18 is optimization-limited or needs another scenario redesign.
- [x] Run a switch-limited operational audit on the v18 balanced40k source run
  to separate deployable dwell-limited dynamics from original high-frequency
  dynamic heuristics.
- [x] Add a v19 SPC/laser boundary sensor config and structural gate runner
  after v18 same-setting PPO tuning was exhausted.
- [x] Launch the v19 SPC/laser boundary structural gate on `remote-gpu`.
- [x] Monitor the v19 structural gate, sync compact results, and decide
  whether the boundary change justifies a learned PPO probe.
- [x] Reject v19 as a PPO target after its structural margins came in below
  v18.
- [x] Add and launch one v18 no-candidate-prior PPO ablation to test whether
  the weak prior suppresses event-laser exploration.
- [x] Monitor the v18 no-candidate-prior PPO ablation, sync compact metrics,
  and compare event/calm duty against balanced40k.
- [x] Add and launch one v18 low-AWBC no-prior ablation to test whether the
  strong oracle-greedy imitation loss is keeping the policy SPC-heavy.
- [x] Monitor the v18 low-AWBC no-prior ablation, sync compact metrics, and
  decide whether v18 algorithm tuning is exhausted.
- [x] Close v18 same-scene tuning as exhausted after low-AWBC/no-prior failed.
- [x] Launch a new structural gate instead of more v18 PPO variants.
- [x] Monitor the v20 event-dominant profile-scan structural gate, sync compact
  results, and decide whether any profile justifies a learned PPO probe.
- [x] Add and launch one v20 event-flux reduced PPO diagnostic after the
  profile scan found the strongest overall structural margin there.
- [x] Monitor the v20 event-flux reduced PPO diagnostic, sync compact metrics,
  and compare against static, original dynamic, and duty-constrained baselines.
- [x] Choose the next diagnostic after v20 event-flux PPO failed by the same
  event-side, low-laser/high-SPC transfer mechanism.
- [x] Run direct v20 event-pair replay diagnostics on the completed split-run
  oracle, including the structural laser pair and FC4-heavy action30 variants.
- [x] Run a broader top-auto single-pair replay scan for the remaining
  behavior-valid v20 event-flux dynamic candidates.
- [x] Reject the v20 event-flux branch after replay failed the strict
  best-static, best-deployable-static, and dynamic-baseline gates.
- [x] Add and launch a v21 bursty-event structural gate after closing v20.
- [x] Monitor v21 bursty-event structural gate completion and sync compact
  results.
- [x] Decide whether any v21 profile justifies a learned PPO probe.
- [x] Add and launch a v22 FC4-boundary structural gate after v21 showed event
  timing alone is insufficient.
- [x] Monitor v22 FC4-boundary structural gate completion and sync compact
  results.
- [x] Decide whether any v22 profile justifies a learned PPO probe.
- [x] Add and launch exactly one v22 event-flux reduced PPO diagnostic.
- [x] Monitor v22 event-flux PPO completion and compare against static,
  original dynamic, and duty-constrained baselines.
- [x] Run one direct v22 event-pair replay on the completed split-run oracle
  to separate learned-policy failure from split-oracle structural failure.
- [x] Decide the next structural direction after v22 replay.
- [x] Add v23 met/laser exchange sensor config and structural gate runner.
- [x] Validate and launch v23 structural gate in CPU mode while GPUs are busy.
- [x] Monitor v23 structural gate completion and sync compact results.
- [x] Decide whether v23 justifies any learned PPO probe.
- [x] Add, validate, sync, and launch one v23 dual-flux reduced PPO diagnostic.
- [x] Monitor v23 dual-flux PPO completion and sync compact metrics.
- [x] Audit v23 PPO against static, deployable static, original dynamic, and
  duty-constrained baselines.
- [x] Run one direct v23 event-pair replay on the completed split-run oracle.
- [x] Replay the exact V23 diverse-top cyclic schedule under the split-run
  oracle.
- [x] Decide whether v23 is a policy-learning failure only or a split-oracle
  transfer failure.
- [x] Add cyclic AWBC teacher support for mask-pool imitation.
- [x] Launch one V23 cyclic-teacher reduced PPO probe.
- [x] Monitor V23 cyclic-teacher PPO completion, sync, and audit.
- [x] Launch one stronger cyclic-imitation probe after AWBC0.8 missed best
  duty baseline by `0.000829`.
- [x] Monitor V23 AWBC1.2 cyclic-teacher PPO completion, sync, and audit.
- [x] Choose the next non-coefficient path after AWBC1.2 showed current
  cyclic-teacher PPO is a learnability blocker.
- [x] Add optional agent-cycle phase features and a V23 phase60 AWBC0.8 runner.
- [x] Validate and launch the V23 phase60 AWBC0.8 cyclic-teacher probe.
- [x] Monitor V23 phase60 AWBC0.8 completion, sync, and audit.
- [x] Run same-run exact cyclic replay control on the phase60 split/oracle.
- [x] Choose the next scene/objective branch after closing V23 learned-PPO
  tuning.
- [x] Run a same-run cyclic replay timing sweep to test whether V23 can be
  salvaged without PPO.
- [x] Define that the next branch needs a same-run split-oracle replay gate
  before any further PPO training.
- **Status:** complete

### Phase 14: Robust Same-Run Replay Gate
- [x] Close V23 as a learned-PPO route after phase60 and cyclic timing replay
  failed the same-run duty/deployable reference.
- [x] Identify the correct hard reference: not raw static alone, but the
  split-run `duty_constrained_feasible_static_projected` / best duty
  non-PD-PPO baseline under the same oracle and final-test windows.
- [x] Define a two-stage gate:
  1. TCN structural scan may select a candidate only if the behaviour-valid
     dynamic row beats deployable/duty-static references by a material margin.
  2. Before PPO, create a split-run oracle with `total_timesteps=0` and run
     exact replay on the same oracle/start windows; the replay must beat the
     best deployable/duty reference by at least `0.005` absolute loss or `1%`
     relative, whichever is larger.
- [x] Add or select the next scene/objective candidate under this gate.
- [x] Run the cheap TCN structural screen for that candidate.
- [x] If the screen passes, run the zero-PPO split-oracle replay gate.
- [x] Only if replay passes, launch one reduced PPO diagnostic.
- [x] Monitor the reduced PPO diagnostic and compare learned PD-PPO against
  same-run AoI, duty-constrained, deployable/static, and replay references.
- [x] Monitor the locked `41--45` expansion and decide whether V24 can be
  promoted to the paper mainline without changing the contribution framing.
- **Status:** complete

### Phase 15: Post-V24 Recovery Direction
- [x] Do not promote V24 particle-heavy cyclic-teacher PD-PPO to the paper
  mainline as a learned result.
- [x] Decide whether the next cheap action is a V24 dual/event split-replay
  gate, or whether the branch should move to a new scenario/training mechanism.
- [x] Do not launch more V24 particle-heavy PPO seeds without a new mechanism
  that addresses learned transfer across seeds.
- [x] Monitor V24 dual-flux and event-flux split-replay gates to completion.
- [x] Sync compact gate artifacts and decide whether either profile justifies
  a new learned-policy mechanism.
- [x] Monitor the V24 event-flux reduced PPO diagnostic launched from the
  Phase-15 split-replay winner.
- [x] Monitor the one stronger-imitation V24 event-flux AWBC1.2 diagnostic.
- [x] Reject same-recipe V24 event-flux AWBC coefficient tuning as a paper
  mainline path unless a new training mechanism or contribution framing is
  introduced.
- **Status:** complete

### Phase 16: V24 Event-Flux Phase-Visible Learned Transfer Probe
- [x] Add a V24 event-flux phase24 runner that exposes episode-relative cycle
  phase to the actor for the top2 lead0 dwell12 replay teacher.
- [x] Validate local and remote syntax without changing prior AWBC0.8/AWBC1.2
  output directories.
- [x] Launch exactly one seed45 reduced PPO probe; do not expand seeds unless
  the strict same-run learned gate passes.
- [x] Sync compact artifacts, audit against full-open, best static, deployable
  static, original dynamic, and duty non-PD-PPO references.
- [x] Update progress/findings/CHANGELOG with the result and next decision.
- **Status:** complete

### Phase 17: Strict-Static Replay Contract And V24 Dual Learned Confirmation
- [x] Fix split-replay gate logic so replay-local best static candidates are
  enforced in addition to the source-run baseline reference.
- [x] Re-run strict-static split replay gates for V24 event-flux and dual-flux
  into new output directories without overwriting Phase-15 artifacts.
- [x] Reject V24 event-flux after strict-static replay failure.
- [x] Add V24 dual-flux learned PPO runner(s) from the strict replay winner.
- [x] Launch reduced seed45 learned confirmation for no-phase and/or phase24
  only; do not expand seeds unless a strict learned pass appears.
- [x] Launch locked `41--45` expansion for the only strict learned pass
  (`dual_flux` phase24 AWBC0.8).
- [x] Sync compact artifacts, audit, and update progress/findings/CHANGELOG.
- **Status:** complete

### Phase 18: Post-V24 Learned-Candidate Closure
- [x] Record that V24 dual-flux phase24 seed45 did not replicate under locked
  seeds `41--45`.
- [x] Record that the current V20+ series still has no learned PD-PPO candidate
  that can move into the paper mainline without changing the contribution
  framing or adding a new mechanism.
- [x] Decide the next structural/training mechanism before any more PPO seed
  expansion.
- **Status:** complete

### Phase 19: SCENEBAL-1 Strong-Claim Expansion and True-Static Boundary Audit
- [x] Pivot from V24/TEMPORAL failure modes to SCENEBAL-1 simulator/objective
  balancing while preserving PPO as the final learned scheduler.
- [x] Complete SCENEBAL-1 seeds `93--110` and build corrected 18-seed
  operational, macro, replay, behavior, and true-static aggregates.
- [x] Diagnose and fix the true-static macro scale-mixing issue in
  `scripts/73_v31_collect_oldclaim_gate.py`.
- [x] Diagnose seed95 strict-margin true-static step boundary.
- [x] Start stress-wave waiter `scenebal1_waitfree_111_116_20260621` on
  `remote-gpu`, launching only when all six GPUs are idle.
- [x] Integrate the corrected 18-seed SCENEBAL-1 claim into the canonical ESWA
  manuscript and verify `paper/main.pdf` builds.
- [x] Add a seed-level 18-seed evidence figure to the results section and verify
  the rendered PDF contains the corrected `18/18` and `17/18` claim boundary
  without stale `13/14` or `10/14` wording.
- [x] Fix local SCENEBAL-1 watcher Markdown `printf` errors and restart the
  local monitor/post-collect tmux sessions.
- [x] Add automatic 24-seed decision audit and pre-register next-layer pivot
  designs for simulator/data, oracle/teacher, PPO observation/auxiliary,
  reward/evaluation, and moderate sensor/noise changes.
- [x] Add automatic next-action protocol materialization after the 24-seed
  decision audit, so the watcher writes the bounded next unit instead of only a
  verbal recommendation.
- [x] Verify stress-wave seeds `111--116` launched after all six GPUs became
  idle.
- [x] Extend the post-collect watcher status page with per-seed training
  progress and latest key log lines, not just artifact bitsets.
- [x] Monitor stress-wave seeds `111--116` when GPUs become idle; sync,
  aggregate, and update claim if the wave completes.
- [x] Update the manuscript and seed-level evidence figure from 18-seed to
  24-seed sign-bounded strong-claim evidence.
- **Status:** complete

### Phase 20: SCENEBAL-1 Router-Threshold Strict-Margin Reaudit
- [x] Diagnose the only 24-seed strict-margin true-static step boundary:
  seed `95` has explicit replay step headroom against true fixed static, but
  learned PPO at the original router threshold realizes only a thin margin.
- [x] Run a seed95 eval-only subtype-router confidence scan without retraining
  or changing the sensor geometry.
- [x] Add `scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh` to
  apply router confidence `0.5` uniformly across seeds `93--116`, rerun
  saved-checkpoint evaluation, rerun behavior audit, collect oldclaim/macro
  aggregates, and write a decision audit.
- [x] Sync and launch the 24-seed router-conf0.5 reaudit on `remote-gpu`.
- [x] Monitor completion, sync aggregate outputs, and decide whether the
  strict-margin true-static step gate becomes `24/24` without behavior
  collapse.
- [x] Confirm the bounded unit passed: decision audit is
  `upgrade_allseed_strict`, with operational step/macro, replay step/macro,
  behavior, true-static macro, true-static step sign, and strict-margin
  true-static step gates all `24/24`; minimum strict true-static step margin is
  `0.004242`.
- [x] Parameterize the router-conf0.5 reaudit script so future confirmation
  waves write separate aggregate labels instead of overwriting the 24-seed
  evidence directory.
- [x] If router-conf0.5 fails or only fixes seed95 by weakening other seeds,
  pivot away from router-threshold tuning and into reward/oracle or PPO credit
  assignment; do not try more than ten threshold/redeployment units.
- **Status:** complete

### Phase 21: SCENEBAL-1 Pre-Fixed Router-Conf0.5 Confirmation Wave
- [x] Add a fresh-seed confirmation runner:
  `scripts/run_v31_scenebal1_prefixed_conf05_confirm_117_122_20260621.sh`.
- [x] Keep PPO/PD-PPO, the SCENEBAL-1 simulator/objective, and the
  met-plus-one-specialist sensor geometry unchanged.
- [x] Fix the deployment threshold to router confidence `0.5` before observing
  confirmation results; this tests whether the posthoc seed95 threshold finding
  survives as a uniform deployment protocol on new seeds.
- [x] Sync the confirmation runner and updated collectors to `remote-gpu`.
- [x] Launch seeds `117--122` in tmux when GPUs are idle.
- [x] Monitor completion, sync outputs, and run the independent decision audit.
- [x] If the pre-fixed wave fails a major gate, do not continue threshold
  tuning; pivot to reward/oracle credit assignment, PPO temporal credit, or
  simulator/data generation according to the failure mechanism.
- **Status:** complete_pivot
- **Decision:** The pre-fixed six-seed wave fails true-static step sign on seed
  `122` under the fixed router-conf0.5 deployment protocol, even though
  operational, replay, macro, and behavior gates pass. Do not continue
  router-threshold tuning.

### Phase 22: SCENEBAL-2 Simulator/Data-Balance Pivot Pilot
- [x] Add `scripts/run_v31_scenebal2_seed_sweep_20260621.sh` to create a
  bounded simulator/data-balance variant that preserves PPO as the final
  scheduler and preserves the met+one-specialist sensing geometry as baseline.
- [x] Add `scripts/run_v31_scenebal2_pivot_pilot_117_122_20260621.sh` for the
  first locked pivot pilot on seed `122` plus control seed `117`.
- [x] Sync SCENEBAL-2 scripts and current plan/state files to `remote-gpu`.
- [x] Launch the two-seed SCENEBAL-2 pivot pilot in tmux if GPUs are idle.
- [x] Monitor through PPO, router-conf0.5 evaluation, strict replay, behavior
  audit, and aggregate decision.
- [x] Keep SCENEBAL-2 only if seed `122` recovers true-static step sign while
  preserving operational, replay, macro, and behavior gates. If not, pivot to
  deeper PPO credit/teacher architecture rather than another threshold unit.
- **Status:** complete_breakthrough_pilot
- **Decision:** SCENEBAL-2 recovers the prior seed `122` true-static step sign
  failure and passes all pilot gates on seeds `122` and `117`. Decision audit:
  `reports/aggregate/scenebal2_pivot_conf05_122_117_decision_audit_20260621.json`
  reports `upgrade_allseed_strict`. This is not yet a manuscript-grade strong
  claim; it authorizes a fresh multi-seed confirmation.

### Phase 23: SCENEBAL-2 Six-Seed Fresh Confirmation
- [x] Add `scripts/run_v31_scenebal2_confirm_117_122_20260621.sh`.
- [x] Sync the confirmation runner, pilot report, findings, plan, and
  `research-state.yaml` to `remote-gpu`.
- [x] Launch missing seeds `118--121` in tmux and reuse completed pilot seeds
  `117` and `122` for the final `117--122` aggregate.
- [x] Monitor training, router-conf0.5 evaluation, strict replay, behavior
  audit, and combined decision audit.
- [x] If the six-seed aggregate is all-strict, upgrade SCENEBAL-2 from pilot to
  fresh-confirmed candidate and decide whether to extend to a larger seed set or
  migrate the claim. If it fails, pivot to ORACLE-2/PPO-REGIME-2 according to
  the failure mechanism.
- **Decision:** SCENEBAL-2 six-seed confirmation is all-strict. Decision audit
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_decision_audit_20260621.json`
  reports `upgrade_allseed_strict`; operational step/macro, explicit replay
  step/macro, behavior, true-static macro, true-static step sign, strict-margin
  true-static step, and old-claim step/macro are all `6/6`.
- **Status:** complete_breakthrough_confirmation

### Phase 24: SCENEBAL-2 Expansion And Manuscript-Fit Decision
- [x] Write a paper-fit audit for whether the current scene is over-specialized
  relative to the original PD-PPO claim:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_paper_fit_audit_20260621.md`.
- [x] Add and launch a 12-seed SCENEBAL-2 expansion runner:
  `scripts/run_v31_scenebal2_expand_117_128_20260621.sh`.
- [x] Prepare the conditional manuscript migration patch plan:
  `reports/aggregate/scenebal2_manuscript_migration_patch_plan_20260621.md`.
- [x] Add a parameterised SCENEBAL evidence figure generator and smoke-test it
  on the six-seed SCENEBAL-2 aggregate:
  `paper/figures/gen_fig_scenebal_evidence.py`.
- [x] Add a SCENEBAL decision-audit summary table generator and smoke-test it:
  `scripts/77_v31_write_scenebal_summary_table.py`.
- [x] Monitor tmux `scenebal2_expand_117_128_20260621` and local watcher
  `scenebal2_expand_local_watch_20260621`.
- [x] Extend SCENEBAL-2 beyond six seeds before replacing the existing 24-seed
  SCENEBAL-1 manuscript evidence.
- [x] If the 12-seed SCENEBAL-2 aggregate remains all-strict, prepare a concrete
  manuscript migration patch plan. If it fails, keep SCENEBAL-2 as a diagnostic
  improvement and pivot to oracle/PPO credit assignment rather than more scene
  balancing.
- [x] Generate 12-seed manuscript assets:
  `paper/figures/figure_scenebal2_12seed_evidence.pdf`,
  `paper/figures/figure_scenebal2_12seed_evidence.png`, and
  `paper/tables/scenebal2_12seed_staticnorm_macro_summary.tex`.
- [x] Write the 12-seed breakthrough and paper-fit report:
  `reports/aggregate/scenebal2_confirm_conf05_117_128_12seed_breakthrough_report_20260621.md`.
- **Decision:** SCENEBAL-2 `117--128` is all-strict. Decision audit
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128_decision_audit_20260621.json`
  reports `upgrade_allseed_strict`; operational step/macro, replay step/macro,
  behaviour, true-static macro, true-static step sign, strict-margin
  true-static step, and old-claim step/macro are all `12/12`.
- **Status:** complete_breakthrough_confirmation

### Phase 25: SCENEBAL-2 24-Seed Extension And Manuscript Migration Decision
- [x] Decide whether to keep SCENEBAL-1 24-seed as the main manuscript result and
  add SCENEBAL-2 12-seed as a fresh-confirmed mechanism result, or to extend
  SCENEBAL-2 to 18/24 seeds before replacing the main result.
- [x] If prioritising the strongest ESWA claim, launch the next SCENEBAL-2
  extension wave with router confidence fixed before launch.
- [x] Add and launch the SCENEBAL-2 `117--134` extension runner:
  `scripts/run_v31_scenebal2_expand_117_134_20260621.sh`.
- [x] Monitor tmux `scenebal2_expand_117_134_20260621` and local watcher
  `scenebal2_expand_local_watch_117_134_20260621`.
- [x] If the `117--134` aggregate remains all-strict, generate 18-seed paper
  assets and decide whether to patch the manuscript or extend to 24 SCENEBAL-2
  seeds.
- [x] Confirm the `117--134` aggregate is all-strict:
  operational step/macro, explicit replay step/macro, behaviour,
  true-static macro, true-static step sign, strict-margin true-static step, and
  old-claim step/macro are all `18/18`.
- [x] Generate 18-seed SCENEBAL-2 evidence assets:
  `paper/figures/figure_scenebal2_18seed_evidence.pdf`,
  `paper/figures/figure_scenebal2_18seed_evidence.png`, and
  `paper/tables/scenebal2_18seed_staticnorm_macro_summary.tex`.
- [x] Add 18-seed breakthrough report:
  `reports/aggregate/scenebal2_confirm_conf05_117_134_18seed_breakthrough_report_20260621.md`.
- [x] Add the SCENEBAL-2 `117--140` extension runner:
  `scripts/run_v31_scenebal2_expand_117_140_20260621.sh`.
- [x] Launch tmux `scenebal2_expand_117_140_20260621` and local
  watcher `scenebal2_expand_local_watch_117_140_20260621`.
- [x] Monitor tmux `scenebal2_expand_117_140_20260621` and local watcher
  `scenebal2_expand_local_watch_117_140_20260621` through PPO, router eval,
  strict replay, behaviour audit, and 24-seed aggregation.
- [x] Confirm the `117--140` aggregate is all-strict:
  operational step/macro, explicit replay step/macro, behaviour,
  true-static macro, true-static step sign, strict-margin true-static step, and
  old-claim step/macro are all `24/24`.
- [x] If the `117--140` aggregate remains all-strict, generate 24-seed paper
  assets and replace SCENEBAL-1 as the manuscript main evidence.
- [x] Add 24-seed breakthrough report:
  `reports/aggregate/scenebal2_confirm_conf05_117_140_24seed_breakthrough_report_20260621.md`.
- [x] Patch the manuscript main evidence to SCENEBAL-2 24-seed while keeping the
  claim bounded to the regime-balanced specialist-bottleneck setting.
- [x] Compile `paper/main.pdf` and verify rendered text contains SCENEBAL-2
  24/24 evidence, seed `117`, margins `0.0285`, `0.0710`, `0.0773`, and seed
  range `117--140`.
- **Status:** complete

### Phase 26: Post-Breakthrough Evidence Audit And Paper Packaging
- [x] Run final claim-residual searches over active paper sources and rendered
  PDF.
- [x] Sync paper assets, aggregate outputs, reports, and planning files to
  `remote-gpu`.
- [x] Decide whether any non-scene audit or ablation remains necessary; do not
  expand same-direction seeds blindly after the 24/24 breakthrough.
- **Status:** complete

### Phase 27: Manuscript Claim-Framing And Submission-Fit Audit
- [x] Assess whether the SCENEBAL-2 backbone-plus-one-specialist scene is
  over-specialised relative to the original paper claim.
- [x] If needed, patch introduction/abstract/results wording so the benchmark
  reads as a natural microclimate specialist-budget testbed rather than a
  post-hoc synthetic trick.
- [x] Prepare a concise claim-boundary note for the manuscript and supervisor
  report.
- **Status:** complete

### Phase 28: Final Evidence Package And Supervisor-Facing Summary
- [x] Produce a concise supervisor-facing summary that states the final claim,
  evidence gates, scenario boundary, and remaining limitations.
- [x] Confirm the final paper PDF and report artifacts are synced to
  `remote-gpu`.
- [x] Decide whether the active goal can be marked complete after a
  requirement-by-requirement audit, or whether manuscript polishing remains.
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use a separate plan directory for PD-PPO recalibration | Root, v1, and PD-PPO already have planning files; isolation prevents context recovery from mixing objectives. |
| Keep this fork independent from v1 while still reading v1 records | User explicitly required no further v1/PD-PPO confusion, but clarified that v1's long unsuccessful exploration can still be used as an archived lesson source. v1 may inform what not to repeat; it cannot supply active code, method claims, or main result evidence for this PD-PPO fork. |
| Treat the scene as manually calibrated | The simulator is not a field-calibrated physical deployment; scenario design is part of the research method. |
| Gate before PPO | Full PPO is expensive and uninformative if static remains structurally dominant. |
| Focus dynamic diagnostics on SPC/fc4/context complementarity | Earlier archived scenario screens suggested value through temporal complementarity, not laser gating; current evidence must still come from this PD-PPO fork. |
| Promote `transport_v6`, B=1.10 to remote TCN gate | Local auto-pair linear gate passed with +1.80% overall and +2.05% event margin. |
| Re-enable structural energy-account gating | Instantaneous-budget-only scenes keep producing strong static masks; energy storage/harvest is the next physically meaningful mechanism for adaptive value. |
| Match gate coverage with PPO coverage | Earlier gate/PPO mismatch allowed no-coverage oracle-lift candidates to look promising while PPO evaluated coverage-constrained schedules. |
| Run all further experiments on the server | User explicitly disallowed local experiment runs; local machine is limited to editing, syncing, and result aggregation. |
| Add a separate operational-baseline view | Unconstrained round-robin/AoI heuristics may switch unrealistically often; compare them separately from duty-constrained deployment-style baselines without hiding the original numbers. |
| Test no-warmup only with hard duty correction | Partial no-warmup results often beat static but fail the dynamic-duty target; a reduced hard-duty probe is the fastest check before abandoning this route. |
| Abandon no-warmup + hard duty as a main scene | It fixed dynamic duty in 3/3 seeds but lost to the best original dynamic baseline in 3/3 seeds and to the best constrained baseline in 2/3 seeds. |
| Add switch-limited operational baselines next | Current constrained baselines still switch frequently; the remaining baseline advantage is partly high-frequency scheduling that is operationally unrealistic. |
| Keep all env-dwell12 seeds in reported evidence | Dropping the non-winning seed would be cherry-picking unless a data/config bug is found; current evidence must be reported as 2/3 static, 3/3 dynamic/duty-constrained, and 3/3 behaviour. |
| Treat env-dwell12 as qualified positive, not full static-break completion | It beats dynamic and duty-constrained baseline families in all seeds and satisfies deployment behaviour in all seeds, but does not uniformly or on average beat the strongest static shortcut. |
| Add a direct deployable selected-static comparator | Existing `duty_constrained_feasible_static_projected` was a priority-static variant, not the validation-selected static mask under duty guard. The fair static-shortcut question requires `duty_constrained_validation_selected_static`. |
| Launch h75 only as a symmetric deployment-constraint diagnostic | Seed42's deployable selected static survives by a very small margin and uses high radiometer/SPC duty. A stricter `duty-high=0.75` is a plausible deployment limit, but it must be imposed on PD-PPO and baselines together and all seeds must remain included. |
| Treat h75 as strongest operational branch | h75 gives `3/3` wins against deployable selected static, original dynamic heuristics, and duty-constrained non-PD-PPO baselines with valid behaviour `3/3`; it still does not beat the undeployable original compact static shortcut. |
| Launch h75 seeds 44--45 with no parameter changes | A 3-seed supervisor result is enough to restart writing, but a locked-parameter 5-seed check is the fastest way to test stability without another tuning loop. |
| Use 5-seed h75 as the honest operational result | 5-seed result is stronger evidence than 3-seed despite seed45 boundary: behaviour `5/5`, original dynamic `5/5`, deployable selected static `4/5`, duty non-PD-PPO `4/5`, compact static `3/5`. |
| Do not lock h75 for the stronger static-dominance requirement | The completed h75 10-seed table beats original dynamic `10/10` and best duty non-PD-PPO `9/10`, but deployable selected static only `4/10` with mean delta `-0.000320`. |
| Add v14 instead of reusing v13 | v13 structural gates already failed; v14 targets the specific seed42 static bundle by moving `radiometer+ultrasonic+shielded+SPC` above B=0.65 while preserving met/SPC and met/FC4 alternatives. |
| Stop v14 budget sweeping after partial failure | B=0.60 lost to original/duty dynamic baselines and used always-off channels; B=0.65 seed41 still lost to deployable selected static. More budget seeds would repeat a failed mechanism. |
| Test event-gated PPO before v15 scene rewriting | Rollout audit showed PD-PPO and deployable static had nearly identical event/non-event duty. The next minimal correction is to verify whether the existing PPO architecture can learn event-conditioned switching when its event-gated path is actually exposed through the runner. |
| Move to v15 structural scene physics after v14 event-gated probe | v14 event-gated PPO created event/non-event duty shifts, but still lost to dynamic/duty baselines and made laser structurally impossible at B=0.65. The next lever is scene/cost structure, not more PPO tuning. |
| Compare v15 dynamic headroom against deployable static in structural gates | The existing oracle-lift gate used raw always-on static masks and immediately selected `met+radiometer+laser`; this is useful diagnostic information but not the fair operational static comparator used in the manuscript target. |
| Replace online greedy AWBC with explicit event-pair teacher for the next probe | Medium online-greedy teacher kept balanced duty but lost all fair baselines and still underused event laser/FC4. The next learnability test should imitate the TCN-gate dynamic pair directly. |
| Launch v18 PPO after event-density analysis | Existing v17 rollouts require roughly `0.58--0.63` event-window dominance to beat dynamic baselines. The v18 structural gate raises event dominance and passes with a large behaviour-valid dynamic margin, so one learned-policy probe is justified before any multi-seed expansion. |
| Try medium event emphasis before redesign | V18 balanced weak-prior PPO beats all static families and full-open, with clean behaviour, but loses AoI by only `0.000401` and duty round-robin by `0.002083`. Event/calm audit shows non-event loss is already best, while event loss needs only about `0.0039` improvement to beat the duty baseline if calm performance holds. |
| Reject longer same-setting PPO training on v18 | Balanced80k worsened PD-PPO from `0.411854` to `0.429545`, including worse event and non-event loss, while preserving clean behaviour. The strict dynamic-baseline gap is not fixed by simply doubling timesteps. |
| Treat v18 balanced40k as a qualified operational positive | Balanced40k breaks static families and beats switch-limited/dwell operational dynamics, but still loses original high-frequency AoI and duty-constrained round-robin by small margins. Final evidence must separate those baseline classes. |
| Launch v19 as an SPC/laser boundary gate | V18 learned high SPC duty and low laser duty, while the structural gate's best eligible dynamic needed event-side laser. Raising only SPC cost makes the calm SPC bundle and event laser bundle both tight under B=1.10 without making laser cheaper. |
| Reject v19 after structural gate | V19 still passes, but margins are lower than v18: `0.051004/0.051576` overall/event vs v18 `0.053378/0.055077`. Do not spend PPO on a weaker structural gate. |
| Launch one v18 no-prior ablation | The v18 candidate-prior table ranks SPC/FC4 static masks highest and contains no laser in the top 12 rows, matching the learned high-SPC/low-laser duty failure. Disable the candidate prior once while keeping the rest of balanced40k fixed. |
| Reject v18 no-prior ablation | No-prior worsened PD-PPO to `0.415339`, no longer beat best static, and barely changed event laser duty (`0.131938 -> 0.134668`). Candidate prior is not the main suppressor. |
| Stop v18 same-scene algorithm tuning | Low-AWBC/no-prior worsened PD-PPO to `0.436716` and failed every fair gate; event emphasis, replay teacher, longer training, no-prior, and low-AWBC all failed. |
| Launch v20 profile-scan structural gate | With v18 same-scene PPO tuning exhausted and v19 weaker than v18, scan existing v7 target profiles under the v18 event-dominant geometry before spending more GPU time on PPO. |
| Launch one v20 event-flux PPO diagnostic | `event_flux_particle_v7` is the only scanned profile with overall structural margin above v18 (`0.063723`), but its event margin is lower than v18, so run exactly one reduced PPO probe and keep acceptance strict. |
| Reject v20 event-flux PPO | Learned PD-PPO is valid but loses best static, deployable static, original dynamic, and duty dynamic baselines; event loss and low event-laser/FC4 duty remain the failure mechanism. |
| Close v20 event-flux after direct replay | Direct event-pair replay improved over learned PPO but still failed strict gates: best top-auto pair `0.400381` loses best static `0.398205`, best deployable static `0.400316`, original round-robin `0.397568`, and duty round-robin `0.396908`. |
| Launch v21 bursty-event structural gate | V20 replay showed the same event-dominant geometry does not transfer even with direct event pairs. V21 changes event geometry to shorter, stronger, separated bursts before spending any further PPO time. |
| Reject v21 bursty-event gate | Only `particle_heavy_flux_v7` formally passes, and it has negative event margin; `event_flux_particle_v7` has positive event margin but negative overall margin. Do not launch PPO. |
| Launch v22 FC4-boundary gate | V20 artifact audit showed remaining static shortcuts include FC4-heavy masks, especially `met+radiometer+ultrasonic+fc4`; v22 raises FC4 cost while holding the stable event-dominant geometry fixed. |
| Launch one v22 event-flux PPO diagnostic | V22 event-flux is the strongest structural point with overall margin `0.059582` and event margin `0.044922`. The static reference is still laser-based, so this is diagnostic and acceptance remains strict. |
| Reject v22 learned PPO diagnostic | The completed policy has valid duty (`mid=8`, no always-on/off sensors, zero aborts) but loses every strict comparison: best static, deployable selected static, best deployable static, original dynamic, duty dynamic, and full-open. Event/calm and duty audits show low event laser/FC4 transfer. |
| Close v22 after direct replay | Direct event-pair replay improves to `0.396653` but still loses best static `0.394480`, deployable selected static `0.394044`, and best deployable static `0.393007`; static-mask replay shows pure laser static is weak (`0.420640`) while action 2 `met+radiometer+surface+SPC` is the blocker (`0.394668`). |
| Launch v23 met/laser exchange gate | Raising met and lowering laser by matched amounts makes action 2 steady power `1.11` while preserving action 7 calm `0.94` and action 15 event `1.10`; this directly targets the final-eval shortcut found by v22 replay. |
| Launch exactly one v23 dual-flux PPO diagnostic | The v23 gate passed for all scanned profiles and broke the laser/static shortcut. `particle_heavy_flux_v7` had the largest margin but its best dynamic row used `always_on=1`, `always_off=2`; `dual_flux_particle_v7` had a smaller positive margin but the cleanest behaviour-valid row (`mid=8`, no always-on/off sensors), so it is the only justified learned PPO probe. |
| Reject v23 learned PPO before seed expansion | Seed45 is behaviour-clean and beats static, but loses best deployable static / duty non-PD-PPO by `0.010531` and AoI by `0.001611`; learned duty remains low on laser/FC4. |
| Keep V23 and fix learnability next | Exact cyclic diverse-top replay transfers under the same split oracle (`0.437728`) and beats best duty/static/dynamic references. The scene has headroom; current PPO/AWBC cannot imitate the required mask-pool policy. |
| Add cyclic teacher rather than change scene | The needed behaviour is a lead-6 dwell-12 top5 mask-pool cycle, not a single event pair. A narrow `event_cyclic` AWBC mode directly tests learnability without changing the validated V23 scene. |
| Try one stronger cyclic-imitation probe | AWBC0.8 learned a valid policy that beats static and AoI but misses best duty static by only `0.000829`; event loss is the only remaining gap and the learned top-mask distribution is less faithful than exact replay. |
| Close V23 learned-PPO tuning | AWBC0.8, AWBC1.2, phase60, and minor cyclic timing variants all failed the strict same-run duty/deployable reference. The phase60 exact replay control also lost `duty_constrained_feasible_static_projected` by `0.000212`, so V23's strict margin is too oracle-sensitive for seed expansion. |
| Require split-oracle replay before PPO | V23's standalone TCN structural gate reported a `0.030123` dual-flux margin, but the same mechanism nearly vanished under the actual split-run oracle. Future candidates must pass replay on the exact split oracle/start windows before spending GPU time on PPO. |
| Launch V24 particle-heavy PPO after Phase-14 gate | V24 `particle_heavy_flux_v7` passed Stage-1 and its same-run split replay gate: `split_top2_l6_dwell12=0.414078` vs same-run AoI reference `0.429470`, margin `0.015392` absolute / `3.58%` relative, with `mid=8`, zero always-on/off, and switch `0.043712`. |
| Reject V24 particle-heavy cyclic-teacher as paper-mainline learned result | Locked seeds 41--45 kept valid behaviour `5/5`, but strict learned wins were only `1/5` against best original dynamic and `1/5` against best duty non-PD-PPO; mean deltas were negative for best deployable static (`-0.012423`), best original dynamic (`-0.014409`), and best duty non-PD-PPO (`-0.011868`). |
| Run V24 dual/event split-replay gates before any new PPO | V24 Stage-1 passed for `dual_flux_particle_v7` and `event_flux_particle_v7`, but particle-heavy learned transfer failed multi-seed. The cheapest non-redundant check is whether the other two profiles survive the stricter same-run split-oracle replay gate. |
| Launch one V24 event-flux learned diagnostic | Both dual/event split-replay gates passed, but `event_flux_particle_v7` has the stronger Phase-15 margin (`0.010099` vs `0.007295`) and a simpler `lead=0,dwell=12` top-2 teacher. This justifies one reduced seed45 PPO diagnostic, not a seed expansion. |

| Treat SCENEBAL-2 12-seed as a natural manuscript candidate, not a universal claim | The `117--128` aggregate is all-strict with clean behaviour, but the scenario is a regime-balanced backbone-plus-one-specialist benchmark. It should be framed as a specialist-bottleneck microclimate result, not as proof that PD-PPO universally beats fixed static scheduling. |
| Prefer SCENEBAL-2 18/24-seed extension before replacing the main paper result | SCENEBAL-2 has fresher all-strict 12-seed evidence, while the current manuscript result has 24 seeds from SCENEBAL-1. Extending SCENEBAL-2 keeps the better framing and improves sample-size symmetry before a main-result replacement. |
| Promote SCENEBAL-2 24-seed to main manuscript result | It preserves 24-seed scale and passes all strict gates `24/24` with minimum true-static step margin `0.028498`; the specialist-bottleneck mechanism is cleaner and more natural than SCENEBAL-1 for the paper's final claim. |

## Acceptance Gate
| Criterion | Target |
|----------|--------|
| Feasible candidate diversity | Enough static masks for nontrivial selection, not one forced subset |
| Laser shortcut | Best static should not be a laser-dominant shortcut; top-5 static laser fraction should be low |
| Structural dynamic headroom | TCN gate: behaviour-valid dynamic schedule beats deployable/duty static by a material margin, not only raw static |
| Split-oracle replay headroom | Pre-PPO split-run replay: exact dynamic schedule beats the same-run `duty_constrained_feasible_static_projected` / best duty non-PD-PPO reference by at least `0.005` absolute loss or `1%` relative, whichever is larger |
| Event mechanism | Improvement should appear in event loss, not only in calm windows |
| Runtime feasibility | No excessive guard drops, energy deficits, or warmup aborts in gate diagnostics |
| Dynamic duty | At least several sensors have intermediate duty, with no collapse to multiple always-on or always-off sensors |
| Switching realism | Nonzero dynamic switching, but not high-frequency thrashing |
| Hard behavioral filter | Prefer `mid_duty_sensor_count >= 5`, `always_on_sensor_count <= 1`, `always_off_sensor_count <= 1`, and nonzero bounded switching; stricter interpretation for final evidence is no multiple always-on/off sensors |

## Current Candidate
| Field | Value |
|-------|-------|
| Profile | No current learned PD-PPO candidate is eligible for paper-mainline migration. V24 event-selective laser `particle_heavy_flux_v7` and `dual_flux_particle_v7` both produced useful single-seed diagnostics but failed locked learned multi-seed validation. |
| Budget | Stage-1 gate uses B=`1.10`, h=`0.82`, startup peak `1.55`. |
| Startup peak | `1.55` |
| Scene | Active Stage-1 scene `windblown_sensors_physical_event_v24_event_selective_laser.yaml`: keep V23 power boundary, degrade non-event laser particle noise, preserve event laser fidelity/availability. |
| Target weights | Scanning `particle_heavy_flux_v7` = `0.03 0.03 0.10 0.01 0.01 0.0 16.0 22.0 22.0`, `event_flux_particle_v7` = `0.03 0.03 0.10 0.01 0.01 0.0 30.0 12.0 12.0`, and `dual_flux_particle_v7` = `0.03 0.03 0.10 0.01 0.01 0.0 22.0 16.0 16.0` |
| Target scales | `5.0 5.0 5.0 1.0 1.0 100.0 0.0001 0.2 5.0` |
| Constraints | Default next-gate constraints remain max active `4`, dwell `12`, duty low/high `0.12/0.75`, harvest `0.82`, capacity `180`, reserve `20`, SOC buffer `40`, SOC penalty `0.08` unless the new scene explicitly justifies changing them symmetrically for all policies. |
| Teacher | Last reduced PPO used `event_cyclic` AWBC from the strict V24 dual-flux split-replay winner: calm pool `radiometer_basic+surface_temp_ir+shielded_thermo_hygro+laser_disdrometer` and `radiometer_basic+shielded_thermo_hygro+snow_particle_counter`; event pool `surface_temp_ir+shielded_thermo_hygro+fc4_flux` and `surface_temp_ir+shielded_thermo_hygro+snow_particle_counter`; lead `0`, dwell `12`, phase period `24`, AWBC `0.8`. |
| Active pilot | None. V24 dual-flux phase24 seed45 passed every same-run reference, but locked seeds `41--45` failed strict learned replication. Do not expand more V24 same-recipe PPO seeds. |
| Output | Latest V24 dual-flux phase24 locked result: `reports/v31_static_break_v24_event_selective_laser_dual_flux_dwell12_ppo_seed45_b1p10_h082_cyclicteacher_awbc0p8_phase24_eventfraction_20260620/v24_eventlaser_dualflux_cyclicteacher_awbc0p8_phase24_seeds41_45_h082_eventfraction_summary.csv`. Strict replay output: `reports/v31_static_break_v24_event_selective_laser_dual_flux_particle_v7_split_replay_gate_strict_static_ref_seed45_h082_20260620`. |
| Last structural gate | V23 met/laser exchange completed with all profiles passing: `particle_heavy_flux_v7` margin `0.058551` but `always_on=1`, `always_off=2`; `dual_flux_particle_v7` margin `0.030123`, event margin `0.022259`, `mid=8`, no always-on/off; `event_flux_particle_v7` margin `0.028069`. |
| Active structural scan | None. V24 particle-heavy, event-flux, and dual-flux follow-ups have all been resolved under the corrected strict-static replay / learned replication contract. |
| Scan output | V24 event-flux was rejected after replay-local static reference enforcement; V24 dual-flux survived strict replay but failed locked learned replication. |
| Caveat | A TCN structural pass alone is no longer sufficient. V23 proved that a separate structural oracle can overstate the headroom that remains under the actual split-run oracle used by PPO. |

## Current Failure Mode
| Failure | Evidence | Required Correction |
|---------|----------|---------------------|
| Single-sensor frozen-oracle shortcut | `custom_ppo` selected `snow_particle_counter` for `99.66%` of steps and left 7/8 sensors always off while beating static on oracle loss | Add action-level coverage or minimum-active constraints so oracle loss cannot be improved by an almost-static one-sensor policy |
| Static headroom absent after duty fix | balanced v7 run achieved `mid=7`, `always_on=0`, `always_off=1`, but `custom_ppo` loss `0.13034` still lost to feasible static `0.12253` and round-robin `0.12923` | Re-run structural scene/cost/objective gate with dynamic-duty constraints before launching more PPO seeds |
| PPO replication unverified on v10 | v10 B=0.65 PPO seed 41 passed the dynamic-duty and baseline-comparison target, but this is still one seed and B=0.65 narrowly missed the strict TCN margin | Complete B=0.70 PPO seed 41, then replicate the better operating point on seed 42 before expanding |
| Structural gate did not guarantee PPO transfer | v10 B=0.70 passed the strict TCN dynamic gate, but learned PPO loss `0.16170` lost to validation-selected static `0.14722`, AoI `0.15631`, feasible static `0.16009`, and round-robin `0.16148` | Treat B=0.70 as a failed learned-policy transfer; replicate B=0.65 and inspect whether training stability or budget tightness drives the difference |
| B=0.65 seed 42 did not replicate | PD-PPO `0.13797` lost to validation-selected static `0.12743`; duty also failed the hard target with `always_on=1`, `always_off=2` because `met_station_core` was nearly unused and laser was off | Micro-calibrate the v10 costs so the static `radiometer+ultrasonic+shielded+SPC` shortcut is infeasible while keeping met/SPC and met/FC4 dynamic alternatives feasible |
| Soft duty feedback is insufficient | Seed 42 kept met nearly off and radiometer nearly on despite `duty_score_feedback=2.5` | Add hard duty guard in the action-score path: after grace, low-duty sensors receive a strong positive score and high-duty sensors receive a strong negative score before projector feasibility |
| Event-window evaluation is insufficient | Seed43 event-window eval reached event rate `0.34408`, but PD-PPO `0.16880` still lost to feasible static `0.15500` and round-robin `0.15951` | Return to scene structure: weaken static bundles directly rather than tuning final-test start selection |
| Residual env-dwell12 static shortcut | Mainline env-dwell12 beats original dynamic `3/3` and duty-constrained non-PD-PPO `3/3`, but only beats strongest static `2/3`; seed42 validation-selected static remains lower (`0.138138` vs PD-PPO `0.149620`) | Do not delete seed42; inspect the selected static mask and either make the static deployability comparison fairer or recalibrate the scene so static masks cannot satisfy the target without dynamic complementarity |
| Direct deployable selected-static still barely survives in seed42 | `duty_constrained_validation_selected_static` has valid duty (`mid=8`, `always_on=0`, `always_off=0`) and losses `0.135529`, `0.149349`, `0.142127`; PD-PPO wins seeds 41/43 but loses seed42 by only `0.000271` | Static shortcut is mostly broken by deployable duty constraints but not fully eliminated. Next highest-ROI test is a stricter duty-high setting if it is imposed symmetrically on PD-PPO and baselines. |
| h75 diagnostic launched | A symmetric `duty-high=0.75` reduced retrain is now running on GPU5 for seeds 41--43 | Treat result as a structural diagnostic unless it wins all relevant families without hurting deployment behaviour. |
| h75 resolves deployable-static comparison | Final h75 results: deployable selected static `3/3`, best original dynamic `3/3`, best duty non-PD-PPO `3/3`, behaviour `3/3`; original compact static only `1/3` | Use h75 for operational constrained scheduling claims. Keep original compact static as a diagnostic static-shortcut row, not the fair deployment baseline. |
| V15 medium teacher failed | `custom_ppo=0.293273` lost to validation-selected static `0.272686`, deployable selected static `0.288991`, best original dynamic `0.283346`, and best duty non-PD-PPO `0.279821`; behaviour was balanced but warmup aborts were 5 | Do not expand medium-teacher PPO. Test explicit event-pair imitation from the TCN structural gate. |
| Dwell36 is not the right abort fix | `eventpair4_dwell36` achieved valid balanced duty and reduced aborts to 1, but `custom_ppo=0.290391` lost to validation static `0.275914`, feasible static `0.276809`, round-robin `0.282384`, and duty-constrained round-robin `0.287921` | Do not keep increasing dwell. The failure is loss collapse from over-smoothing, not residual thrashing. |
| Abort source is energy reserve, not event timing | In the stronger dwell12 eventpair4 run, aborts occurred when SOC was near reserve (`20-21`) and mean power `0.805` exceeded harvest `0.65` | Test a minimal energy-account recalibration before retraining: increase harvest only enough to avoid reserve clipping. |
| Exact harvest sweep selected `h=0.74` | Exact ep4 replay: `h=0.74` gives loss `0.277467`, abort `0`, `mid=8`, switch `0.033425`; `h=0.75` is similar, while `h=0.72` is worse and `h=0.65` still aborts | Launch a single-seed PPO probe with identical settings to eventpair4 except `harvest_per_step=0.74`. |
| h0.74 learned PPO is one abort short | h0.74 PD-PPO beats deployable selected static, best deployable static, original dynamic, and duty non-PD-PPO, but has `warmup_abort_count=1` | Launch h0.75 with no other parameter changes; accept only if zero-abort behaviour is restored without losing fair-baseline wins. |
| h0.75 failed | h0.75 PD-PPO has `warmup_abort_count=2` and loses to best original/duty dynamic baselines | Return to h0.74 and strengthen event-pair imitation (`awbc_coef=0.40`) rather than further relaxing harvest. |
| h0.74/AWBC0.40 is strongest loss-positive branch so far | PD-PPO `0.278159` beats feasible static, original dynamic, best duty non-PD-PPO, deployable selected static, and best deployable static, but has `warmup_abort_count=1` | Test h0.75/AWBC0.40. If it clears abort while preserving loss wins, replicate seeds; otherwise consider a reserve-aware/SOC-soft penalty rather than more harvest. |
| h0.75/AWBC0.40 passed seed41 | PD-PPO `0.277030`, `warmup_abort_count=0`, `mid=8`, and wins against feasible static, deployable static, original dynamic, and duty non-PD-PPO | Replicate seeds 42--43 with locked parameters before any further tuning. |
| h0.75/AWBC0.40 failed replication | Seed42/43 kept dynamic duty (`mid=8`) but had aborts `4/3`; combined result only beat deployable selected static `2/3`, best deployable static `1/3`, and original dynamic `2/3` | Do not expand this branch. Exact teacher replay shows seed42 is structurally blocked by a residual `met+surface+laser` shortcut. |
| V16 surface-boundary gate launched | Raising only `surface_temp_ir` cost makes `met+surface+laser` infeasible while preserving `met+radiometer+laser` and the calm pair | If seed42 structural gate shows deployable-static headroom, launch a PPO probe; otherwise continue scene design rather than PPO tuning. |
| V16 micro_flux TCN gate failed | Dynamic margin was `-0.000404`; best deployable static shifted to `radiometer+surface+shielded+fc4` | Do not launch PPO on micro_flux v16. Test `micro_particle_v6` objective before adding more sensor-cost changes. |
| V16 micro_particle TCN gate failed narrowly | Best unrestricted dynamic beat static but violated behaviour (`always_off=3`); best eligible dynamic lost by `0.000133` absolute | Try particle microstructure decorrelation before PPO. |
| Structural gate deployable static was over-strong | Deployable static rows switched at `0.37-0.44/step`, unlike the dwell12 deployment setting | Added `--env-min-dwell-steps` to scripts `49` and `63`; rerun v16 micro_particle with dwell12 before further scene changes. |
| Corrected v16 micro_particle dwell12 gate passed | Dynamic margin `+0.022003`, event margin `+0.021998`; best eligible dynamic `auto_non10_event20` | Launch one PPO seed42 probe with the same scene/profile and event-pair teacher from the gate. |
| V16 dwell12 PPO seed42 needs energy-account correction | PD-PPO loss `0.409595` beats static, dynamic, and duty baselines with `mid=8`, no always-on/off, and `26` masks, but `warmup_abort_count=6`; mean power `0.9028` exceeds harvest `0.75` | Launch h0.92 single-seed probe before any seed replication. |
| H0.92 correction failed as training change | h0.92 retraining gave zero abort but `custom_ppo=0.415797`, losing to duty round-robin `0.411874`; h0.75 checkpoint replay at h0.92 also lost (`0.415615`) | Keep h0.75 but add reserve-aware SOC soft penalty and stronger abort penalty so PPO learns the guard behaviour cleanly. |
| Replay-only h0.81--0.83 did not fully pass | h0.81/h0.83 lost to duty-constrained round-robin; h0.82 lost to AoI, despite zero abort and static wins | Retrain directly at h0.82 with the same reserve-aware controls instead of replaying an h0.75 policy. |
| H0.82 direct retraining passed seed42 | `custom_ppo=0.409735`, zero abort, `mid=8`, no always-on/off, and wins original dynamic, duty-constrained dynamic, feasible static, and deployable selected static | Replicate locked h0.82 settings on seeds 41 and 43. |
| H0.82 failed locked replication | Combined seeds 41/42/43: PD-PPO wins strongest static only `1/3`, deployable static `1/3`, original dynamic `2/3`, duty dynamic `2/3`; behaviour is clean `3/3` with zero abort and all sensors mid-duty | Static shortcut is still structural, not an energy-account issue. Next step must change scenario/objective or training target around event-window complementarity; do not add more h0.82 seeds before a structural fix. |
| Multi-seed structural screen launched | Added v7 flux+particle target profiles and launched seed41/42/43 TCN gate on v16 with dwell12 | If no profile passes all three seeds, change scene generation or sensor uniqueness before PPO; if one passes, launch a single PPO probe on the weakest passing seed. |
| AWBC teacher mismatch suspected | Seed41 structural gate best eligible dynamic uses `auto_non14_event15`; event mask 15 is `met_station_core|radiometer_basic|laser_disdrometer`, while previous PPO teacher used `met+radiometer+surface+fc4` | Launched an isolated seed41 PPO probe that keeps all h0.82 settings but changes the teacher to calm `surface+ultrasonic+shielded+SPC` and event `met+radiometer+laser`. |
| Teacher-aligned PPO worsened seed41 | Laser-event teacher produced `custom_ppo=0.347668`, worse than old h0.82 `0.331129`, with one abort and losses to static, dynamic, and duty baselines | Do not increase AWBC or continue teacher-only fixes. Wait for multi-seed structural screen; if margins remain around 1%, use stronger scenario/objective separation or an oracle-prior training change rather than imitation-only tuning. |
| PPO final-test selection was not aligned with the structural gate | Structural gates use `event_transport_rich` windows, while split-protocol PPO final-test previously used uniform non-overlapping starts | Added event-rich final-test selection to scripts `58` and `59`, and launched a seed41 h0.82 probe that changes only final-test selection. |
| Dual flux+particle 5-seed expansion failed | Seeds 41--43 beat deployable static, but seeds 44--45 lose every baseline family while keeping clean duty behaviour; combined deployable-static win count is only `3/5` with mean delta `-0.007197` | Do not promote the 3-seed result as stable. Next experiments must address residual duty-valid laser/static shortcuts and calm-window loss, not add more seeds unchanged. |
| Particle-heavy profile selected as next main route | Targeted structural screen shows `particle_heavy_flux_v7` is strongest in seeds 42--44 and passes seed45 while `dual_flux_particle_v7` fails seed45 | Shift learned experiments to particle-heavy plus adaptive `oracle_greedy`; keep dual oracle-greedy only as a mechanism diagnostic. |
| Use the independent particle-heavy route plan as the recovery entry point | The active path has moved beyond generic static-break exploration. The new route file records the v1 boundary, current candidate, gates, fallback v17 simulator plan, and execution order. |
| Dual oracle-greedy teacher is insufficient | Seed44 dual oracle-greedy improves loss from `0.411077` to `0.372997` and beats best duty non-PD-PPO, but still loses deployable static `0.368683` due calm-window loss | Do not continue dual-profile tuning unless particle-heavy also fails. |
| V16 particle-heavy oracle-greedy seed45 failed the learned gate | PD-PPO `0.432414` has valid behaviour (`mid=8`, no always-on/off, abort `0`, switch `0.038187`) and beats deployable selected static `0.436687`, but loses best deployable static `0.431815`, round-robin `0.418746`, and raw feasible static `0.391799` | Do not expand this PPO branch. Run a corrected v17 structural gate with `particle_heavy_flux_v7`, h0.82, and env dwell 12 before more training. |
| Corrected v17 particle-heavy B=1.15 gate failed narrowly | Best behaviour-valid dynamic `dynamic:met_context__event_thermal_flux` lost to deployable static by `-0.001091` overall while event margin was slightly positive `+0.000439`; best any dynamic `dynamic:snow_core__event_laser_fc4` won (`0.648781` vs deployable static `0.663613`) but collapsed behaviour (`mid=3`, `always_on=2`, `always_off=3`) | Do not relax the behaviour gate for final evidence. Run a targeted B=1.05/1.10/1.20 scan to determine whether a nearby budget yields behaviour-valid dynamic headroom before changing PPO again. |
| V18 same-setting PPO tuning exhausted | Balanced40k breaks static families but misses original dynamic baselines narrowly; event-emphasis, fixed event-pair replay, and balanced80k all worsened or failed to transfer | Do not launch more same-scene v18 PPO probes unless the claim is explicitly limited to switch-limited operational dynamics. For strict dynamic dominance, move to a new scenario/objective gate. |
| V19 SPC/laser boundary gate launched | V18 learner overuses cheap SPC and underuses event laser; v19 raises only SPC cost from `0.52/0.68` to `0.62/0.83`, making calm SPC and event laser bundles similarly tight | Monitor the structural gate before any PPO. Launch learned training only if deployable-static headroom and dynamic-diversity gates improve. |
| V19 did not improve on v18 | V19 gate pass margins are slightly lower than v18 and the best dynamic remains `auto_non14_event15` | Do not launch v19 PPO. Test whether v18's weak candidate prior is suppressing event-laser exploration. |
| No-prior did not improve v18 | No-prior worsened event loss and barely changed laser duty, so candidate prior is not the main event-laser suppressor | The remaining controlled algorithmic test is lowering `awbc_coef`; if that fails, stop v18 algorithm tuning. |
| Low-AWBC did not improve v18 | Low-AWBC/no-prior collapses non-event quality and fails every fair baseline | Do not run more v18 PPO variants; move to a new structural gate. |
| V20 replay did not transfer | Best direct v20 event-pair replay is `eventflux_auto_non2_event15_l0=0.400381`; it beats deployable selected static but loses best deployable static, best static, original round-robin, and duty round-robin | Close v20. The next strict-dominance attempt must change scene/objective structure beyond the same event-dominant v18 geometry, not add another PPO recipe. |
| V21 bursty geometry failed | `particle_heavy_flux_v7` passes overall but has event margin `-0.023047`; `event_flux_particle_v7` has event margin `0.006708` but overall margin `-0.010957`; `dual_flux_particle_v7` fails both | Do not launch PPO. A viable next branch must create positive overall and event headroom together, likely through sensor/noise structure rather than event timing alone. |
| V22 learned PPO failed despite valid duty | `custom_ppo=0.411906` with `mid=8`, zero aborts, and no always-on/off sensors, but it loses best static `0.394480`, deployable selected static `0.394044`, best deployable static `0.393007`, round-robin `0.401172`, and best duty non-PD-PPO `0.393007`; event laser duty remains only `0.122384` | Do not run more same-recipe PPO. Run a direct v22 event-pair replay to test whether the structural pair transfers under the actual split-run oracle before changing scene structure again. |
| V22 direct replay still loses static | Best replayed pair `non2/event15_l0=0.396653` beats learned PPO and round-robin but loses best static and deployable static; static-mask replay shows action 2 `met+radiometer+surface+SPC=0.394668` is the shortcut, while pure laser static is weak at `0.420640` | Close v22 and test v23 met/laser exchange, which makes action 2 infeasible while preserving action 7 and action 15. |
| V23 structural gate passes but needs learned confirmation | V23 breaks the action-2 shortcut and passes all scanned profiles; dual-flux has the cleanest behaviour-valid dynamic row (`dynamic:diverse_top5_lead6_dwell12`, loss `0.380097`, event `0.527918`, non-event `0.227583`, `mid=8`, no always-on/off) | Run exactly one reduced PPO diagnostic on dual-flux seed45 before any multi-seed expansion. |
| V23 learned PPO fails strict dynamic/duty gates | `custom_ppo=0.449127` with valid behaviour beats static but loses `aoi=0.447516` and `duty_constrained_feasible_static_projected=0.438596`; laser/FC4 duties remain `0.140625/0.128662` | Do not expand seeds. Run a direct event-pair replay on the same split oracle to determine whether the structural dynamic candidate transfers. |
| V23 cyclic replay passes strict split-oracle gate | `v23_dual_diverse_top5_l6_dwell12=0.437728`, event `0.557965`, non-event `0.298486`, zero aborts, `mid=8`, no always-on/off sensors; beats best duty/deployable static by `0.000868` | Add cyclic AWBC teacher support and run exactly one reduced PPO probe before any seed expansion. |
| V23 AWBC0.8 cyclic teacher is a near miss | `custom_ppo=0.441380`, valid behaviour, wins static/AoI, but loses best duty/deployable static `0.440551` by `0.000829`; learned top mask is too concentrated (`42.48%`) | Run one stronger cyclic-imitation probe; if it still misses, treat learnability as the blocker rather than launching more seeds. |
| V23 AWBC1.2 cyclic teacher still misses | `custom_ppo=0.440397`, valid behaviour, wins static/AoI/duty-AoI, but loses best duty/deployable static `0.436732` by `0.003665` and duty round-robin `0.439321` by `0.001076`; event loss worsens to `0.580687` | Stop same-recipe AWBC coefficient tuning and do not expand seeds. V23 is structurally valid by exact cyclic replay, but current PPO transfer is the blocker. |
| V23 phase60 probe launched | Exact cyclic replay depends on `(current_idx - episode_start_idx) mod 60`; previous learned probes did not expose this state directly | Run one phase-aware AWBC0.8 probe. If it still misses strict duty/deployable references, phase hiding is not the sole learnability blocker. |
| V23 phase60 closes learned tuning | Phase-aware `custom_ppo=0.447119` loses static, AoI, duty round-robin, and duty feasible static; same-run exact cyclic replay `0.437319` is clean but loses same-run duty feasible static `0.437106` by `0.000212` | Close V23 learned-PPO tuning. The exact dynamic margin is too small / oracle-sensitive, so do not launch more V23 PPO variants or seed expansion. |
| V23 cyclic timing sweep failed | Best minor timing variant `phase60_top5_l3_dwell12=0.439674` is clean but still loses same-run duty feasible static `0.437106` by `0.002568` | Do not salvage V23 with timing tweaks. Next branch must first pass a same-run exact dynamic replay gate with a materially larger margin. |
| V24 particle-heavy learned replication failed | Locked seeds `41--45` are behaviour-clean `5/5`, but strict learned wins are only `1/5`; mean deltas versus best deployable static, best original dynamic, and best duty non-PD-PPO are all negative | Do not promote V24 particle-heavy to the paper mainline and do not add more same-recipe seeds. |
| V24 dual/event split-replay gates passed | Dual replay margin is `0.007295`; event-flux replay margin is `0.010099`, both with clean top2 lead0 dwell12 behaviour | Event-flux justified exactly one new learned diagnostic because it had the larger replay margin and the simpler teacher. |
| V24 event-flux AWBC0.8 near miss | `custom_ppo=0.418312` is behaviour-clean and barely beats the best duty/deployable non-PD-PPO reference by `0.000134`, but loses full-open, best static, and AoI | Run one stronger-imitation probe only; do not expand seeds from this near miss. |
| V24 event-flux AWBC1.2 closed the same-recipe path | `custom_ppo=0.436344` is behaviour-clean and wins full-open/AoI, but loses selected static by `0.024201`, deployable selected static by `0.010824`, and best duty non-PD-PPO by `0.003587` | Stop AWBC coefficient tuning for this V24 event-flux recipe. There is no learned PD-PPO result in the current V20+ series that can move into the paper mainline without a new mechanism or reframing. |
| Test V24 event-flux phase24 before abandoning V24 learned transfer | V24 event-flux has a materially stronger split-replay margin (`0.010099`) than the V23 phase60 control that failed (`0.000212`), and its winning teacher is top2 lead0 dwell12 | Run exactly one phase-visible AWBC0.8 probe with period `24` and dwell `12`; this is a distinct learnability test, not another AWBC coefficient sweep. |
| Correct the split-replay gate reference contract | V24 event-flux appeared to pass because the old gate referenced AoI, but replay-local static action8 was stronger (`0.403818`) than best replay (`0.406600`) | Future split-replay gates must enforce both the source-run reference and the replay-local best static candidate. |
| V24 dual-flux survives the corrected strict-static replay gate | Strict dual replay has best replay `0.410668`, source reference `0.417963`, replay-local best static `0.418077`, and static margin `0.007409` | Close event-flux and launch learned confirmation only for dual-flux. |
| V24 dual-flux phase24 failed locked learned replication | Locked seeds `41--45` remained behaviour-clean `5/5`, but strict win counts were best static `1/5`, deployable selected static `2/5`, best deployable static `2/5`, original dynamic `2/5`, and best duty non-PD-PPO `1/5`; mean deltas were negative for every strict static/dynamic/duty reference except full-open | Do not promote any V20+ learned PD-PPO result to the paper mainline. The next step requires a new structural/training mechanism, not additional same-recipe V24 seed expansion. |

### Phase 29: State-dependent physical-resource route
- [x] Run a vectorized heater/effective-power screen over the four V541 traces.
- [x] Refine the screen in the low-budget region and select B=2.15 for geometry.
- [x] Run corrected 32-subset frozen forecast geometry at B=2.15.
- [x] Run the initial online observability/transfer probe without training PPO.
- [x] Increase chronological training coverage once to distinguish sample
  scarcity from intrinsic transfer failure.
- [x] Close the route before PPO after the transfer gate remained negative.
- **Status:** complete; physics and forecast geometry passed, online transfer
  failed, and no PPO was launched.

### Phase 30: Heater-dependent quality truth regeneration
- [x] Audit the V541 truth/resource linkage after the V558 transfer failure.
- [x] Confirm that V541 resource state was not coupled to channel quality.
- [x] Declare the heater-quality relation and repair its generator schema.
- [x] Generate four corrected truth traces for seeds `7241--7244`.
- [ ] Rebuild frozen forecasting assets from the corrected truth.
- [ ] Re-run 32-subset forecast geometry before any online transfer or PPO.
- **Status:** truth regeneration complete; asset rebuild is the next gate.

### Phase 31: Event-specific quality/resource route
- [x] Close V559 after refit-TCN geometry remained below the material gate.
- [x] Repair the quality generator so declared risk columns may come from
  truth event structure when they are not present in the resource trace.
- [x] Generate event-specific quality truth for seeds `7241--7244`.
- [x] Refit frozen assets and verify all four manifests retain 32 candidates at
  B=`2.15`.
- [ ] Proceed to online transfer only if the predeclared geometry gate passes.
- **Status:** truth and assets complete; V562 geometry is running. No policy
  training is authorized.

### Phase 32: V562 event-specific frozen geometry
- [x] Launch the four-seed, four-start, 32-subset geometry audit from the V561
  frozen assets at B=`2.15`/startup=`2.60`.
- [x] Sync and inspect condition-level and operating-state opportunity gaps.
- [x] Apply the material opportunity and near-optimal-static-intersection
  gates before considering online transfer.
- **Status:** complete; the event-specific route is closed before online
  transfer and PPO because operating-state opportunity remained below the
  material gate in all four seeds.

### Phase 33: Physical-budget calibration gate
- [x] Compare the current normalized budget (`2.15`) with the resource
  manifest's physical development budget (`55 W`) and controller budget
  (`1200 W`).
- [x] Check whether the existing repository contains an authoritative supply
  specification that justifies a new binding arbitrary-subset budget.
- [ ] Do not launch another asset or PPO wave until a documented budget scale
  is selected independently of policy results.
- **Status:** calibration gate open. The current `2.15` route is a normalized
  development proxy, while `55 W` makes the current six-channel subset family
  nearly unconstrained; no defensible new budget is available from the current
  repository evidence alone.

### Phase 34: Persistent target-dominance truth route
- [x] Apply the existing persistent-mode target-dominance generator to the
  V561 event-quality truth without changing event labels or power traces.
- [x] Verify mode duration, target amplitude, quality coupling, and online
  feature availability from truth-only diagnostics.
- [x] Refit frozen assets and audit all 32 subsets at the already screened
  normalized budget B=`2.15` only if the truth support gate passes.
- [x] Continue to online transfer or PPO only after condition and operating
  forecast geometry pass the material gate.
- **Status:** closed before online transfer and PPO. V564 operating-state
  opportunity was `0.00053--0.00859`, below the material `0.01` gate; the
  low-cost candidate-023 shortcut remained near-optimal across states.

### Phase 35: Post-V564 route decision
- [x] Sync and aggregate the complete V564 geometry artifacts.
- [x] Record whether persistent target dominance creates a material executable
  forecast frontier.
- [x] Select the next route only from an independently justified physics or
  resource-scale change; do not tune target gains or train PPO against this
  failed geometry.
- **Status:** selected a fixed sampling-frequency effective-cost screen using
  the same six physical channels and a mandatory CR1000Xe backbone. No policy
  training is authorized yet.

### Phase 36: Frequency-scaled effective-cost screen
- [x] Add a self-contained sensor configuration with fixed effective costs and
  explicit frequency multipliers outside the action space.
- [x] Add a resource manifest distinguishing effective-cost scenario design
  from installed-watt telemetry.
- [x] Verify syntax and enumerate the resulting arbitrary-subset surface.
- [ ] Prepare frozen assets from the existing V563 truth only after reviewing
  the resource manifest and confirming the budget is predeclared.
- [x] Audit subset forecast geometry before online transfer or PPO.
- **Status:** closed before online transfer and PPO. V566 operating-state
  opportunity was `0--0.00547` across four seeds, below the material `0.01`
  gate, despite the tighter 16/32 feasible surface.

### Phase 37: Continuous online-exposure truth gate
- [x] Generate target/quality truth from noisy-nowcast-derived persistent
  exposure states without exposing the latent states to the scheduler.
- [x] Derive the matching heater/resource trace from the same exposure states.
- [x] Audit time alignment, target/quality coupling, state persistence,
  event-label preservation, and resource-state support before asset fitting.
- [x] Prepare frozen assets and run geometry only if the truth-only gate passes.
- **Status:** truth/resource generation running remotely; no asset preparation
  or policy training is authorized yet.
- **Decision:** closed at the truth-only gate. Exposure fraction above 0.5 was
  `0.9908--0.9925` and GMX heater occupancy was `0.9825--0.9898`; these traces
  are effectively always-high and cannot support a state-dependent resource
  geometry without post hoc threshold tuning.

### Phase 38: Training-partition quantile exposure calibration
- [x] Generate quantile-calibrated exposure truth using only indices
  `[0,31500)` for each seed.
- [x] Derive the matching resource trace and audit state support, target
  coupling, labels, and calibration metadata.
- [ ] Prepare frozen assets and audit geometry only if the truth-only gate
  passes; no threshold changes may use validation or test occupancy.
- **Status:** truth-only gate passed for the development screen: exposure
  occupancy was `0.807--0.821`, heater occupancy was `0.507--0.784`, direct
  nowcast-to-state correlations were positive for all three state families,
  and quality bounds/labels remained valid. Asset preparation is authorized;
  no policy training is authorized.

### Phase 39: Quantile-exposure frozen geometry
- [ ] Prepare frozen TCN/static assets from V568 truth with online nowcast
  columns only; no latent exposure columns may enter scheduler state.
- [ ] Audit all arbitrary subsets under the coupled dynamic resource trace.
- [ ] Continue to online transfer or PPO only if operating-state forecast
  opportunity is at least `0.01` in every development seed and no static
  near-optimal intersection survives.
- **Status:** asset preparation running remotely as `pdppo_v569_assets`; no
  policy training is authorized.

### Phase 40: V577 Resource-Geometry Phase Screen
- [x] Close V576 after the independent-nowcast heater trace failed the
  operating opportunity gate in all four predeclared windows.
- [ ] Run a predeclared budget phase screen over the same frozen V575 assets,
  truth windows, and dynamic resource trace; do not train PPO or select new
  evaluation starts from the screen.
- [ ] Determine whether any physically documented effective-budget interval
  makes the heater-dependent laser alternatives compete with the persistent
  non-laser static subsets.
- [ ] If no interval passes the executable subset-geometry gate, close the
  heater-only route and redesign the resource mapping before online transfer.
- [ ] If an interval passes, freeze it and run online transfer before any PPO.
- [x] Complete the budget phase screen and close the route when no all-seed
  operating interval passes.
- **Status:** closed; no V577 interval justified online transfer or PPO.

### Phase 41: V578 Persistent-Target Cross Geometry Screen
- [x] Close V577 after the full effective-budget screen failed the all-seed
  operating opportunity gate.
- [x] Reuse the pre-existing persistent target truth with its four online
  nowcast columns and apply the already frozen independent GMX/Parsivel heater
  controller; do not modify targets, event labels, or policy observables.
- [ ] Audit truth-only state support and resource alignment.
- [ ] Prepare assets and audit geometry only if the truth-only gate is valid;
  no policy training is authorized before the operating geometry gate passes.
- [x] Prepare frozen assets and audit all four predeclared windows.
- **Status:** closed before online transfer and PPO. The route had state-wise
  candidate changes and no 1% static intersection, but operating gaps passed
  the `0.01` threshold in only one of four seeds.

### Phase 42: V579 Persistent-Target Resource Phase Screen
- [x] Freeze the V578 target/assets and retain the same four evaluation windows.
- [ ] Screen the predeclared effective budgets `0.55--2.15` without changing
  truth, resource controller, or candidate definitions.
- [ ] If no budget passes the all-seed operating geometry gate, close the
  heater/resource route and stop before online transfer and PPO.
- [ ] If an interval passes, freeze it and run online transfer before any PPO.
- [x] Complete all ten budget points and aggregate the four-seed operating
  geometry results.
- **Status:** closed before online transfer and PPO. Budgets `0.85--1.85`
  removed the static intersection but did not create material operating gaps;
  B=`2.15` had a material gap in only one seed.

## Errors Encountered
| Error | Resolution |
|-------|------------|
| First V569 asset-only manifests omitted the dynamic-resource mapping, so downstream geometry would have silently reconstructed static costs | Discarded the first artifacts; added trace, five power mappings, budget, feasibility source, and online-state metadata to the manifest writer; synced the fix, regenerated all four seeds, and verified the corrected manifests before geometry |
| First V570 geometry attempt read the base truth CSV without merging the trace referenced by the manifest, causing missing effective-power columns during environment construction | Added manifest-driven time-aligned trace merging with duplicate and coverage checks to the geometry audit, passed local `py_compile` and `git diff --check`, synced the script, and relaunched V570; no failed-run result is interpreted |

### Closed routes

- **V616:** Corrected empirical cold-availability floor to zero. The trace had
  real heater-state variation, but the frequency-weighted complete-subset
  forecast gap was below `0.002` in three seeds and zero in one; the same
  fixed subset won all seeds. Closed before online transfer/PPO. Any successor
  must predeclare state occupancy and persistence from the physical controller
  and pass the complete-subset geometry gate.

- **V570:** Dynamic power alone yielded operating gaps below `0.01` in three
  seeds and left a 1% static intersection in three seeds.
- **V572:** Adding the declared GMX500/Parsivel quality restoration did not
  change the conclusion; three seeds retained a 1% static intersection.
- **V574:** The absolute device-controller route had no state variation in the
  frozen final windows, so its operating gap was zero in every seed. Minority
  states elsewhere in test were not used to select new windows.

## V581 bounded follow-up

V581 is the final bounded truth-only follow-up to V580. It replaces absolute
driver clipping with q10/q90 normalization fitted on the first 31,500 rows of
each seed, then applies the same persistent filters, specialist target and
quality relations, delayed alert construction, independent heater trace, and
fixed evaluation starts. No test rows determine thresholds or starts. Failure
closes the Stage-B route before asset creation.
| Initial v14 eventgate rsync used the repository root with `--relative`, creating a nested `rl_sensor_scheduling_framework/rl_sensor_scheduling_framework` copy on the server | Re-synced from the framework root with `./scripts/...` and `./configs/...`, removed the nested copy and misplaced top-level files, then ran remote `py_compile` and `bash -n` successfully |
| V24 dual/event split-replay rsync initially copied the script/YAML files into the remote framework root | Immediately moved misplaced root copies into `scripts/` and `configs/sensors/`, then reran remote `py_compile` and `bash -n` before launching tmux jobs |
| First AWBC1.2 launch command had a broken nested SSH/awk quote and exited before starting anything | Relaunched using a quoted heredoc (`ssh ... 'bash -s' <<'REMOTE'`) and verified tmux `pdppo_v24_eventflux_cyclicppo_awbc1p2_seed45_h082_20260620` started |
| Phase16 early monitor used a GPU2-only inline `awk` filter that broke under nested SSH quoting | The experiment was unaffected and the active process showed the intended phase arguments; use full `nvidia-smi --query-gpu` output or avoid inline awk in later monitors |
| Remote non-interactive `python` was not on PATH during v15 validation | Re-ran validation after `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate darts`; remote `py_compile` and runner syntax checks passed |
| V16 first rsync repeated the nested-path risk by using `--relative` from the repository root | Re-synced individual files to `configs/sensors/` and `scripts/`, removed the nested directory, and verified remote YAML/boundary checks |
| V16 first tmux launch exited immediately because log redirection parent directory did not exist | Created `reports/v31_static_break_v16_surface_boundary_gate_seed42_20260609/` before launching tmux |
| H0.82 seeds 41/43 runner passed two GPU IDs as separate args | `59_v31_split_protocol_grid.py` expects `--gpu-ids` as one comma-separated string; changed to `--gpu-ids 3,5` and relaunched |
| First v19 rsync used the framework root as target | The two new files were briefly placed at `rl_sensor_scheduling_framework/` on the server root instead of their subdirectories; re-synced into `configs/sensors/` and `scripts/`, removed the misplaced root copies, and verified placement with remote `test` commands |
| V22 audit command used a duplicated framework prefix | The first local audit call ran from `rl_sensor_scheduling_framework/` but invoked `rl_sensor_scheduling_framework/scripts/68_v31_operational_rollout_audit.py`, producing file-not-found | Re-ran as `python scripts/68_v31_operational_rollout_audit.py ...` from the framework root |
| V23 rsync initially targeted the framework root | The first v23 sync placed the YAML and runner at the remote framework root | Moved the YAML to `configs/sensors/`, moved the runner to `scripts/`, set executable bit, and reran remote validation |
| Remote v23 PPO content check used `python` without activating Conda | Non-interactive SSH PATH did not include Python | Replaced it with shell `grep` checks after remote `py_compile` had already passed under the `darts` environment |
| GPU5 filter command had an inline `awk` quoting error | SSH quoting broke the `$1 ~ /^5/` expression | Re-ran the full `nvidia-smi --query-gpu` command and confirmed GPU 5 was idle before launching |
| Local SCENEBAL-1 watcher Markdown list lines used `printf '- ...'` | Bash treated the leading hyphen in the format string as a `printf` option and emitted repeated `printf: - : invalid option` messages | Changed those lines to `printf -- '- ...'`, ran `bash -n`, restarted the local watcher/postcollect tmux sessions, and synced the fixed scripts to `remote-gpu` |
| Remote sync verification repeated the non-interactive `python` PATH issue | The first remote YAML validation after sync printed `bash: line 1: python: command not found` | Re-ran validation with `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate darts && python`; remote `research-state.yaml` parsed successfully |
| Paper rsync command accidentally sent `main.pdf` to `paper/sections/` on `remote-gpu` | A multi-source rsync line targeted the sections directory while including `paper/main.pdf` | Removed the misplaced remote `paper/sections/main.pdf` immediately and kept the correct `paper/main.pdf`; future paper sync should send section `.tex` files and root `main.pdf` in separate rsync commands |
| Pilot watcher overwrote local postcollect status with stale remote Markdown | The pilot watcher included broad `scenebal1_*_20260621.md` files, so its periodic aggregate sync could pull old remote status files back over newer local status files | Added `--exclude "*local_watch*"` and `--exclude "*postcollect_status*"` before includes, removed stale remote status Markdown, restarted both local watcher tmux sessions, and confirmed both status files update at `13:06` |
| Multi-source asset rsync placed SCENEBAL-2 helper assets in the remote project root | A command synced `scripts/77_v31_write_scenebal_summary_table.py` and `paper/tables/scenebal2_6seed_staticnorm_macro_summary.tex` together to the framework root, so rsync used basenames instead of preserving subdirectories | Re-synced the script to `scripts/`, re-synced the table to `paper/tables/`, removed the misplaced root copies, and verified correct placement with remote `test` commands |
| Remote monitor command printed `printf: --: invalid option` | The monitor snippet used `printf "--- seed%s ---\n"`, and bash treated the leading hyphen in the format string as an option | Treat as monitor-only noise; use `printf --` or a non-hyphen-leading format in later remote monitor commands |
| Local plan read used the project root instead of framework root | The first post-compaction command tried to read `.planning/...` and `research-state.yaml` from `/home/horeb/_code/microclimate_demo` | Re-ran from `/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework`; framework root is the authoritative planning location for this task |
| V559 truth generation initially failed on relation-schema drift | `heater_quality_relation_v2.json` used descriptive field names while the generator required `risk_columns`, `quality_columns`, `risk_quality`, and `heated_quality` | Added the explicit schema fields, validated JSON and Python compilation, then reran all four truth jobs successfully |
| V561 asset launch used the V559 truth suffix | The reusable asset launcher selected `_heater_quality.csv` while V561 generated `_event_quality.csv` | Parameterized `V559_TRUTH_SUFFIX` and relaunched V561; no asset or scientific result from the failed launch is retained |
| First V564 geometry launch exited before Python started because the redirected output directory did not exist | Created the output directory before redirection and relaunched the same predeclared geometry command; no scientific output was produced by the failed attempt |
| V605/V606 quality-coupled asset replay initially exposed path, stale-resource-column, and state-dimension mismatches | Repaired bundle metadata, merged the exact resource trace before oracle fitting, made the 15-state contract explicit, and reran one complete geometry audit; the resulting scientific gate failed on insufficient forecast-value separation |

### Phase 43: V618 Budget Phase Geometry Screen
- [x] Evaluate the predeclared interface budgets `1.25, 1.50, 1.75, 1.90,
  2.05, 2.15` with startup budgets `1.70, 1.95, 2.20, 2.35, 2.50, 2.60`.
- [x] Reuse V616 frozen assets, the physical 55 W resource trace, all 32
  candidate masks, and the locked eight-start geometry protocol.
- [x] Aggregate all 24 budget--seed combinations before selecting any point.
- [x] Apply the all-seed operating materiality gate and stop before online
  transfer/PPO when it fails.
- **Status:** closed. No budget passed the all-seed `>=0.01` operating gap
  gate. The V614--V618 empirical-cold/heater route is closed. The next route
  is a V557 causal-scene observability audit, not another budget or learner
  sweep.

### Phase 44: V619 V541 Context Repair
- [x] Verify that the V541 truth contains decision-time meteorology and
  delayed alert proxies while its stored `agent_context_columns` is empty.
- [x] Preserve V541 truth/resource/assets protocol and add only those eight
  existing online columns to a fresh asset preparation run.
- [ ] Validate all four repaired manifests.
- [ ] Re-run complete-subset geometry before online transfer.
- [ ] Run online transfer only if geometry remains material and no static
  shortcut gate is introduced.
- **Status:** asset preparation running remotely in `v619_context_assets`;
  PPO remains blocked.

### Phase 45: V620 Paired No-Context Control
- [x] Reuse the exact V541 truth, resource trace, sensor configuration,
  partitions, starts, and budgets used by V619.
- [x] Rerun asset preparation with `agent_context_columns` empty to isolate
  TCN/refit randomness from the V619 context intervention.
- [ ] Validate the four control manifests and compare paired oracle hashes,
  geometry gaps, static candidates, and 1% intersections.
- [ ] Decide whether online transfer is scientifically admissible after the
  paired geometry comparison.
- **Status:** control assets are running in `v620_context_control_assets`;
  geometry is chained behind completion; PPO remains blocked.

### Phase 45 execution correction
- [x] Confirm all four V620 control manifests completed.
- [x] Discard the first geometry launch as a working-directory failure; it
  produced no scientific Python output.
- [ ] Complete the corrected geometry rerun in `v620_control_geometry_manual2`.
- **Status:** corrected geometry is running under the `_r2` output root; paired
  transfer comparison remains pending.
### Phase 46: Frozen V541 Context Injection
- [ ] Copy the original V541 frozen oracle and preserve its truth/resource,
  partitions, normalization, candidate masks, and static-selection outputs.
- [ ] Inject only the eight already available decision-time context columns
  into the copied truth and observation metadata; do not refit the TCN.
- [ ] Verify oracle hash equality and exact state/context metadata before
  geometry auditing.
- [ ] Re-run the V557 four-start geometry audit and compare it with V541.
- [ ] Run online transfer only if frozen geometry remains material and no
  static shortcut gate is introduced.
- **Status:** planned; V619/V620 refit waves remain diagnostic only.

### Phase 47: Frozen-context online transfer
- [x] Run V621 with V541's frozen oracle and only the eight online context
  columns added to metadata.
- [x] Verify exact oracle hash and geometry equality with V541.
- [x] Evaluate held-out transfer with two chronological training starts and
  two held-out starts per seed.
- [x] Apply the predeclared positive-margin transfer gate.
- **Status:** closed. Mean transfer margin was `-0.011233` with `2/4`
  positive seeds; no PPO training is admissible on this route.

### Phase 48: Leave-one-start-out transfer sensitivity
- [ ] Generate one online observation probe for each of the four locked
  geometry starts using the V621 frozen-context bundle.
- [ ] Run all four leave-one-start-out folds per seed with no event labels,
  future targets, bandit output, or policy training.
- [ ] Aggregate fold margins by seed before any decision; no fold may be
  selected post hoc.
- [ ] Promote the route only if every seed has a positive mean held-out
  margin and at least three of four folds are positive; otherwise close it.
- **Status:** planned bounded transfer-sample sensitivity check.

### Phase 48 closeout
- [x] Generate one observation probe for each of the four locked starts.
- [x] Run all 16 leave-one-start-out folds and aggregate by seed.
- [x] Apply the predeclared positive-mean and `>=3/4` positive-fold rule.
- **Status:** closed. No seed had a positive mean transfer margin; PPO remains
  blocked on this scene/observation route.

### Phase 49: Unclipped transfer diagnostic correction
- [ ] Recompute the V621 four-seed geometry with `oracle_loss_clip` set above
  the observed loss range, keeping all assets, starts, budgets, and metadata
  unchanged.
- [ ] Verify that candidate losses are not dominated by an artificial clip.
- [ ] Repeat the two-by-two and leave-one-start-out transfer diagnostics from
  the corrected loss files.
- [ ] Reapply the transfer gate before considering PPO.
- **Status:** reopened. V622/V623 results are invalidated by the clipped loss
  artifact and cannot close the route.

### Phase 49 closeout
- [x] Recompute all four geometry audits with a nonbinding loss clip.
- [x] Verify zero artificial clipping and positive operating geometry.
- [x] Repeat all 16 leave-one-start-out transfer folds.
- [x] Reapply the transfer gate to corrected losses.
- **Status:** closed. Every seed had a negative mean transfer margin and only
  `3/16` folds were positive; PPO is not admissible on this route.

### Phase 50: Coverage-policy transfer diagnostic
- [ ] Generate the same four-start observation probes under a feasible
  round-robin coverage policy, with event labels and future targets excluded.
- [ ] Reuse the corrected V624 candidate-loss geometry and repeat the full
  leave-one-start-out transfer audit.
- [ ] Compare against V625 only as a predeclared observation-coverage
  diagnostic; do not select a favorable fold or alter the scene from results.
- **Status:** planned. This tests whether V625 failed because candidate-000
  observations under-cover channel histories; it does not change PD-PPO.

### Phase 50 closeout
- [x] Run the four-start round-robin coverage probes.
- [x] Repeat all 16 corrected leave-one-start-out transfer folds.
- [x] Compare seed-level means without selecting a favorable fold.
- **Status:** closed. Coverage probing did not restore transfer; no PPO is
  admissible for this scene.

### Phase 51: Observable heater/resource route
- [x] Build a hardware-manifest-based heater/resource trace from decision-time
  weather variables, without event labels or future targets.
- [x] Couple causal operating factors to specialist target quality and audit
  all 32 optional subsets under the dynamic resource guard.
- [x] Repeat the operating geometry with decision-time nowcast inputs for the
  heater controller.
- [x] Audit the complete truth time axis for dynamic feasible-frontier
  occupancy, independently of forecast loss.
- [ ] Complete a truth-only resource-state-coverage geometry audit before any
  PPO training.
- **Status:** V627--V635 established real dynamic resource occupancy and a
  changing 32-subset frontier, but the original V635 starts all sampled the
  simultaneous-heater state. V636 is the controlled coverage correction;
  PPO remains blocked until its downstream opportunity gate passes.

### Phase 52: Sensor-specific quality closure
- [x] Couple heater state to the quality of the exposed GMX500 and Parsivel
  channels using a bounded icing-protection factor.
- [x] Add a bounded radiometer signal-quality factor based on decision-time
  solar irradiance, without changing the reward or policy architecture.
- [x] Complete V642 geometry and apply the same all-seed `0.01` opportunity
  gate.
- **Status:** closed. V642 passed in `3/4` seeds with operating gaps
  `0.046997`, `0.119788`, `0.017873`, and `0.000000`; the all-seed gate
  failed at seed `7180`. No PPO training is admissible for this scene family.
  The resource frontier is real, but the downstream forecast opportunity is
  not stable enough for a defensible learned-policy experiment.
### Phase 53: Entity-energy admissibility and route correction (2026-09-10)

- [x] Audit the existing finite-energy implementation, hardware manifest, and
  available truth duration before reusing SOC parameters.
- [x] Separate controllable sensor loads from the legacy non-controllable
  external heater envelope.
- [x] Reject the current SOC artifact as primary deployment evidence: it has
  only 25 hourly rows, and the 600 W external heater is not a selectable
  sensing-system load.
- [ ] Do not launch PPO with the existing `h/capacity` constants.
- [ ] Admit a future SOC route only after a traceable supply/storage trajectory
  and a sufficiently long truth window are available.
- [ ] Otherwise continue with hardware-derived instantaneous budget breakpoints
  and complete-subset forecast geometry, with no post-hoc budget selection.

**Decision:** the current implementation is retained as code infrastructure and
diagnostic history, but its SOC results are not promoted to scene evidence.
V643 remains the latest valid geometry screen. The next admissible work unit is
either a data-backed supply/storage trace audit or a predeclared physical budget
phase screen; no policy training is allowed before the subset-level opportunity
and online-transfer gates pass.

### Phase 54: Empirical Cold-Wind Target Coupling (2026-09-10)
- [x] Re-audit the existing V616 32-subset forecast geometry before creating
  new truth.
- [x] Record that operating opportunity gaps were `0.001953`, `0.001002`,
  `0.001046`, and `0.000000` for seeds `7401--7404`; condition-specific
  winners changed in three seeds, but the all-seed materiality gate failed.
- [x] Implement a truth-only delayed AR(1) wind innovation scaled by the
  measured cold-availability risk. The audit column is excluded from policy
  context and no exact test labels are used.
- [x] Run V645 truth-only generation on `remote-gpu` and audit support,
  temporal lag, clipping, and policy-input exclusion.
- [x] Rebuild matched frozen assets and rerun resource/subset geometry after
  the truth-only audit was judged plausible.
- [x] Close this route before PPO because the all-seed `>=0.01` opportunity
  gate and the 1% near-optimal-intersection gate failed.
- **Status:** closed before online transfer/PPO. V645 did not establish a
stable task-level adaptive opportunity.

### Phase 55: Dynamic-resource unit reconciliation (2026-09-10)
- [x] Inspect the environment, projector, runner arguments, manifest, and
  current trace generator.
- [x] Confirm that dynamic-resource feasibility is enforced separately from
  the normalized interface-cost projector.
- [x] Identify the historical prefix/unit mismatch and record its impact on
  reproducibility.
- [x] Make the training entry point prefer explicit physical-watt columns
  while retaining legacy compatibility.
- [ ] Run focused tests and regenerate a current physical-unit geometry screen.
- [ ] Only reopen downstream forecast geometry if the corrected screen is
  reproducible and passes the existing all-seed materiality gate.
- [x] Rebuild matched V646 assets and complete the corrected four-seed
  downstream geometry screen.
- [x] Close the route before online transfer/PPO because only `1/4` seeds
  passed the materiality gate.
- **Status:** complete; next route selection required

### Phase 56: Physical-only 32-subset geometry screen (2026-09-10)
- [x] Quantify the action-space effect of the legacy normalized interface
  budget: physical-only feasibility averages `30.21/32` masks, while the
  combined V646 constraints average `15.78/32`.
- [x] Define a matched physical-only screen with normalized steady/startup
  budgets set to nonbinding `10.0/10.0`, physical dynamic budget fixed at
  `55 W`, and all other V646 settings frozen.
- [x] Launch remote four-seed asset preparation.
- [x] Complete the 32-subset downstream geometry audit.
- [x] Advance to online transfer after the all-seed
  opportunity and near-optimal-static gates.
- **Status:** complete; V648 passed and V649 transfer screen active

### Phase 57: Chronological online transfer after V648
- [x] Diagnose and reject the V647 mask-normalization asset failure.
- [x] Rebuild V648 forecaster assets with complete candidate-mask coverage.
- [x] Verify V648 four-seed operating forecast geometry.
- [x] Evaluate a heater-state winner lookup trained on three starts and tested
  on a held-out start.
- [x] Reject the heater-state transfer lookup after held-out losses were
  non-positive in only `0/4` seeds.
- [x] Run the bounded observable-context fixed-mask transfer probe.
- [x] Close this physical-resource route before PPO because the probe was
  worse than the static reference in `3/4` seeds and had no dwell execution.
- **Status:** complete; no PPO result is promoted

### Phase 58: Route closeout and evidence handoff (2026-09-10)
- [x] Preserve V648 as geometry evidence, not policy evidence.
- [x] Preserve V649 as a failed resource-state transfer diagnostic.
- [x] Preserve V650 as a failed observable-context opportunity probe.
- [x] Append exact results to progress, findings, and CHANGELOG.
- **Status:** complete; next route must change the physical/forecast coupling,
  not tune PPO on the closed route

### Phase 59: Stage-B persistent target/resource re-screen (2026-09-10)
- [x] Reuse the predeclared persistent transport/thermal truth chain from
  Stage-A planning, with its noisy nowcast driver and causal quality link.
- [x] Reuse the matched entity resource traces and physical budget settings.
- [x] Rebuild forecaster assets with complete candidate-mask coverage.
- [x] Complete the first four-seed geometry run and identify its operating-label
  priority bug.
- [x] Complete corrected four-seed 32-subset forecast geometry.
- [ ] Apply the same online-transfer gate before any PPO training.
- **Status:** complete; V654 failed held-out operating geometry and binding
  resource gates; PPO remains blocked on this route

### Phase 60: Binding budget correction
- [x] Run a policy-free 2/3/5/10/20 W occupancy screen on the existing
  physical resource traces.
- [x] Select 5 W as the bounded breakpoint because it leaves 12/32 masks
  always feasible while changing the frontier on about 84--86% of rows.
- [x] Complete matched V656 budget-5 asset preparation.
- [x] Complete V657 5 W complete-subset geometry and the chronological online-transfer
  gate on V656 before any PPO.
- **Status:** complete; V657 failed held-out operating transfer and this route
  is closed before PPO

V657 test operating gaps were `0`, `0`, `0`, and `0.001437`. The binding
resource budget alone did not make the persistent-factor opportunity
chronologically identifiable.

### Phase 61: Observable specialist-mode route (2026-09-10)
- [x] Build truth-only persistent specialist modes from training-prefix
  calibrated nowcasts, with delayed noisy online alert proxies.
- [x] Verify mode support, alert correlation, and multi-level resource load
  support on all four seeds without exposing mode labels to the scheduler.
- [x] Finish matched V659 frozen asset preparation at the 5 W physical budget.
- [x] Run V660 complete 32-subset forecast geometry with corrected
  multi-factor operating labels.
- [x] Apply chronological online-transfer gates before any PPO training.
- [x] Reject PPO on this route after unstable transfer evidence.
- **Status:** complete; V658--V666 closed before PPO.

### Phase 62: Operating-label correction (2026-09-10)
- [x] Complete V660 on the unchanged V659 assets and identify the label
  priority failure.
- [x] Correct the audit to use the persistent truth-only mode id before its
  continuous component scores.
- [x] Complete the matched V661 train/test geometry rerun.
- [x] Reapply chronological online-transfer gates before any PPO probe.
- **Status:** complete; V660 was invalid as an operating-stratification result,
  V661 corrected it, and no policy evidence was promoted.

### Phase 63: Fixed final-window coverage screen (2026-09-10)
- [x] Audit V661 mode support across a predeclared final-window grid.
- [x] Complete V662 geometry at starts `82600, 83900, 85200, 86500, 87800`
  for every seed.
- [x] Aggregate the grid without selecting a favorable window after seeing
  forecast losses.
- [x] Reject online transfer and redesign the causal state/quality link.
- **Status:** complete; V662 exposed heterogeneous windows, and V664/V666
  failed stable transfer.

### Phase 64: Deployable observation transfer (2026-09-10)
- [x] Complete pooled V662 geometry on the fixed final-window grid.
- [x] Generate online observations for the same train/test windows without
  event labels or future targets.
- [x] Fit train-only candidate-loss regressors and evaluate test transfer.
- [x] Add the predeclared fourth training start and repeat the transfer check.
- **Status:** complete; V664/V666 improved three seeds but had mean loss
  degradation and a persistent seed7233 failure. PPO was correctly blocked.

### Phase 65: Causal quality-link redesign gate (2026-09-10)
- [x] Close the V658--V666 specialist-mode route before PPO.
- [ ] Design a new causal quality/resource chain with cross-seed support,
  preserving truth-only generation and online observability.
- [ ] Repeat policy-free resource occupancy, subset geometry, and online
  transfer gates before considering any PPO probe.
- **Status:** active; no PPO has been run on the flexible-subset redesign.

### Phase 66: Dwell-aware repair of the alert-coupled transfer audit (2026-09-10)
- [x] Confirm that V601 assets include minimum dwell six and that V602
  fixed-subset geometry used the same environment-level execution guard.
- [x] Confirm that V604 selected candidate rows independently and therefore
  did not execute its selected sequence through the dwell-aware environment.
- [x] Implement a train-only observation-value policy whose selections are
  replayed through `WarmupSchedulingEnv`, with event labels excluded.
- [x] Complete the four-seed, eight-window executable transfer audit.
- [x] Close the alert-coupled route because executable transfer is not stable
  across seeds; do not start PPO on this route.
- **Status:** complete; V667 found 16/32 window wins, but the pooled mean
  transfer delta remained positive in loss and seed7403 degraded.

### Phase 67: Next causal scene design (2026-09-10)
- [ ] Reuse only the physics and observation lessons from V627--V667; do not
  reuse a scene whose online transfer gate has failed.
- [ ] Define a new truth-only causal chain in which forecast-relevant channel
  quality and effective resource load share observable drivers, with explicit
  cross-seed and full final-window support requirements.
- [ ] Run policy-free resource occupancy and complete-subset geometry before
  generating any PPO asset or checkpoint.
- [ ] Run the executable train-only observation transfer audit before any PPO
  probe.
- **Status:** active; flexible-subset PPO remains blocked pending a new scene.

### Phase 68: Corrected V610 frequency-mode geometry (2026-09-10)
- [x] Confirm that the original V610 audit stratified rows by heater columns,
  despite the frequency-cost trace exporting `resource_frequency_mode_id`.
- [x] Patch the geometry audit to prioritize the truth-only resource mode id
  before generic factor, duty, or heater labels.
- [x] Reuse the V610 assets, budget, evaluator, and four predeclared starts;
  do not change truth, resource costs, or PPO.
- [x] Complete the corrected four-seed geometry audit.
- [x] Apply the online-transfer gate decision from the corrected operating
  geometry before considering any policy run.
- **Status:** complete; operating gaps were `0`, `0.0000262`, `0`, and `0`,
  with a nonempty 1% static intersection in every seed. V610 is closed before
  online transfer and PPO.

### Phase 69: Causal scene redesign after V668 (2026-09-10)
- [x] Withdraw the invalid heater-partition V610 conclusion.
- [x] Confirm that the corrected frequency-mode route still fails the
  deployable forecast-geometry gate.
- [x] Reuse only the 09-02 guidance: share observable drivers between
  forecast-relevant quality and resource load, with truth-only latent state.
- [x] Define cross-seed support, persistence, full-window geometry, and
  executable online-transfer acceptance tests before generating assets.
- [x] Implement the next truth-only screen without creating PPO checkpoints.
- [ ] Generate matched frozen evaluator assets only if the truth-only screen
  remains physically supported after the remote reproducibility check.
- **Status:** active; V669 passes the local occupancy screen, but PPO remains
  blocked pending complete-subset geometry and online transfer.

### Phase 70: V669 complete-subset geometry (2026-09-10)
- [x] Reproduce the V669 truth/resource screen on `remote-gpu`.
- [x] Prepare four frozen TCN assets with 32 candidate masks and the declared
  shared-nowcast context columns.
- [x] Complete four-seed geometry at the fixed starts
  `82600, 83000, 84800, 85500`.
- [x] Close the scene before transfer because final windows lacked operating
  state support and retained a static intersection.
- [ ] Close the scene before transfer if any seed retains a material static
  intersection or fails the operating opportunity threshold.
- **Status:** complete; V670 operating gaps were zero in all four seeds. The
  schema-corrected route is closed before transfer and PPO.

### Phase 71: Resource-controller support redesign (2026-09-11)
- [x] Diagnose that V669 full-sequence heater variation was absent from the
  fixed final windows.
- [x] Screen a conditional dew-point/humidity controller without a forced
  low-temperature heater trigger.
- [ ] Reject controller variants with negligible occupancy or unsupported
  hardware semantics.
- [ ] Define the next physical resource state with full final-window support
  before rebuilding any forecaster.
- **Status:** active; no PPO is permitted.

### Phase 72: V672 shared-driver resource geometry (2026-09-11)
- [x] Define a shared-driver hysteresis controller with fixed frequency costs.
- [x] Screen final-window state support and feasible-frontier occupancy.
- [ ] Generate matched resource traces remotely from the frozen V669 truth.
- [ ] Complete 32-subset forecast geometry before any transfer audit.
- [ ] Close the route if the operating opportunity or static-intersection
  gate fails.
- **Status:** active; V672 passes resource support only.
