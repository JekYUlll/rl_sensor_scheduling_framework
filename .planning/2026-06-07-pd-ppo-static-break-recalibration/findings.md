# Findings & Decisions

## 2026-09-10 V648 geometry pass

The corrected mask-balanced assets remove the V647 normalization failure. All
four seeds pass the operating opportunity threshold with gaps
`0.043614/0.022431/0.026198/0.016515`, and every seed has an empty 1% near-
optimal static intersection. The best operating candidate changes between
heater states in all four seeds. This establishes task-level adaptive
opportunity under the physical resource chain. It does not establish that the
online observation identifies the winning subset, so the next required test is
chronological online transfer; no policy result is promoted yet.

V649 uses a heater-state winner lookup trained on three starts and evaluated
on held-out start `82600` before any PPO training. A positive result would
justify a small policy probe; a failure would block PPO and require closing or
revising the resource-state route.

## V649/V650 closeout decision

V649 failed the chronological transfer gate: the heater-state lookup was no
 better than the training static candidate in only two seeds and was worse in
 the other two. V650 used a richer online feature set and no dwell execution,
 but improved over static in only one of four seeds. The physical route is
 closed before PPO. The evidence supports the narrower conclusion that
 state-dependent feasible-subset geometry exists, but the current online
 observations do not identify its forecast-value winner robustly enough for a
 learned-policy claim.

## 2026-09-10 V647 invalid-asset diagnosis

V647's large losses are invalid for scientific interpretation. The physical
budget change exposed a missing asset-preparation setting: candidate masks
were not included in forecaster training because `oracle-candidate-mask-repeat`
was left at its default zero. The stored mask-channel standard deviations were
`1e-6` for constant dimensions, while sparse-mask evaluation changed those
dimensions by approximately one unit. The resulting normalized features were
about `1e6` and the TCN output reached `2.4e5` before the normal `10.0` loss
clip. This was verified by replaying the same seed with the V647 checkpoint.

V647 is therefore discarded as a geometry result. The corrected V648 asset
preparation explicitly repeats the candidate-mask policies once per candidate;
the physical resource chain, truth, budgets, starts, and evaluation code are
unchanged. No PPO launch is permitted until V648 passes the existing
all-seed forecast-geometry and online-transfer gates.

## 2026-09-09 V559 heater-quality truth

- The V541 `entity_effective_with_resource` assets exposed dynamic heater and
  effective-power columns, but their existing channel-quality columns came
  from the earlier specialist-separated truth and were not functions of those
  heater states. This explains why V557 privileged forecast crossover did not
  transfer through the online observation.
- After repairing the relation schema, V559 generated four 90,000-row truth
  files with `truth_targets_changed=false` and `event_labels_changed=false`.
  The artifact is therefore a quality/observation transformation, not a new
  target or event-label definition.
- Under the declared storm-conditioned proxy relation, GMX500 heater occupancy
  was `5.79%--5.93%` and laser heater occupancy `7.32%--7.53%`. When those
  heaters were on, the corresponding quality mean was `1.0`; when off, the
  mean quality was approximately `0.54--0.58` for the met channel and
  `0.37--0.39` for the laser channel. This establishes a measurable online
  resource-to-observation link, but it is still a synthetic proxy and not
  installed telemetry.
- Radiometer, surface-IR, and FC4 quality remained storm-conditioned under the
  same predeclared relation, while the mandatory backbone remained at quality
  `1.0`. Rebuild frozen assets before interpreting any forecast advantage.
- Decision: V559 truth passes the linkage prerequisite. No geometry, online
  transfer, or PPO claim is authorized until the forecaster is refit on these
  files and the 32-subset gate is rerun.
- Artifact: `reports/analysis/v559_heater_quality_truth_20260909/`, including
  `quality_coupling_summary.json` and the four truth CSVs.

## 2026-09-09 V559 asset interface correction

- The first asset-only launch failed before model fitting because the generated
  truth retained `resource_effective_power_*` columns and the runner merged the
  same columns from its explicit resource trace. This was an interface error,
  not an asset or forecast result.
- `merge_dynamic_resource_trace()` now treats the explicit trace as
  authoritative by dropping stale same-named columns before the one-to-one
  merge. Local and remote Python compilation passed, and the corrected asset
  rebuild is running.

## 2026-09-09 V559 asset completion

- Asset-only regeneration completed for seeds `7241--7244`; every manifest
  reports the full 32 raw subset candidate surface and the intended
  chronological training partitions.
- The V560 geometry audit is now running on four held-out starts per seed with
  the refit V559 TCNs. No result is inferred from asset completion alone.

## 2026-09-09 V560 geometry closeout

- Refit-TCN complete-subset geometry at B=`2.15` produced condition-level
  opportunity gaps `0.00345/0.00383/0/0.00687` and operating-duty gaps
  `0.00183/0.00194/0.000087/0.000064` for seeds `7241--7244`.
- The state-dependent resource trace therefore changed feasibility without
  creating a material forecast-value frontier. Most condition winners were
  low-cost non-laser subsets, so the route still contains a static shortcut.
- Decision: close V559 before online transfer and PPO. A future route must
  create a physically justified complementary quality/resource relation, not
  merely increase heater occupancy or tune PPO against this geometry.
- Artifact: `reports/analysis/v560_heater_quality_geometry_b2p15_20260909/`.

## 2026-09-09 V561 event-specific quality truth

- V560's shared storm-risk relation was insufficient: most subset winners were
  the same non-laser allocation. V561 uses the existing declared
  particle/flux/thermal quality relation and heater restoration, so the route
  changes the physical observation relation rather than the PPO objective.
- Truth-only checks show different quality ordering by subtype in all four
  seeds. For example, the laser quality mean is about `0.22` in particle
  windows and `0.41--0.49` in flux/thermal windows, while surface-IR quality
  is about `0.30` in particle windows and `0.41--0.47` in flux/thermal windows.
- Decision: admit V561 to truth-matched frozen-asset preparation. No online
  transfer or PPO claim is authorized until the complete-subset geometry gate
  passes.

## 2026-09-09 V558 closeout

- More chronological training coverage did not improve the held-out online
  transfer probe. The failure persists with roughly six times more training
  decisions than the initial probe, while the classifier still fits the
  training labels.
- This separates two facts: V557 establishes privileged downstream value
  crossover, but V558 shows that the crossover is not reliably recoverable from
  the currently available online observation. Launching PPO here would mix a
  privileged geometry result with an unverified deployment signal, so it is
  rejected.
- The next successful route must change the observable physical/forecast link
  or define a new resource state that is genuinely available online before
  any PPO training is considered.

## 2026-09-09 V558 online transfer

- The first probe does not yet establish deployable transfer. A small MLP can
  fit the privileged best-subset labels on the training rollouts but transfers
  poorly to held-out starts. Because the training sample contains only about
  140 decision rows per seed and 15--16 label classes, a single low-sample
  failure is insufficient to distinguish limited coverage from intrinsic
  observability failure.
- The next bounded test increases chronological training-start coverage only;
  it does not add privileged features, use final-test feedback, or alter the
  PPO method.

## Requirements
- Recalibrate the PD-PPO scene to break static shortcuts.
- Keep this work independent from v1 algorithm development and ESWA manuscript
  rewriting.
- Use only `rl_sensor_scheduling_framework` code and PD-PPO results for the
  active evidence chain.
- v1 records may still be read as archived diagnostic context. Its long
  exploration without stable success is useful negative evidence for avoiding
  repeated failed routes, but it must not be merged into the PD-PPO main method
  or main result tables.
- Append each obtained result to the root `CHANGELOG.md`.
- Dynamic scheduling is now an explicit validity requirement: candidates should
  not contain multiple sensors that are permanently on or permanently off.

## Research Findings

## 2026-09-09 V557 forecast geometry

- Corrected V557 shows that B=2.15 is not merely a changing feasibility mask:
  forecast-loss geometry also changes. Across seeds, condition-level static
  opportunity is `0.00748--0.02094`, and no single candidate remains within 1%
  of the best candidate in every condition. The next risk is online transfer,
  not static-collapse geometry.

## 2026-09-09 V554 error

- A geometry audit against V554 stopped at metadata parsing because
  `normalization_start_idx` and `normalization_end_idx` were null. This is a
  protocol-construction error caused by omitted CLI partition arguments, not a
  scientific outcome. The incomplete V554 assets are not used for evidence.

## 2026-09-09 geometry accounting correction

- The geometry audit now uses a fixed-target policy with a feasible fallback,
  but records a candidate's forecast loss only on rows where that target mask
  was actually executed. Projection-fallback rows are excluded from that
  candidate's score and executable-row support is reported explicitly.
- This prevents an infeasible subset from inheriting the loss of a different
  projected subset while retaining continuous rollout state for dwell and
  warmup accounting.

## 2026-09-09 geometry metadata correction

- The first V555 geometry attempt exposed a second reproducibility mismatch:
  training assets use 15 state columns, including three subtype latent state
  columns, while the audit fallback listed only 12. The audit now uses the
  same 15-column state definition. No asset or scientific result was changed.
- An archived prior scenario screen changed the problem structure through sensor
  costs, startup peaks, warmup, event noise, event observation probability,
  energy/storage settings, and snow-transport-focused objective weights. This
  is historical context only, not active evidence.

## 2026-09-09 V555 disposition

- The corrected 32-action audit shows that the B=1.15 channel-quality scene is
  not a viable adaptive-scheduling scene. Condition-level opportunities are
  effectively zero in three seeds and only `0.03020%` in seed7241; a common
  5%-near-optimal static family remains in every seed.
- The missing operating labels prevent an operating-state claim, but do not
  rescue the condition-level geometry. No transfer probe or PPO training is
  authorized from V555.
- The relevant archived sensor pattern:
  - `laser_disdrometer`: high power/startup, useful but not a cheap static default.
  - `snow_particle_counter`: moderate cost, noisy/saturated during events.
  - `fc4_flux`: moderate cost direct snow-mass-flux channel.
  - cheap context sensors remain useful but incomplete.
- PD-PPO `SensorSpecV2` already supports `event_noise_std` and `event_observation_probability`; no runtime interface rewrite is required for this migration.
- Existing PD-PPO oracle-lift schedule diagnostics were laser-oriented. This is insufficient for the intended temporal-complementarity mechanism.
- Local auto-pair linear gate found one promising candidate:
  `transport_v6`, B=1.10, peak=1.60. It broke the laser shortcut and showed
  +1.80% overall dynamic margin and +2.05% event dynamic margin.
- The passing dynamic pair is:
  - non-event: `met_station_core|radiometer_basic|fc4_flux`;
  - event: `met_station_core|radiometer_basic|surface_temp_ir|ultrasonic_anemometer_hd`;
  - lead: 4 steps.
- Caveat: this mechanism was found under a linear frozen oracle, so it is a
  structure candidate rather than final evidence. TCN-oracle gate is required
  before PPO training.
- B=0.70 failed under TCN despite passing the linear screen. Single-window
  budget tightening is insufficient; the next correction is to align validation
  and final windows to event-transport-rich periods, matching the scientific
  target of blowing-snow monitoring.
- Event-transport-rich start selection did not fix the issue under linear smoke;
  the stronger correction is to change sensor costs so SPC and fc4 cannot be
  bundled with most context sensors in the same static subset.
- v7 diverse schedules passed duty diagnostics under a linear oracle but failed
  the TCN gate, showing that forced diversity without predictive value is not
  sufficient.
- v8 created intermittent laser/SPC/fc4 duty under a linear gate, but the loss
  margin was too weak and event-rich selection reintroduced static laser
  shortcuts.
- v8 split-pilot candidate-prior tables confirmed that the laser shortcut
  returns under the chronological split: the best prior static mask is
  `laser_disdrometer` alone.
- v9 debundled the cheap context core, but the diverse linear gate was negative
  overall. The next correction should therefore affect the learned objective,
  not only the sensor-cost scene.
- v7 remains the cleaner duty-aware PPO pilot candidate because it removes the
  laser shortcut at the tested budget, even though an earlier forced-diverse TCN
  schedule failed.
- v7 B=1.00 split-pilot candidate prior generated 88 feasible masks; the top
  candidates are SPC/context combinations and do not show a laser shortcut.
- v7 fast duty-aware PPO (`awbc=0`, `lambda_duty_balance=0.6`) failed the main
  target: PD-PPO oracle loss `0.07482` vs best static `0.07293` and AoI
  `0.07364`; duty still had 2 always-on and 2 always-off sensors.
- The likely next failure source is actor initialization/prior bias toward
  oracle static candidates plus a duty penalty that is too weak relative to the
  forecast loss.
- v7 no-prior strong-duty pilot fixed duty behavior but destroyed forecast
  quality: PD-PPO oracle loss `0.09545`, `mid=7`, `always_on=0`,
  `always_off=1`, `warmup_abort_count=24`.
- Therefore the useful operating region is between the two tested settings:
  weaker than `lambda_duty_balance=2.0`/0.10--0.90, but less static-biased than
  `lambda_duty_balance=0.6` with actor prior.
- v7 intermediate no-prior pilot was worse: PD-PPO oracle loss `0.14376` with
  5 always-off sensors. Removing the prior entirely is not viable for short
  runs; use weak prior rather than no prior.
- v7 weak-prior B=1.00 improved over no-prior but still failed: PD-PPO `0.08353`
  vs best static `0.07413`, with 2 always-on and 2 always-off sensors. B=1.00
  remains too static-friendly.
- v7 B=0.90 particle/flux improved duty (`mid=6`, `always_off=1`) but still
  failed forecast quality: PD-PPO `0.07247` vs AoI `0.06874` and feasible static
  `0.06834`. The missing piece is forecast-quality guidance during PPO, not only
  scene budget or duty shaping.
- v7 B=0.90 sparse-AWBC produced an apparent oracle-loss win but failed the
  clarified behavioral target:
  - PD-PPO oracle loss `0.06273` beat validation-selected static `0.06536`;
  - PPO selected `snow_particle_counter` for `99.66%` of final-test steps;
  - duty collapsed to `mid=0`, `always_on=1`, `always_off=7`;
  - instant MAE/DTW exploded to about `184.8`, while feasible static MAE was
    about `1.83`.
- This is a frozen-oracle shortcut, not a valid adaptive scheduler. Future
  acceptance requires action-level coverage/duty feasibility in addition to
  oracle-loss improvement.
- Enabling coverage groups on the same v7 B=0.90 sparse-AWBC setup blocked the
  one-sensor shortcut but did not solve the target:
  - feasible static projected `0.07621`;
  - round-robin `0.08239`;
  - PD-PPO `0.08448`;
  - AoI `0.08509`;
  - PD-PPO duty improved to `mid=5`, but still had `always_on=1` and
    `always_off=2`;
  - `snow_particle_counter` remained on for `100%` of steps, while `fc4_flux`
    and `laser_disdrometer` remained off for `100%`.
- Because CustomPPO uses discrete candidate masks, duty-score feedback must
  apply to `step_mask`, not only to `step_scores`. Otherwise it cannot affect
  the PPO path.
- Runtime duty-score feedback produced the first promising seed:
  - run `v7_b0p90_particle_lambda1p2_awbc0p05s16_cov_dfb2p5_prior1p0_kl0p1_ent0p003`;
  - PD-PPO oracle loss `0.08954`;
  - AoI `0.09480`;
  - round-robin `0.09756`;
  - feasible static projected `0.10007`;
  - validation-selected static `0.10360`;
  - duty `mid=7`, `always_on=0`, `always_off=1`;
  - no warmup aborts.
- This is not final evidence: full-open unconstrained remains lower (`0.07898`),
  reconstruction MAE remains weak, and radiometer/surface are near-static
  (~`92.5%`). Replication is required before expansion.
- Protocol correction: implementing duty-score feedback inside the environment
  also modified static and heuristic baselines during evaluation. Therefore the
  first feedback seed-41/seed-42 comparisons are diagnostic only, not final
  evidence. The corrected protocol keeps feedback enabled for CustomPPO
  training/evaluation but disables it for candidate prior, validation-selected
  static, feasible static, full-open, AoI, round-robin, and random baselines.
- Corrected coverage-feedback seed 41 failed under the clean protocol:
  - feasible static projected `0.07840`;
  - round-robin `0.08482`;
  - AoI `0.08738`;
  - PD-PPO `0.08757`;
  - duty `mid=6`, `always_on=0`, `always_off=1`;
  - `surface_temp_ir` remained near-always-on (`98.49%`).
- Next direction: remove coverage groups again and keep runtime duty feedback.
  This targets the original no-coverage sparse-AWBC setting, where PPO had
  oracle headroom but collapsed to single-SPC.
- No-coverage feedback seed 41 failed:
  - PD-PPO `0.11628`;
  - best validation-selected static `0.08207`;
  - duty `mid=6`, `always_on=0`, `always_off=1`;
  - switching high (`0.43936`) and reconstruction collapsed.
- Updated direction: no-coverage plus feedback over-rotates. Keep coverage
  groups and strengthen AWBC/forecast guidance instead.
- Stronger-AWBC coverage seed 41 also failed:
  - PD-PPO `0.08864`;
  - feasible static projected `0.07752`;
  - duty regressed to `mid=4`, `always_on=1`, `always_off=1`.
- Updated direction: B=0.90 is too permissive for strong fixed coverage masks.
  Test lower budget `B=0.75` before more PPO hyperparameter tuning.
- B=0.75 coverage-feedback failed:
  - PD-PPO `0.10647`;
  - best validation-selected static `0.08899`;
  - warmup aborts increased to `79`.
- Updated direction: particle/flux-heavy target weights are likely creating
  oracle shortcuts around snow sensors. Test a more balanced microclimate+snow
  target weighting before further PPO tuning.
- Balanced microclimate+snow target weighting fixed the worst behavior but not
  the main claim:
  - PD-PPO `0.13034`;
  - feasible static projected `0.12253`;
  - round-robin `0.12923`;
  - AoI `0.12983`;
  - duty was acceptable (`mid=7`, `always_on=0`, `always_off=1`,
    `switches_per_step=0.13984`, no warmup aborts).
- Updated direction: the immediate blocker is structural headroom against
  static masks, not dynamic-duty learnability. Do not expand the balanced v7
  setting to more seeds; return to scene/cost/objective gate search with the
  dynamic-duty gate active.
- Protocol mismatch found and fixed:
  - reduced split/PPO path used coverage groups by default;
  - oracle-lift calibration previously hard-coded `coverage_groups=()`;
  - this can explain why some no-coverage structural gates did not transfer to
    coverage-constrained PPO runs.
- Energy-account path exists in `25_v2_train_custom_ppo.py` and `WarmupSchedulingEnv`
  but was not exposed through `58/59` split-protocol wrappers. It is now
  available for reduced PPO after a gate passes.
- Current structural hypothesis: instantaneous budgets alone still favor compact
  static masks. A storage/harvest account is the physically meaningful way to
  create adaptive value without inventing artificial sensor-usage constraints.
- Low-budget coverage gates exposed a real projector issue:
  - coverage groups were satisfied greedily in group order;
  - high-scoring but expensive weather/surface choices could make the later
    snow-transport group infeasible even when a joint feasible coverage
    combination existed;
  - this invalidated some low-budget failures as projector artifacts rather
    than scenario evidence.
- Fixing coverage projection by small exhaustive search makes low-budget
  coverage scenarios testable again, which is important because low budgets are
  the most plausible way to break static shortcuts.
- Completed coverage+energy TCN diverse gates did not yet satisfy the target:
  - v7: 10 valid combinations, 0 strict-duty gate passes;
  - v6: 10 valid combinations, 0 strict-duty gate passes.
- The only positive dynamic margins were at too-low budgets:
  - v7 `particle_flux_v6`, B=0.60: +3.77% overall, +4.72% event;
  - v6 `particle_flux_v6`, B=0.50: +4.17% overall, +4.62% event;
  - both had `always_off=3`, so they violate the clarified no-multiple-off target.
- Therefore the useful search band is narrow: high enough for all non-laser
  snow/context sensors to be feasible (roughly B>=0.65 in v7), but low/energy-
  constrained enough that static SPC/context masks do not dominate.
- v7 B=0.65 candidate inspection showed that FC4 is too weak:
  - best static and best event masks still use SPC;
  - FC4 static candidates have much higher event loss;
  - diverse schedules that include FC4 improve duty but lose oracle quality.
- New v10 hypothesis:
  - make SPC less reliable specifically during events;
  - make FC4 a cleaner event flux channel;
  - keep SPC useful outside events for particle microstructure;
  - lower met core cost enough that it is not structurally always off.
- First v10 TCN row supports the direction:
  - `particle_flux_v6`, B=0.65 has dynamic margin `+0.81%` and event margin
    `+3.79%`;
  - dynamic duty passes (`mid=7`, `always_on=0`, `always_off=1`);
  - it narrowly misses the stricter `+1%` overall gate but is the first TCN
    result that satisfies the clarified behavioral target and shows positive
    headroom.
- v10 B=0.70 is the first strict TCN gate pass:
  - `particle_flux_v6`, B=0.70;
  - dynamic margin `+2.18%`, event margin `+2.70%`;
  - dynamic duty passes (`mid=7`, `always_on=0`, `always_off=1`,
    `switches_per_step=0.03565`).
- Current best candidate for reduced PPO is v10, `particle_flux_v6`, B=0.70.
- Clarified acceptance target:
  - the scene must force meaningful dynamic scheduling, not just lower oracle
    loss;
  - candidates are invalid if several sensors become permanently on or
    permanently off;
  - current operational filter is `mid_duty_sensor_count >= 5`,
    `always_on_sensor_count <= 1`, `always_off_sensor_count <= 1`, and nonzero
    bounded switching, with final interpretation requiring no multiple
    always-on/off sensors.
- First completed v10 PPO probe is genuinely positive:
  - run `v10_b0p65_particle_energy_cov_dfb2p5_prior1p0_kl0p1_ent0p003_seed41`;
  - PD-PPO oracle loss `0.14945`;
  - feasible static `0.15142`, round-robin `0.15380`, AoI `0.15581`,
    random `0.16304`, validation-selected static `0.16891`;
  - duty passes the clarified target (`mid=7`, `always_on=0`,
    `always_off=1`, `switches_per_step=0.12283`);
  - sensor use is dynamic rather than static: met `86.74%`, radiometer
    `26.27%`, surface `86.67%`, ultrasonic `6.81%`, shielded `6.76%`,
    SPC `84.11%`, laser `0%`, FC4 `15.89%`.
- Interpretation of the B=0.65 PPO result:
  - it is the first clean reduced-PPO result satisfying the dynamic-duty target
    while beating the constrained static and dynamic baselines on oracle loss;
  - it is not final evidence because it is one seed and B=0.65 only narrowly
    missed the strict TCN structural margin;
  - B=0.70 remains the priority confirmation because it was the first strict
    TCN gate pass.
- B=0.70 PPO did not transfer despite strict TCN gate success:
  - run `v10_b0p70_particle_energy_cov_dfb2p5_prior1p0_kl0p1_ent0p003_seed41`;
  - validation-selected static `0.14722`;
  - AoI `0.15631`;
  - feasible static `0.16009`;
  - round-robin `0.16148`;
  - PD-PPO `0.16170`;
  - random `0.17723`;
  - full-open unconstrained under energy guard `0.18616`.
- B=0.70 duty itself was acceptable:
  - `mid=7`, `always_on=0`, `always_off=1`,
    `switches_per_step=0.24597`;
  - sensor use: met `74.10%`, radiometer `70.58%`, surface `70.68%`,
    ultrasonic `40.21%`, shielded `40.14%`, SPC `85.96%`, laser `0%`,
    FC4 `8.86%`.
- Interpretation of B=0.70:
  - the scenario has structural dynamic headroom, but the PPO reduction did not
    learn a lower-loss policy;
  - B=0.70 should not be promoted unless later seeds or hyperparameters reverse
    the result;
  - the immediate replication target is B=0.65 because it is the only learned
    positive candidate.
- B=0.65 seed 42 did not replicate:
  - validation-selected static `0.12743`;
  - PD-PPO `0.13797`;
  - round-robin `0.13960`;
  - AoI `0.14189`;
  - random `0.14348`;
  - feasible static `0.15467`;
  - full-open unconstrained under energy guard `0.20734`.
- B=0.65 seed 42 fails the clarified duty target:
  - `mid=5`, `always_on=1`, `always_off=2`,
    `switches_per_step=0.19734`;
  - sensor use: met `0.22%`, radiometer `99.90%`, surface `65.14%`,
    ultrasonic `60.21%`, shielded `69.51%`, SPC `89.94%`, laser `0%`,
    FC4 `10.06%`.
- The seed-42 validation-selected static shortcut is explicit:
  - selected mask `radiometer_basic|ultrasonic_anemometer_hd|shielded_thermo_hygro|snow_particle_counter`;
  - power `0.64`, just below B=0.65;
  - this mask leaves met, surface, laser, and FC4 off, yet gets the best oracle
    loss.
- Updated structural correction:
  - v10 is close but still too static-friendly at B=0.65;
  - increase SPC cost slightly so
    `radiometer+ultrasonic+shielded+SPC` becomes infeasible at B=0.65;
  - preserve feasible alternatives `met+radiometer+SPC` and
    `met+radiometer+FC4` so dynamic event/non-event switching remains possible.
- v11 implements that correction with a minimal cost-only change:
  - sensor config `windblown_sensors_physical_event_v11_spc_static_break.yaml`;
  - SPC steady power `0.40 -> 0.43`, startup peak `0.56 -> 0.58`;
  - old shortcut now costs `0.67` at B=0.65 and is infeasible;
  - met+radiometer+SPC (`0.63`), met+radiometer+FC4 (`0.62`), and
    met+surface+SPC (`0.65`) remain feasible;
  - all non-laser sensors still appear in feasible masks.
- v11 linear probe failed the clarified target:
  - B=0.65: dynamic margin `+0.69%`, event margin `+0.98%`,
    but duty collapsed (`mid=0`, `always_on=3`, `always_off=5`);
  - B=0.70: dynamic margin `+0.54%`, event margin `+0.26%`,
    also collapsed (`mid=0`, `always_on=3`, `always_off=5`);
  - conclusion: raising SPC cost breaks one static shortcut but makes the best
    dynamic candidates near-static, so v11 should not be promoted to PPO.
- v10 narrow-budget linear probes also failed:
  - B=0.62 cuts the seed-42 0.64 static shortcut and gives dynamic margin
    `+0.998%`, but event margin is only `+0.73%` and duty still has
    `always_off=2`;
  - B=0.63 gives apparent headroom but the best dynamic is near-static
    (`mid=0`, `always_on=3`, `always_off=5`);
  - conclusion: budget-only micro-tuning is insufficient.
- Gate script correction:
  - `63_v31_static_break_calibration.py` previously allowed `gate_pass=True`
    when dynamic diversity failed unless `--require-diverse-dynamic` was set;
  - since dynamic duty is now a hard objective, `gate_pass` now always requires
    `dynamic_diversity_ok`.
- Hard duty guard seed 42 result:
  - run `v10_b0p65_particle_energy_cov_hguard_l08h90s12_dfb2p5_seed42`;
  - PD-PPO `0.13873`;
  - validation-selected static `0.13672`;
  - round-robin `0.14317`;
  - AoI `0.14351`;
  - feasible static `0.15921`;
  - duty target is fixed (`mid=7`, `always_on=0`, `always_off=1`);
  - sensor duty: met `8.01%`, radiometer `92.99%`, surface `89.94%`,
    ultrasonic `21.46%`, shielded `74.27%`, SPC `88.62%`, laser `0%`,
    FC4 `11.38%`.
- Interpretation of hard duty guard:
  - action-layer guarding solves the clarified behavior target on the failing
    seed 42 case;
  - forecast loss remains `~1.47%` worse than validation-selected static;
  - the next variant should reduce hard-guard force rather than change the
    scene again.
- Milder hard guard with score 8 failed:
  - run `v10_b0p65_particle_energy_cov_hguard_l08h90s8_dfb2p5_seed42`;
  - PD-PPO `0.14511`;
  - validation-selected static `0.13533`;
  - round-robin `0.14237`;
  - AoI `0.14318`;
  - duty remained valid (`mid=7`, `always_on=0`, `always_off=1`) but forecast
    loss worsened relative to score 12.
- Updated interpretation:
  - hard guarding is behaviorally effective, but score=8 is too weak/unstable
    for forecast quality on seed 42;
  - score=12 remains the best hard-guard setting tested so far;
  - next check is whether score=12 preserves the original seed-41 positive
result.

## 2026-09-10 Dynamic-resource unit audit

The environment does apply the dynamic resource guard in `is_mask_executable`
and `dynamic_resource_cost`; it is not merely a logging feature. However, the
resource interface had a reproducibility ambiguity. The physical trace
generator writes both `resource_power_w_*` and `resource_effective_power_*`,
while the training entry point previously selected only the latter. Earlier
V627--V645 artifacts contain physical-looking values (for example 51.5 W for
the heated laser) under the `resource_effective_power_*` name. The current
generator defines that prefix as normalized acquisition cost and reserves the
`_power_w_` prefix for watts.

The entry point now prefers explicit `resource_power_w_*` columns and keeps
the old effective-prefix path only for legacy compatibility. The normalized
steady/startup projector remains a separate declared interface constraint;
the dynamic resource guard consumes the physical-watt trace and its watt
budget. Existing V627--V645 scientific conclusions remain provisional until
the corrected mapping is used to regenerate matched geometry.

## 2026-09-10 V646 corrected geometry result

The matched V646 rebuild uses the same V645 truth, starts, normalized
interface budgets, dwell, candidate masks, and frozen forecasters, but its
metadata maps the dynamic 55 W guard to explicit `resource_power_w_*`
columns. The operating opportunity gaps are `0.000000`, `0.017026`,
`0.002626`, and `0.005402` for seeds `7401--7404`; only `1/4` meets the
predeclared `0.01` materiality threshold. Seed7401 keeps candidate_003 as the
winner in every heater condition and retains a 1% near-optimal intersection;
seeds7403--7404 retain candidate_005 within 5% across operating conditions.

The unit correction therefore changes the measured geometry but does not
reopen the route. The physical resource mapping is now reproducible, while
the downstream adaptive opportunity remains insufficient for online transfer
or PPO. The compact summary is stored in
`reports/v646_unit_reconciled_geometry_b2p15_20260910/unit_reconciled_summary.md`.
- Hard guard score 12 did preserve the seed-41 positive result:
  - PD-PPO `0.14456` beat feasible static `0.15310`, round-robin `0.15560`,
    AoI `0.15580`, random `0.16254`, validation-selected static `0.16887`,
    and full-open unconstrained under the energy guard `0.16776`;
  - duty remained valid with `mid=7`, `always_on=0`, `always_off=1`, and
    `switches_per_step=0.13004`.
- Current evidence is therefore mixed rather than negative:
  - seed 41 is cleanly positive under the hard guard;
  - seed 42 is behaviorally valid and beats feasible static/dynamic heuristics
    but trails validation-selected static by about `1.47%`;
  - seed 43 is the necessary next run before deciding whether to expand or
    modify the protocol.
- Seed 43 failed and changes the interpretation from "mixed but promising" to
  "not stable as-is":
  - PD-PPO `0.17423` lost to round-robin `0.14293`, feasible static `0.14522`,
    AoI `0.14945`, random `0.15293`, and validation-selected static `0.15407`;
  - coarse duty metrics still passed (`mid=7`, `always_on=0`, `always_off=1`),
    but sensor duty showed several near-high-boundary channels:
    radiometer `92.77%`, surface `89.94%`, shielded `86.89%`, SPC `89.94%`;
  - met, ultrasonic, and FC4 were near the low boundary (`~8--10%`).
- Updated diagnosis:
  - hard duty guard prevents literal always-on/off collapse but can still
    produce quasi-static boundary policies;
  - the next correction should tighten the high-duty boundary and reduce static
    prior/AWBC pull before changing the physical scene again.
- Seed43 anti-static variants confirm the diagnosis but do not yet pass:
  - tightening duty high/low alone improved PD-PPO from `0.17423` to
    `0.15519`;
  - tightening duty plus weaker prior/AWBC improved it further to `0.15036`;
  - the weak-prior variant beat AoI, validation-selected static, random, and
    full-open under the energy guard, but still lost to round-robin `0.14319`
    and feasible static `0.14507`;
  - duty also improved from near-low/high counts `4/4` to `2/1`.
- Updated direction:
  - the current best correction is real but incomplete;
  - next tests should check whether the remaining gap is due to short 40k
    training or because the hard high boundary still allows too much
    radiometer-dominant behavior.
- Follow-up tests ruled out both simple fixes:
  - h80/low-high `0.15/0.80` improved duty diversity but worsened PD-PPO to
    `0.17236`;
  - 100k training improved duty diversity further but worsened PD-PPO to
    `0.16230` with `357` warmup aborts;
  - the best seed43 variant remains 40k h85 weak-prior (`0.15036`), which
    beats AoI/validation-static/random but still loses to round-robin and
    feasible static.
- Updated direction:
  - do not continue duty tightening or longer training under the same reward;
  - inspect event-window evaluation/scene pressure and operationally constrained
    heuristic baselines, because uniform low-event final windows keep making
    compact static or round-robin policies competitive.
- Event-window evaluation does not explain the seed43 failure:
  - explicit event-window eval on starts `55500`, `56917`, `58697` produced
    event rate `0.34408`;
  - PD-PPO stayed behaviorally dynamic (`mid=7`, `always_on=0`,
    `always_off=1`, `switches_per_step=0.18923`) but oracle loss was
    `0.16880`;
  - feasible static remained best at `0.15500`, round-robin was `0.15951`,
    AoI was `0.16429`, and validation-selected static was `0.16764`;
  - conclusion: the failure is not merely low event density in uniform final
    windows. Static bundles remain too strong in the scene.
- Next structural direction:
  - target the radiometer/shielded/SPC and radiometer/SPC static bundles
    directly;
  - preserve feasible dynamic alternatives involving met/radiometer/SPC and
    met/radiometer/FC4;
  - do not spend more runs on longer training, tighter hard-duty bounds, or
    event-only evaluation until the static bundle is weakened.
- Important wrapper finding:
  - the truth generator supports event microstructure, but the active
    calibration/split wrappers were not forwarding the microstructure
    parameters;
  - default wrapper behavior therefore kept `event_microstructure_sigma=0.0`,
    making particle/flux targets largely explainable from static meteorological
    context;
  - this is a plausible structural cause of static shortcuts and should be
    tested before more PPO hyperparameter work.
- Current structural test:
  - run v10 with event microstructure enabled on the server;
  - compare `sigma=0.8` and `sigma=1.2`;
  - pass criteria remain strict dynamic duty plus positive dynamic margin.
- v10 microstructure with coverage is close but not enough:
  - `sigma=0.8`, B=0.58 produced positive dynamic margin `+1.48%` and event
    margin `+0.65%`, but met was always off because the budget cannot fit
    met+surface+snow coverage;
  - B>=0.60 restores met feasibility but also restores the strong static
    `met+radiometer+SPC` shortcut;
  - next promising structural move is v11-style SPC cost increase combined
    with event microstructure, likely around B=0.62.
- v11 microstructure did not fix the shortcut:
  - B=0.62 gets valid dynamic duty, but `met+radiometer+FC4` becomes the best
    static and beats dynamic;
  - increasing microstructure further worsens transfer;
  - next objective should make both one-modality snow triads incomplete:
    only-SPC should miss direct flux and only-FC4 should miss particle
    microstructure.
- Flux/particle decorrelation is directionally useful but insufficient at the
  tested amplitudes:
  - v10 B=0.58 can reach `+3.28%` dynamic margin, but met is infeasible and
    always off;
  - v13 made the intended complementary masks feasible but still lost to static
    triads;
  - next test should increase both flux sigma and particle perturbation scale
    with correlation fixed near zero.
- High-amplitude decorrelation did not help:
  - v10 high-amplitude gate had `0/6` passes; best margin was already negative
    (`-1.82%`) and strict-diversity rows were around `-11%` or worse;
  - v13 high-amplitude gate had `0/8` passes; the best strict-diversity row was
    `-6.26%`;
  - increasing microstructure amplitude makes the task harder but does not
    remove compact static-triad dominance.
- Current structural conclusion:
  - the scene-family search did not find a strict positive candidate satisfying
    both dynamic headroom and dynamic-duty constraints;
  - the next useful comparison is not another small scene tweak, but an
    operational-baseline audit: original heuristics remain reported, while
    deployment-style duty-constrained heuristics are evaluated separately.
- Operational-baseline audit result:
  - PD-PPO beats the best duty-constrained heuristic baseline in all three
    representative replays:
    seed41 by `10.42%`, seed42 by `4.17%`, seed43 by `1.91%`;
  - original unconstrained round-robin remains stronger than PD-PPO for seed43,
    showing that the operational constraint matters rather than merely
    renaming baselines;
  - validation-selected static remains stronger than PD-PPO for seed42, so the
    static-baseline limitation is not solved by constraining heuristics.
- Claim boundary after this round:
  - acceptable: PD-PPO can produce smoother dynamically varying schedules that
    outperform deployment-style constrained heuristics;
  - not acceptable: PD-PPO uniformly dominates selected static allocations or
    all unconstrained heuristic schedules.
- No-warmup full-grid partial result after 15 completed runs:
  - B=1.65 remains strong against selected/static (`9/10`) but never beats the
    best fair non-PPO baseline (`0/10`);
  - B=1.70 is now `5/5` against selected/static but only `1/5` against the
    best fair non-PPO baseline;
  - new B=1.70 seed45 repeats the same failure mode:
    PD-PPO `0.12826`, round-robin `0.12761`, AoI `0.12765`, static
    `0.15625`;
  - duty remains invalid without hard guard, with seed45 at
    `mid=4`, `always_on=1`, `always_off=3`.
- Updated no-warmup interpretation:
  - removing warmup is a useful way to break selected-static performance;
  - by itself it does not meet the clarified dynamic-scheduling target;
  - promotion depends on the hard-duty continuation, not on the full-grid
    no-warmup baseline alone.
- No-warmup hard-duty continuation after seeds 41--42:
  - hard duty is doing what it was designed to do: both runs have
    `mid=8`, `always_on=0`, `always_off=0`;
  - performance does not transfer: seed41 beats static but loses round-robin
    and the best constrained baseline; seed42 loses to selected/static and
    round-robin while only beating constrained/AoI/random baselines;
  - aggregate status: static win `1/2`, best original fair baseline win
    `0/2`, best duty-constrained win `1/2`;
  - current conclusion: this is a behavioral fix, not a sufficient main-result
    scene.
- No-warmup hard-duty final result after seeds 41--43:
  - dynamic-duty validity is `3/3`;
  - PD-PPO beats validation-selected/static in only `2/3`;
  - PD-PPO beats the best original fair/dynamic baseline in `0/3`;
  - PD-PPO beats the best duty-constrained baseline in only `1/3`;
  - seed43 confirms the pattern: PD-PPO `0.13745` beats selected/static
    `0.14597`, but loses to round-robin `0.13406`, AoI `0.13502`, and best
    duty-constrained baseline `0.13648`.
- Updated direction:
  - do not continue no-warmup + hard-duty PPO training as the main route;
  - the remaining plausible operational argument is not duty alone but
    switching realism: round-robin/AoI still switch much more frequently than
    PD-PPO and should be tested under minimum-dwell or switch-rate constraints;
  - original unconstrained rows must remain reported side by side.
- Env-dwell12 replay completed:
  - applying `min_dwell_steps=12` uniformly at the environment level gives
    PD-PPO wins against the best original dynamic heuristic in `3/3` seeds and
    against the best duty-constrained baseline in `2/3` seeds;
  - PD-PPO beats validation-selected/static in only `1/3` seeds;
  - all PD-PPO rows have `mid=8`, `always_on=0`, `always_off=0`, and zero
    warm-up aborts.
- Trained env-dwell12 seed41 is the first genuinely useful positive result on
  this operational branch:
  - PD-PPO `0.132886` vs validation-selected static `0.137648`,
    feasible static `0.146423`, round-robin `0.158871`, AoI `0.141682`,
    and best duty-constrained baseline `0.134288`;
  - behaviour is valid: `mid=8`, `always_on=0`, `always_off=0`,
    `switches_per_step=0.024377`, zero aborts.
- Compliance with the current policy is now partial but real:
  - satisfied: minimum dwell is enforced in the execution environment for all
    policies in the env-dwell replay/training branch;
  - satisfied: original unconstrained heuristic rows are retained rather than
    hidden;
  - satisfied: constrained dynamic baselines are reported as a separate view;
  - not yet fully satisfied: maximum switch rate and duty upper/lower bounds
    are not yet uniformly imposed on every dynamic policy as one single
    deployment contract. Duty-constrained rows exist, but they remain an
    additional baseline family.
- Paper implication:
  - do not replace the locked conservative fixed-budget table with the single
    trained env-dwell12 seed;
  - if seeds 42--43 replicate, the operational constrained branch can become
    a secondary positive result; otherwise it remains an appendix diagnostic.
- No-warmup main grid partial refresh after 17 completed rows:
  - B=1.65: PD-PPO beats static `9/10`, beats best original dynamic only
    `1/10`, and has valid dynamic duty `0/10`;
  - B=1.70: PD-PPO beats static `7/7`, beats best original dynamic only
    `1/7`, and has valid dynamic duty `0/7`;
  - conclusion: no-warmup is a static-break lever but not a complete
    deployable scheduling result. It should not trigger English paper writing
    unless paired with uniform operational constraints that replicate beyond
    seed41.
- Trained env-dwell12 seed42 result:
  - PD-PPO `0.149620` loses to best static `0.138138`;
  - it still beats best original dynamic `0.154632` and best duty-constrained
    `0.160709`;
  - behaviour remains excellent: `mid=8`, `always_on=0`, `always_off=0`,
    `switches_per_step=0.026860`, zero aborts.
- Current trained env-dwell12 conclusion:
  - deployment behaviour and dynamic-baseline advantages are now replicated
    across seeds 41--42;
  - static superiority is not replicated (`1/2`);
  - English paper writing should still wait for seed43, and the final claim
    must remain conservative unless seed43 passes the static gate.
- Trained env-dwell12 final 3-seed result:
  - seed41: PD-PPO `0.132886` beats best static `0.137648`, best original
    dynamic `0.141682`, and best duty-constrained non-PD-PPO `0.134288`;
  - seed42: PD-PPO `0.149620` loses to best static `0.138138`, but beats best
    original dynamic `0.154632` and best duty-constrained non-PD-PPO
    `0.160709`;
  - seed43: PD-PPO `0.140702` beats best static `0.144098`, best original
    dynamic `0.150048`, and best duty-constrained non-PD-PPO `0.151801`;
  - aggregate: static `2/3`, original dynamic `3/3`, duty-constrained
    non-PD-PPO `3/3`, deployment behaviour `3/3`.
- Mainline baseline audit conclusion:
  - answer is qualified yes, not full static dominance;
  - PD-PPO genuinely beats the fair dynamic and duty-constrained baseline
    families under env-dwell12;
  - it does not beat full observation and does not uniformly or on average beat
    the strongest validation-selected static shortcut;
  - removing seed42 from the main result would be cherry-picking unless a
    data/config bug is found.
- Current active blocker for the original static-break goal:
  - the selected static shortcut can still exploit final-test sequences where a
    compact mask captures enough forecast variance;
  - PD-PPO is now more deployable (`mid=8`, no always-on/off), but that broader
    duty contract is not yet imposed on the selected static baseline in the same
    way;
  - the next useful step is not deleting bad seeds, but inspecting seed42's
    static mask and testing either a fair deployable-static comparator or one
    more narrow scene calibration that weakens compact static masks.
- Direct deployable selected-static replay:
  - adding `duty_constrained_validation_selected_static` shows that the static
    shortcut is mostly but not fully broken under deployable duty constraints;
  - PD-PPO beats deployable selected static in seeds `41` and `43`, loses seed
    `42` by only `0.000271`, and still beats best duty-constrained non-PD-PPO
    in `3/3`;
  - the remaining static advantage is tied to high radiometer/SPC duty in seed
    `42`, so the next diagnostic is a symmetric stricter duty-high run.
- H75 diagnostic:
  - launched `duty-high=0.75` reduced retrain for seeds `41`--`43`;
  - this is acceptable only because the same bound is applied to PD-PPO and
    baselines and all seeds remain included;
  - treat the result as a deployment-constraint sensitivity test until the full
    3-seed table is available.
- H75 seed41:
  - PD-PPO remains deployment-valid (`mid=8`, no always-on/off, max duty
    `0.742350`) and beats all fair baseline families;
  - it only narrowly beats deployable selected static (`0.132783` vs
    `0.133001`), so the branch still depends on seed42/43 replication.
- H75 seed42:
  - the stricter duty-high setting fixes the previous deployable-static loss:
    PD-PPO `0.148363` vs deployable selected static `0.150508`;
  - it still loses to the original compact static shortcut (`0.137324`), which
    has `3` always-on and `5` always-off sensors and should not be treated as a
    deployable comparator;
  - after two seeds, h75 is `2/2` against deployable selected static, original
    dynamic, and duty-constrained non-PD-PPO baselines, with valid behaviour
    `2/2`.
- H75 final:
  - this branch is the strongest operational result so far;
  - PD-PPO wins `3/3` against deployable selected static, original dynamic
    heuristics, and duty-constrained non-PD-PPO baselines;
  - it keeps valid deployment behaviour in `3/3` seeds;
  - it still wins only `1/3` against the original compact static shortcut, so
    manuscript claims must present that row as an undeployable diagnostic rather
    than as the fair operational comparator.
- H75 final 5-seed expansion:
  - locked-parameter extension changes the conclusion from perfect 3-seed
    operational dominance to a more honest robust-positive result;
  - PD-PPO wins original dynamic heuristics `5/5`, deployable selected static
    `4/5`, duty-constrained non-PD-PPO `4/5`, original compact static `3/5`,
    and full-open `0/5`;
  - deployment behaviour remains valid `5/5`;
  - seed45 is the boundary case: it remains deployable and beats original
    dynamic/static, but loses to deployable selected static and duty-constrained
    round-robin;
  - this is sufficient for a supervisor-facing positive draft if framed as
    operational constrained scheduling rather than universal static dominance.
- H75 10-seed audit:
  - deployable selected static is still not beaten comprehensively:
    `4/10`, mean baseline-minus-PD-PPO delta `-0.000320`;
  - budget and dwell sensitivities did not fix this (`5/10`, `1/10`, `4/10`,
    and `6/10` deployable-static wins across the checked sensitivity tables);
  - the user's stronger requirement therefore requires scene recalibration,
    not manuscript reframing.
- v14 boundary-switch hypothesis:
  - previous v13 gates failed, so v14 targets the concrete seed42 shortcut
    rather than repeating generic decorrelation;
  - the problematic `radiometer+ultrasonic+shielded+SPC` static bundle is moved
    to cost `0.71`, above B=0.60/0.65;
  - complementary dynamic alternatives remain feasible:
    `met+SPC` (`0.58`) and `met+radiometer+FC4` (`0.59`);
  - acceptance requires PD-PPO to beat deployable selected static under the
    same h75 duty and dwell constraints, not merely beat unconstrained static.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Add `windblown_sensors_physical_event_v6_static_break.yaml` | Keeps the recalibrated scene explicit and reproducible. |
| Add `--schedule-family v6_static_break` to oracle-lift | Allows dynamic diagnostic schedules that match the intended SPC/fc4/context mechanism. |
| Add `63_v31_static_break_calibration.py` | Provides a repeatable budget/objective search before PPO training. |
| Add automatic event/non-event static-pair schedules | The first manual dynamic gate failed partly because fixed hand-written schedules did not target the actual top event/non-event static masks. |
| Add duty-balance metrics and PPO shaping | The clarified target is behavioral, so it must be measured and made learnable rather than inspected only after training. |
| Keep v8 pilots but launch v7 in parallel | v8 is informative for duty shaping, but its candidate prior is laser-dominated and therefore not acceptable as final scene evidence. |
| Treat single-sensor oracle-loss wins as invalid | The sparse-AWBC run showed that oracle loss alone can be exploited by near-static SPC selection with catastrophic reconstruction error. |
| Apply runtime duty feedback to discrete masks | Coverage groups alone leave SPC permanently selected; feedback must modify projection behavior for CustomPPO candidate masks. |
| Treat the feedback result as a candidate, not a conclusion | It passes the first duty+oracle gate on seed 41 but still needs seed replication and mechanism cleanup. |
| Do not apply runtime duty feedback to baselines | Baselines must retain their intended semantics; otherwise static and heuristic comparisons are contaminated. |
| Evaluate duty-constrained baselines as a separate class | This avoids contaminating original baselines while testing whether PD-PPO's smoother schedules are advantaged under realistic switching/duty restrictions. |
| Keep original and constrained baselines side by side | The constrained view is operationally meaningful, but the original static/round-robin rows are still necessary for scientific honesty. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Multiple planning files exist across root, archived v1, and PD-PPO | Created isolated plan under `.planning/2026-06-07-pd-ppo-static-break-recalibration/`; v1 can be read as failed-route memory but is not part of the active PD-PPO evidence chain. |
| First linear gate found no dynamic headroom | Keep result as failed calibration evidence, then test automatic event/non-event static-pair schedules before changing the physical scene again. |

## Resources
- Active isolated plan: `.planning/2026-06-07-pd-ppo-static-break-recalibration/task_plan.md`
- PD-PPO sensor config: `configs/sensors/windblown_sensors_physical_event_v6_static_break.yaml`
- Oracle-lift diagnostic: `scripts/49_v31_physical_event_oracle_lift.py`
- Calibration gate: `scripts/63_v31_static_break_calibration.py`

## Seed48 Diagnosis

### Finding
- Seed48 is special because the final-test segment is unusually favourable to
  the compact `met_station_core|radiometer_basic|snow_particle_counter` static
  shortcut, not because PD-PPO becomes invalid.

### Evidence
- In the h75 reduced 10-seed table, seed48 is simultaneously:
  - the best PD-PPO seed: `custom_ppo=0.126287`;
  - the best selected-static seed: `validation_selected_static=0.118698`;
  - the best deployable selected-static seed:
    `duty_constrained_validation_selected_static=0.126034`.
- Rollout audit for h75 reduced seed48:
  - PD-PPO event/non-event loss: `0.156048 / 0.116332`;
  - selected static event/non-event loss: `0.145755 / 0.109648`;
  - deployable selected static event/non-event loss: `0.154124 / 0.116638`;
  - PD-PPO remains deployment-valid: `mid=8`, no always-on/off, switch rate
    `0.030360`.
- Truth-segment audit:
  - event rate is `0.250651`, below the 10-seed mean `0.277588`;
  - event mean snow-particle velocity is `1.09544`, the lowest of 10 seeds;
  - event mean particle diameter is `0.03219`, the lowest of 10 seeds;
  - event mean snow-mass flux is also the lowest of 10 seeds
    (`4.23382e-06`).

### Interpretation
- The event windows in seed48 are weak and particle microstructure is easy, so a
  simple SPC/radiometer static mask remains highly predictive across both
  event and non-event portions.
- PD-PPO still learns a valid smooth schedule and beats dynamic baselines, but
  it has no structural advantage over the static shortcut in this particular
  weak-event regime.
- The next useful correction is scene-level: make event particle complexity and
  cross-sensor complementarity stronger, or make the `met+radiometer+SPC`
  compact mask less sufficient. Further PPO hyperparameter tuning alone is not
  expected to solve this seed class.

## V15 Deployable-Static Gate

### Finding
- The v15 scene creates positive dynamic headroom only when the static shortcut
  is evaluated under the same deployable duty guard used in the operational
  claim.

### Evidence
- Completed deployable-static gate rows show:
  - `micro_flux_v6`, B=1.10: dynamic `0.688373` vs deployable static
    `0.692592`, margin `+0.609%`, event margin `+1.232%`.
  - `micro_flux_v6`, B=1.15/B=1.20: dynamic `0.696698` vs deployable static
    `0.700701`, margin `+0.571%`, event margin `+0.910%`.
  - `flux_micro_v6`, B=1.15: dynamic `0.750564` vs deployable static
    `0.755691`, margin `+0.678%`, event margin `+0.946%`.
  - `flux_micro_v6`, B=1.10 fails overall.
- Raw static remains stronger and laser-heavy:
  `met_station_core|radiometer_basic|laser_disdrometer`.
- The deployable static replay is behaviorally valid (`mid=8`, no always-on/off),
  so this is a fairer structural reference than raw always-on static.

### Interpretation
- v15 is not a universal static-break scene yet; it is an operational
  static-break candidate.
- `micro_flux_v6`, B=1.15 is the best first PPO transfer point because it has
  multiple laser-feasible masks, a positive deployable-static margin, and avoids
  the too-tight B=1.10 edge case.
- The next experiment should be a single-seed PPO learnability probe before any
  3- or 10-seed expansion.
- Final structural gate result is 5/6 pass against deployable static. This
  supports launching one PPO probe, not yet claiming learned-policy success.

## V15 PPO Probe

### Finding
- The first v15 PPO probe failed despite valid duty behavior.

### Evidence
- Seed41, `micro_flux_v6`, B=1.15:
  - PD-PPO `0.289244`;
  - deployable selected static `0.286190`;
  - best duty non-PD-PPO `0.279024`;
  - best original dynamic `0.283132`;
  - validation-selected static `0.269418`.
- PD-PPO behavior is not the main issue:
  - `mid=8`, no always-on/off;
  - switch rate `0.038523`;
  - one warmup abort.
- Sensor audit shows PD-PPO underuses the event sensors that matter under the
  TCN oracle:
  - laser duty `0.121`, FC4 duty `0.123`;
  - met/radiometer near duty-high and SPC `0.539`;
  - deployable selected static raises laser event duty to `0.552` and wins.

### Interpretation
- The v15 structural pass was produced by a linear oracle. PPO uses a TCN oracle,
  and the transfer failed.
- Further PPO tuning on this setting would be blind. The next required filter is
  a TCN deployable-static gate using the same v15 scene and comparator.

## V15 TCN Gate

### Finding
- The first TCN structural gate row passed, so the v15 scene still has real TCN
  oracle headroom.

### Evidence
- `micro_flux_v6`, B=1.15, peak 1.55:
  - deployable static loss `0.574457`;
  - best eligible dynamic loss `0.562486`;
  - dynamic margin `+2.08%`;
  - event margin `+2.39%`.

### Interpretation
- The previous PPO failure should be treated as a learnability/credit-assignment
  problem, not as proof that v15 cannot break deployable static.
- Next PPO probe should increase teacher guidance or explicitly bias event-sensor
  use; repeating the same PPO controls is not justified.

## V15 PPO Transfer Diagnostics

### Finding
- Medium online-greedy teacher guidance still fails to transfer the TCN dynamic
  headroom into PD-PPO.

### Evidence
- `v31_static_break_v15_micro_flux_ppo_teacher_mid_20260608`, seed41:
  - PD-PPO loss `0.293273`;
  - validation-selected static `0.272686`;
  - best duty non-PD-PPO `0.279821`;
  - best original dynamic `0.283346`;
  - deployable selected static `0.288991`.
- Behaviour was not collapsed:
  - `mid=8`, `always_on=0`, `always_off=0`, switch rate `0.034066`.
- But strict deployment validity still failed:
  - `warmup_abort_count=5`.
- Sensor mechanism:
  - PD-PPO laser event duty `0.134011`;
  - PD-PPO FC4 event duty `0.119224`;
  - validation-selected static uses laser event duty `0.809612`.

### Interpretation
- The learned policy is smooth and duty-balanced, but it still does not adopt
  the event-channel mechanism that produces the TCN structural headroom.
- The issue is no longer generic exploration or duty collapse; it is a teacher
  target mismatch. Online greedy AWBC spreads labels over many actions, while
  the structural gate's best valid mechanism is a small event-conditioned pair.
- The next useful test is explicit event-pair imitation. The current selected
  teacher is:
  - calm: `surface_temp_ir|ultrasonic_anemometer_hd|shielded_thermo_hygro|snow_particle_counter`;
  - event: `met_station_core|radiometer_basic|laser_disdrometer`.

### Event-Pair Teacher Result
- Explicit event-pair imitation worked in the intended direction but did not
  pass the full baseline gate.
- Seed41 eventpair result:
  - PD-PPO `0.287013`;
  - deployable selected static `0.286131`;
  - best original dynamic `0.282018`;
  - best duty non-PD-PPO `0.278138`.
- Event loss improved from the medium-teacher value `0.563039` to `0.535070`.
- Non-event loss worsened to `0.197963`.
- The mechanism is now different:
  - event laser duty increased to `0.505545`;
  - but non-event met duty fell to `0.172860`.
- Interpretation:
  - explicit event-pair teacher is the right mechanism lever;
  - the first pair used a poor calm mask for final-test non-event windows.
  - eventpair2 should use the TCN summary pair:
    calm `met+radiometer+shielded+SPC`, event `met+surface+laser`.

### Exact Event-Pair Replay
- Eventpair2 did not pass after PPO training:
  - PD-PPO `0.288980`;
  - deployable selected static `0.287362`;
  - best original dynamic `0.283795`;
  - best duty non-PD-PPO `0.279684`.
- Exact event-pair replay on the same saved final split revealed that the
  problem is not event-pair control itself:
  - exact `ep4` (`met+radiometer+surface+SPC` in calm, `met+radiometer+laser`
    in event) achieved `0.278440`;
  - exact `ep3` (`met+radiometer+surface+SPC` in calm, `met+surface+laser` in
    event) achieved `0.278710`;
  - both beat deployable selected static, original dynamic, and duty
    non-PD-PPO baselines on oracle loss.
- Remaining issue:
  - exact `ep4` still has `4` warmup aborts;
  - exact `ep3` has `2` warmup aborts.
- Interpretation:
  - the scene now contains a viable operational dynamic schedule on the final
    split;
  - PPO transfer has been chasing suboptimal teacher pairs;
  - the next PPO test should imitate exact `ep4` while increasing the abort
    penalty enough to allow small deviations from the teacher.

### Dwell Is The Wrong Abort Lever
- The learned `eventpair4_dwell36` run failed the loss gate:
  - PD-PPO `0.290391`;
  - validation-selected static `0.275914`;
  - feasible static `0.276809`;
  - round-robin `0.282384`;
  - best duty non-PD-PPO `0.287921`.
- The behaviour gate was otherwise acceptable:
  - `mid=8`, no always-on/off;
  - switch rate `0.012363`;
  - abort count `1`.
- Interpretation:
  - longer dwell suppresses the useful event-conditioned response and makes the
    policy too close to a smoothed static allocation;
  - further dwell increases are not a rational search direction.

### Abort Is Mostly An Energy-Account Calibration Issue
- In the stronger dwell12 eventpair4 run, PD-PPO was loss-positive but had four
  warm-up aborts.
- Direct audit showed:
  - mean power `0.805203` while harvest was `0.65`;
  - SOC median `24.845`;
  - `50.34%` of steps had SOC `<=25`;
  - aborts occurred when SOC was near the reserve floor (`20-21`).
- Interpretation:
  - the scene is asking a dynamic schedule to operate under a long-run energy
    deficit;
  - static masks avoid aborts largely because they do not repeatedly warm
    sensors near the reserve floor;
  - the correct next lever is minimal harvest recalibration, not more PPO
    architecture changes.

### Exact Harvest Sweep Supports `h=0.74`
- Exact eventpair4 replay with energy overrides:
  - `h=0.65`: loss `0.276108`, aborts `4`;
  - `h=0.70`: loss `0.279613`, aborts `1`;
  - `h=0.72`: loss `0.281093`, aborts `0`;
  - `h=0.74`: loss `0.277467`, aborts `0`;
  - `h=0.75`: loss `0.277499`, aborts `0`;
  - `h>=0.85`: loss `0.279039`, aborts `0`.
- Interpretation:
  - `h=0.74` is the best minimal recalibration found so far;
  - it removes aborts without pushing the scenario into a loose-energy regime;
  - a learned h0.74 PPO probe is justified because the h0.65 learned policy
    already improved on the exact teacher loss while keeping the same event-pair
    mechanism.

### H0.74 Learned PPO Is Loss-Positive But Still Has One Abort
- Learned h0.74 seed41 result:
  - PD-PPO `0.283227`;
  - deployable selected static `0.299086`;
  - best deployable static `0.290746`;
  - best original dynamic `0.286486`;
  - best duty non-PD-PPO `0.284414`.
- Behaviour:
  - `mid=8`, no always-on/off;
  - switch rate `0.033791`;
  - duty range `0.123779-0.744141`;
  - warmup abort count `1`.
- Interpretation:
  - raising harvest fixed the baseline comparison against fair deployable
    families, but not the strict zero-abort gate;
  - h0.75 is the next minimal test because exact h0.75 had zero abort and almost
    identical loss to exact h0.74.

### H0.75 Does Not Fix The Learned Policy
- Learned h0.75 seed41 result:
  - PD-PPO `0.282650`;
  - deployable selected static `0.289898`;
  - best deployable static `0.286465`;
  - best original dynamic `0.282316`;
  - best duty non-PD-PPO `0.282022`.
- Behaviour:
  - `mid=8`, no always-on/off;
  - warmup abort count increased to `2`.
- Interpretation:
  - merely increasing harvest is not enough because the learned policy deviates
    from the exact event-pair schedule in ways that can still abort;
  - the next controlled change should strengthen the event-pair imitation
    signal at h0.74 rather than further relaxing the physical energy account.

### AWBC0.40 Restores The Loss Gate But Is One Abort Short
- Learned h0.74/AWBC0.40 seed41 result:
  - PD-PPO `0.278159`;
  - feasible static projected `0.278734`;
  - best original dynamic `0.282001`;
  - best duty non-PD-PPO `0.278977`;
  - best deployable static `0.286145`.
- Behaviour:
  - `mid=8`, no always-on/off;
  - switch rate `0.033913`;
  - warmup abort count `1`.
- Interpretation:
  - stronger imitation successfully moved the learned policy toward the exact
    event-pair mechanism and restored fair-baseline wins;
  - the remaining blocker is still the single low-SOC warm-up abort, so the next
    minimal test is h0.75 plus the same AWBC0.40.

### H0.75/AWBC0.40 Passes Seed41
- Learned h0.75/AWBC0.40 seed41 result:
  - PD-PPO `0.277030`;
  - feasible static projected `0.277872`;
  - best original dynamic `0.281560`;
  - best duty non-PD-PPO `0.280967`;
  - deployable selected static `0.289897`;
  - best deployable static `0.286897`.
- Behaviour:
  - `mid=8`, no always-on/off;
  - switch rate `0.033669`;
  - duty range `0.125488-0.744873`;
  - warmup abort count `0`.
- Interpretation:
  - the combined change is mechanically coherent:
    minimal harvest increase removes reserve-edge aborts, while AWBC0.40 keeps
    the learned policy near the verified event-pair schedule;
  - this setting should be treated as the first candidate worth seed
    replication.

### H0.75/AWBC0.40 Does Not Replicate
- Locked-parameter seed42--43 replication failed the stronger static-break gate.
- Combined seed41--43:
  - raw compact static: `0/3`;
  - deployable selected static: `2/3`;
  - best deployable static: `1/3`;
  - best original dynamic: `2/3`;
  - best duty dynamic: `3/3`;
  - strict zero-abort behaviour: `1/3`.
- Failure details:
  - seed42 PD-PPO `0.389781`, deployable selected static `0.387649`, aborts `4`;
  - seed43 PD-PPO `0.351861`, best deployable static `0.350600`, AoI `0.350504`,
    aborts `3`.
- Interpretation:
  - this is not a no-dynamics failure because all seeds keep `mid=8` and
    event laser duty is elevated;
  - the remaining weakness is structural: a high-duty `met+surface+laser` static
    bundle remains barely feasible and is strong in seed42.

### Exact Teacher Audit Separates Learnability From Scene Structure
- Direct event-pair replay on failed seeds:
  - seed42 best exact teacher: `h0.80/lookahead3`, loss `0.390259`, still worse
    than deployable selected static `0.387649`;
  - seed43 best exact teacher: `h0.75/lookahead6`, loss `0.348800`, beats fair
    deployable/dynamic baselines with zero abort.
- Interpretation:
  - seed42 cannot be fixed by simply training PPO harder to imitate the current
    event-pair teacher;
  - the scene must break the `met+surface+laser` boundary shortcut while keeping
    the intended event pair feasible.

### V16 Surface-Boundary Hypothesis
- Change only `surface_temp_ir` cost:
  - power `0.11 -> 0.16`;
  - startup peak `0.14 -> 0.20`.
- Resulting feasibility at B=1.15/P=1.55:
  - `met+surface+laser`: infeasible (`1.16/1.56`);
  - `met+radiometer+laser`: feasible (`1.10/1.49`);
  - `met+radiometer+surface+SPC`: feasible (`0.92/1.19`).
- Gate launched on seed42:
  `reports/v31_static_break_v16_surface_boundary_gate_seed42_20260609`.

### V16 Breaks Laser Static But Reveals FC4 Static
- Linear smoke gate passed, but the full TCN gate failed:
  - deployable static `0.523706`;
  - best eligible dynamic `0.523917`;
  - dynamic margin `-0.000404`;
  - event margin `-0.000723`.
- New TCN static shortcut:
  `radiometer_basic|surface_temp_ir|shielded_thermo_hygro|fc4_flux`.
- Interpretation:
  - v16 did what it was designed to do for the laser shortcut;
  - under the TCN oracle, mass-flux observation through FC4 plus thermal context
    remains enough for static to match dynamic;
  - the next rational gate is not PPO, but an objective/profile shift toward
    particle diameter/velocity (`micro_particle_v6`) or a further FC4/static
    boundary change.

### Micro-Particle Objective Is Still Not Enough
- v16 + `micro_particle_v6` TCN gate:
  - deployable static `0.456834`;
  - best eligible dynamic `0.456967`;
  - margin `-0.000291`;
  - event margin `-0.000413`.
- The best unrestricted dynamic beats static (`0.456058`) but violates the
  behavioural target (`always_off=3`).
- Interpretation:
  - a useful dynamic signal exists, but it is concentrated in a sparse policy;
  - to make a deployable all-sensor dynamic policy win, the particle variables
    need information that FC4/thermal static cannot infer.
- Next hypothesis:
  - increase and decorrelate event particle microstructure while keeping the
    same v16 sensor costs and `micro_particle_v6` objective.

### Structural Gate Needed A Dwell Correction
- Deployable static rows in the structural gate used the duty guard but did not
  inherit the final env-level `min_dwell_steps=12` constraint.
- Observed switch rates were `0.37-0.44/step`, which is much higher than the
  final deployment baseline and can artificially strengthen static replay.
- Fix:
  - add `--env-min-dwell-steps` to `49_v31_physical_event_oracle_lift.py`;
  - forward it through `63_v31_static_break_calibration.py`;
  - rerun the nearest TCN gate: v16 surface-boundary + `micro_particle_v6` +
    dwell12.

### Corrected Dwell12 Gate Restores Dynamic Headroom
- v16 surface-boundary + `micro_particle_v6` + dwell12:
  - deployable static `0.466835`;
  - best eligible dynamic `0.456564`;
  - dynamic margin `+0.022003`;
  - event margin `+0.021998`.
- Best eligible pair:
  - calm: `surface_temp_ir|shielded_thermo_hygro|snow_particle_counter`;
  - event: `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`.
- Interpretation:
  - under the correct deployment dwell constraint, the static shortcut is
    structurally broken on seed42;
  - learned-policy transfer is now worth testing again.

### V16 Dwell12 PPO Transfers, But H0.75 Is Energy-Inconsistent
- Learned seed42 under v16 + `micro_particle_v6` + dwell12:
  - PD-PPO `0.409595`;
  - feasible static `0.417184`;
  - validation-selected static `0.450758`;
  - deployable selected static `0.436482`;
  - best original dynamic `0.416039`;
  - best duty-constrained non-PD-PPO `0.415802`.
- Behaviour:
  - all eight sensors have intermediate duty;
  - no always-on or always-off sensors;
  - switch rate `0.037454`;
  - unique masks `26`.
- Mechanism:
  - event duty rises for met/radiometer/FC4;
  - calm duty rises for shielded/SPC;
  - the top mask is the intended event pair
    `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`.
- Hard failure:
  - `warmup_abort_count=6`;
  - aborts occur only when SOC is pinned near reserve (`~20`);
  - mean power is `0.9028`, above the configured harvest `0.75`.
- Interpretation:
  - this is the first branch where learned PD-PPO clearly beats all fair
    baseline families and avoids constant sensors on seed42;
  - the next correction should recalibrate the energy account, not alter the
    scene or hide the aborts.

### H0.92 Retraining Removes Aborts But Loses The Edge
- h0.92 retraining result on seed42:
  - PD-PPO `0.415797`;
  - feasible static `0.415090`;
  - best original dynamic `0.414240`;
  - best duty-constrained non-PD-PPO `0.411874`;
  - deployable selected static `0.434326`.
- Behaviour is valid:
  - zero abort;
  - all eight sensors at intermediate duty;
  - no always-on/off channels.
- Interpretation:
  - the harvest increase solved energy feasibility but changed the optimization
    landscape enough that PPO lost the baseline advantage;
  - the high-value hypothesis is now conservative training under h0.75 followed
    by physical h0.92 deployment replay, not h0.92 retraining.

### H0.75 Conservative Policy Does Not Transfer Cleanly To H0.92
- h0.75-trained checkpoint replayed with h0.92 harvest:
  - PD-PPO `0.415615`;
  - best original dynamic `0.415030`;
  - best duty-constrained non-PD-PPO `0.412165`;
  - feasible static `0.416799`;
  - validation-selected static `0.450758`.
- Interpretation:
  - the zero-abort replay still loses to dynamic baselines;
  - the h0.75 win was partly produced by the deterministic energy guard
    dropping expensive loads when SOC was pinned near reserve;
  - a clean accepted policy must internalize this reserve-aware behaviour
    through reward/SOC shaping rather than relying on guard drops.

### Reserve-Aware Shaping Preserves Loss But Not Energy Feasibility
- h0.75 with stronger abort penalty and SOC soft penalty:
  - PD-PPO `0.409591`;
  - best original dynamic `0.414505`;
  - best duty-constrained non-PD-PPO `0.415334`;
  - feasible static `0.415619`;
  - deployable selected static `0.434986`.
- Behaviour remains dynamically valid:
  - all eight sensors are mid-duty;
  - no always-on/off channels.
- Hard failure remains:
  - `warmup_abort_count=5`;
  - mean power remains about `0.9046`.
- Interpretation:
  - current duty/dwell objective wants a roughly `0.90` average-power operating
    point;
  - h0.75 is too low for a clean all-mid-duty deployment, so the next evidence
    check should find the minimal harvest level where the same checkpoint has
    zero abort while still beating baselines.

### Harvest Boundary Is Narrow
- Saved-policy harvest sweep with the h0.75-SOC checkpoint:
  - h0.80 wins static, original dynamic, and duty-constrained dynamic baselines,
    but still has `1` abort;
  - h0.84 and above remove aborts and keep wins over original dynamic/static,
    but lose to duty-constrained round-robin.
- Interpretation:
  - increasing harvest removes the energy hard failure, but also enables the
    constrained round-robin baseline to operate in its strongest regime;
  - the only remaining plausible h0.75-SOC replay window is h0.81--h0.83.

### Fine Harvest Window Does Not Fully Pass
- h0.81--h0.83 all remove aborts and keep the static shortcut broken.
- None wins all fair dynamic families:
  - h0.81 loses to duty-constrained round-robin;
  - h0.82 loses to AoI;
  - h0.83 loses to duty-constrained round-robin.
- Interpretation:
  - the remaining competitor is not static anymore;
  - it is the ability of heuristic policies to rotate more aggressively even
    under dwell12, so the next fair operational test is a stricter common dwell
    constraint.

### Stricter Common Dwell Hurts The Saved Policy
- h0.82 replay with env-level dwell18/24/36:
  - all variants remove aborts;
  - all variants lose to static and dynamic baselines.
- Interpretation:
  - applying a stricter dwell at evaluation time to a dwell12-trained policy is
    too disruptive;
  - if h0.82 is to work, it needs direct retraining under h0.82 rather than
    replay-only adjustments.

### H0.82 Direct Retraining Passes Seed42
- Direct h0.82 reserve-aware training:
  - PD-PPO `0.409735`;
  - best original dynamic `0.412762`;
  - best duty-constrained non-PD-PPO `0.414889`;
  - feasible static `0.416452`;
  - deployable selected static `0.432842`.
- Behaviour:
  - zero abort;
  - all eight sensors mid-duty;
  - no always-on/off channels;
  - switch rate `0.038309`.
- Interpretation:
  - the correct solution was not h0.92 relaxation or replay-only tuning;
  - direct training at the nearest clean energy boundary preserves the learned
    event/calm scheduling advantage while satisfying deployment behaviour.

### H0.82 Does Not Replicate Across Seeds
- Locked h0.82 settings on seeds 41 and 43 completed and were audited from raw
  `v2_custom_ppo_metrics.csv` plus rollout NPZ files.
- Combined seeds 41/42/43:
  - static shortcut: PD-PPO wins `1/3`, mean delta
    `best_static - PD-PPO = -0.018320`;
  - deployable static: wins `1/3`, mean delta `-0.005250`;
  - original dynamic: wins `2/3`, mean delta `-0.005902`;
  - duty-constrained dynamic: wins `2/3`, mean delta `+0.001509`;
  - full-open reference: wins `3/3`, mean delta `+0.020939`.
- Behaviour is not the problem:
  - all three seeds have zero aborts;
  - all three keep `mid_duty_sensor_count=8`;
  - no always-on/off sensors;
  - switch rate stays around `0.036--0.039`.
- Failure mechanism:
  - seed41 is dominated by compact static masks and round-robin;
  - seed43 is almost tied with AoI but still loses to validation-selected
    static;
  - event-window loss is the weak point in seed41/43, while static masks remain
    able to exploit simple event microstructure in those final-test segments.
- Conclusion:
  - h0.82 fixed the energy/duty behaviour but did not eliminate the static
    shortcut;
  - adding more seeds under the same settings would only measure the failure
    rate, not solve it.

### Next Structural Hypothesis
- Seed41/43 failures are not caused by deployment invalidity:
  PD-PPO is balanced and zero-abort, yet compact static masks remain strong.
- The likely remaining issue is insufficient target-level complementarity:
  a static mask can still cover enough of either particle or flux structure in
  the final-test segment.
- The next screen therefore tests stronger joint flux+particle target pressure
  across multiple seeds before any additional PPO training.

### Seed41 Micro-Flux Gate Is Positive
- In the v16 multi-seed structural screen, the first completed row passed:
  - seed41, `micro_flux_v6`;
  - dynamic loss `0.581084` vs deployable static `0.587162`;
  - dynamic margin `+1.04%`, event margin `+1.27%`;
  - dynamic behaviour passes with `mid=7`, `always_on=0`, `always_off=1`.
- This suggests that the h0.82 replication failure was not solely caused by
  v16 sensor costs. The unstable component may be the `micro_particle_v6`
  objective, which lets compact particle/static masks remain too strong in
  seed41/43.

### AWBC Teacher Was Misaligned With The Structural Gate
- In seed41, `micro_particle_v6` also passes the structural gate:
  dynamic loss `0.515194` vs deployable static `0.519278`.
- The best eligible dynamic is `auto_non14_event15`, whose event mask is
  `met_station_core|radiometer_basic|laser_disdrometer`.
- The failed h0.82 PPO branch used an event teacher with FC4 instead of laser:
  `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`.
- This is a concrete training-target mismatch: the scenario has a dynamic
  solution, but the imitation prior points the actor toward a different event
  mask.

### Seed41 Dynamic Headroom Is Consistent Across V6 Profiles
- In the multi-seed structural screen, the first three seed41 rows all pass:
  `micro_flux_v6`, `flux_micro_v6`, and `micro_particle_v6`.
- Margins are modest but consistent (`+0.79%` to `+1.04%`), and event margins
  are positive (`+0.97%` to `+1.27%`).
- All point to a laser event mask rather than the FC4 event teacher used by the
  failed h0.82 PPO branch.

### Teacher Alignment Alone Is Not Sufficient
- The seed41 laser-teacher PPO probe failed:
  - `custom_ppo=0.347668`;
  - old h0.82 seed41 was `0.331129`;
  - best static remains near `0.295--0.300`;
  - best duty baseline is `0.321968`.
- The policy did imitate the intended masks, so this is not a launch/config
  failure:
  - top mask `surface|ultrasonic|shielded|SPC` at `52.1%`;
  - second mask `met|radiometer|laser` at `18.9%`.
- Interpretation:
  - the structural-gate margin is real but small;
  - forcing the actor toward the gate's best pair does not optimize the
    long-horizon final-test reward enough and introduces one abort;
  - further AWBC strengthening is unlikely to solve the main claim.

### Dual Flux+Particle Profile Helps Only Slightly In Seed41
- `dual_flux_particle_v7` gives the best seed41 structural margin so far:
  `+1.063%`.
- This is only a tiny improvement over `micro_flux_v6` (`+1.035%`), so the
  target-profile lever alone is unlikely to create a robust PPO margin unless
  seed42/43 show a much stronger pattern.

### Split-Protocol Final-Test Windows Were Misaligned With The Structural Gate
- The structural screen explicitly uses `event_transport_rich` windows.
- The PPO split-protocol runner previously selected final-test starts uniformly
  and recorded the selection as
  `uniform_random_non_overlapping_without_event_filtering`.
- This matters because the current scene is designed around event/calm
  complementarity. Uniform final-test windows can dilute the event-side value
  and reward compact static masks that perform well in calmer windows.
- The next probe therefore keeps the original h0.82/FC4-teacher PPO settings
  but changes only final-test selection to `event_transport_rich`.
- If this probe passes seed41, the correct route is to replicate seeds under an
  explicitly event-window/storm-window claim. If it still fails, the structural
  margin is not transferring into PPO and scenario/objective separation must be
  strengthened before more retraining.

### Seed41 Structural Headroom Is Robust But Narrow
- All six v16 profiles pass on seed41 against deployable static under dwell12.
- The best profile is `dual_flux_particle_v7`, but its margin is only
  `+1.063%`.
- The weakest passing profiles remain close to `+0.79%`.
- This confirms that the scene is not structurally impossible for dynamic
  scheduling, but also explains why PPO transfer is unstable: the policy has
  little room to beat strong static masks unless training and evaluation are
  tightly aligned to event-rich windows.

### Event-Rich Evaluation Reveals An Event/Calm Tradeoff, Not A Full Fix
- The seed41 event-rich PPO probe almost tied deployable selected static:
  `0.352897` vs `0.352868`.
- The learned policy is operationally clean: zero aborts, all sensors mid-duty,
  and no always-on/off collapse.
- The policy does learn useful event behaviour:
  event loss improves over deployable static by `0.013080`.
- The loss is paid back in calm windows:
  non-event loss is worse by `0.006297`.
- The weighted break-even event rate is about `0.324973`, while the selected
  final windows have event rate `0.323486`.
- Conclusion:
  final-window alignment is not the main remaining blocker; the current target
  profile still leaves too much calm-window static value. The next probe should
  use stronger flux+particle weighting rather than more AWBC imitation.

### Dual Flux+Particle Converts Seed41 From Near-Tie To Deployable Win
- With identical h0.82/dwell12/event-rich settings, changing the target profile
  from `micro_particle_v6` to `dual_flux_particle_v7` improved seed41 PD-PPO
  from a deployable-static near loss to a clear deployable-static win:
  `0.341429` vs `0.346158`.
- The policy also beats the best duty-constrained non-PD-PPO baseline:
  `0.341429` vs `0.342900`.
- Behaviour is clean: zero aborts, all eight sensors mid-duty, no always-on/off.
- The unresolved rows are not the deployable baselines:
  compact static and original round-robin still win, but they rely on
  always-on/off shortcut behaviour.
- This makes `dual_flux_particle_v7` the current best replication candidate,
  pending seed42/43 structural confirmation.

### Dual Flux+Particle Replicates On Seed42 For Static/Deployable Gates
- Seed42 learned PPO under `dual_flux_particle_v7` beats:
  - best static (`0.401397` vs `0.402101`);
  - selected static (`0.401397` vs `0.429319`);
  - deployable selected static (`0.401397` vs `0.421030`);
  - best deployable static (`0.401397` vs `0.409734`);
  - best duty-constrained non-PD-PPO (`0.401397` vs `0.405430`).
- It remains lower-quality than original AoI (`0.394795`), so unconstrained
  dynamic baselines are not solved by this branch.
- Behaviour remains clean: zero aborts, all sensors mid-duty, no always-on/off.
- This is the first branch in the current exploration that shows learned PPO
  wins over deployable static in two consecutive seeds while keeping strict
  deployment behaviour.

### Dual Flux+Particle Meets The Three-Seed Deployable-Static Target
- Seeds 41--43 under the fixed dual-profile/event-rich/h0.82 settings produce:
  - deployable selected static wins `3/3`;
  - best deployable static wins `3/3`;
  - valid deployment behaviour `3/3`;
  - zero aborts `3/3`;
  - no PD-PPO always-on/off collapse `3/3`.
- This directly addresses the user's requirement that PD-PPO must fully beat
  deployable static.
- The branch does not yet support broader dominance:
  - best original dynamic wins remain `0/3`;
  - best duty-constrained non-PD-PPO is only `2/3`.
- Interpretation for the paper should therefore be:
  PD-PPO beats deployment-valid static shortcuts under the calibrated
  event/flux+particle regime, while unconstrained or highly reactive dynamic
  heuristics remain diagnostic comparators rather than the main operational
  baseline.

### Structural Screen Continues To Support Dynamic Headroom
- Seed42 is now fully screened across six v16 flux/particle profiles and all
  six pass the deployable-static structural gate.
- The strongest seed42 margins are large by the standards of this branch:
  `particle_heavy_flux_v7` `+4.70%`, `micro_particle_v6` `+4.62%`, and
  `dual_flux_particle_v7` `+3.77%`.
- Seed43 `micro_flux_v6` also passes with dynamic margin `+1.60%` and event
  margin `+1.95%`.
- This weakens the hypothesis that the new positive learned result is a pure
  seed41/42 accident. The remaining uncertainty is learned-policy transfer,
  not existence of a dynamic oracle solution.

### Dual Flux+Particle Does Not Survive 5-Seed Expansion
- Seeds 44--45 both preserve the intended operational behaviour:
  all eight sensors are mid-duty, no always-on/off collapse, and zero warmup
  aborts.
- They nevertheless lose every baseline family, including deployable selected
  static.
- Combined 41--45 evidence is therefore not a stable positive claim:
  deployable selected static `3/5` with mean delta `-0.007197`; best original
  dynamic `0/5`; best duty non-PD-PPO `2/5`.
- Mechanism:
  - seed44 shows useful event scheduling but loses much more calm-window loss;
  - seed45 deployable static uses a duty-valid laser shortcut
    (`radiometer|shielded|laser` for `61.8%` of steps);
  - PD-PPO keeps laser at the low duty boundary and follows the FC4 event-pair
    teacher, which is not robust across seed-specific event microstructure.
- Correction direction:
  a stronger branch must make laser/static shortcuts non-dominant under the
  same deployable duty guard, or train the actor to choose between laser and FC4
  event modes from state rather than fixing the FC4 teacher.

### Fixed Event-Pair AWBC Is A Likely Transfer Bottleneck
- Seeds 44--45 used `awbc_teacher_mode=event_pair` with:
  - calm action:
    `surface_temp_ir|shielded_thermo_hygro|snow_particle_counter`;
  - event action:
    `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`.
- Training logs show `awbc_label_rate=1.0` and only `1--2` unique teacher
  actions near the end of training.
- That explains why the learned schedules are dynamic but narrow:
  - seed44 deployable static wins calm windows with a radiometer/ultrasonic/FC4
    bundle that is not the calm teacher;
  - seed45 deployable static wins by using a laser-heavy duty-valid mask, while
    the event teacher uses FC4 instead of laser.
- If the targeted structural screen shows dynamic headroom in seeds 44--45,
  the next PPO probe should prefer `oracle_greedy` AWBC or a seed-adaptive
  teacher over another fixed event-pair run.

### Seed44 Structural Headroom Exists Despite PPO Failure
- Targeted seed44 structural screening passes in all four tested profiles under
  the same v16/dwell12/h0.82 setting where fixed-teacher PPO failed.
- The best profile is `particle_heavy_flux_v7`:
  `0.588671` best dynamic vs `0.595923` deployable static.
- `dual_flux_particle_v7` also passes:
  `0.628894` best dynamic vs `0.634652` deployable static.
- `micro_particle_v6` and `micro_flux_v6` also pass, so the effect is not a
  one-profile artifact.
- All profiles choose the same dynamic family,
  `dynamic:auto_non19_event9_lead0`, rather than the fixed FC4 event-pair
  teacher. This strengthens the teacher-transfer hypothesis.
- For the strongest seed44 completed row (`particle_heavy_flux_v7`), the
  decoded masks are:
  - non-event action 19:
    `met_station_core|surface_temp_ir|fc4_flux`;
  - event action 9:
    `met_station_core|radiometer_basic|shielded_thermo_hygro|snow_particle_counter`.
- The failed fixed teacher instead used:
  - calm:
    `surface_temp_ir|shielded_thermo_hygro|snow_particle_counter`;
  - event:
    `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`.
- This is a concrete label mismatch, not just a weak RL signal.

### Seed45 Failure Is A Profile Mismatch
- In the targeted seed45 structural screen, `dual_flux_particle_v7` fails the
  dynamic-headroom gate:
  `0.800709` best dynamic vs `0.802117` deployable static, margin `+0.001755`.
- `particle_heavy_flux_v7` passes in the same seed:
  `0.726187` best dynamic vs `0.732773` deployable static, margin `+0.008988`.
- `micro_particle_v6` also passes, but slightly weaker:
  `0.729728` best dynamic vs `0.735927` deployable static, margin `+0.008423`.
- `micro_flux_v6` fails outright with margin `-0.000822`.
- This explains why seed45 looked special in the learned dual-profile branch:
  the selected profile did not create enough dynamic value in that seed, while
  a particle-heavy profile still does.
- Cross-seed implication:
  `particle_heavy_flux_v7` is now the best candidate profile because it is the
  strongest completed profile in seeds 42, 44, and 45, while still passing seed
  41.

### Particle-Heavy Is The Current Structural Leader
- Completed seed43 structural screening also selects `particle_heavy_flux_v7`
  as the strongest profile:
  `0.704138` best dynamic vs `0.718386` deployable static, margin `+0.019833`.
- Cross-seed structural picture:
  - seed41: particle-heavy passes, though dual is slightly stronger;
  - seed42: particle-heavy is strongest;
  - seed43: particle-heavy is strongest;
  - seed44: particle-heavy is strongest;
  - seed45: particle-heavy is the only completed passing row so far, while
    dual fails.
- This justifies shifting the learned branch from dual-profile to
  particle-heavy rather than further tuning dual-profile PPO.

### Oracle-Greedy Fixes Part Of The Learned Transfer Gap
- Seed44 dual-profile oracle-greedy PPO improves loss from `0.411077` under
  fixed event-pair AWBC to `0.372997`.
- The improvement is mechanism-correct:
  event loss becomes lower than deployable static (`0.555780` vs `0.600764`).
- The remaining failure is calm loss (`0.259391` vs `0.224437`), so teacher
  adaptivity alone does not rescue the dual profile.
- Since particle-heavy has larger structural headroom in seed44 and seed45, the
  next learned-policy evidence should come from particle-heavy plus
  oracle-greedy, not additional dual-profile retries.

### Fork Independence Rule
- The current fork is an independent PD-PPO / RL sensor scheduling route.
- v1 is no longer an active implementation or evidence source for this
  workstream.
- v1 records can be read as archived diagnostic material, especially because the
  long v1 exploration has not produced a stable successful result.
- Do not use v1 code, v1 method claims, or v1 numerical rows as current PD-PPO
  paper evidence.
- Use v1 mainly to avoid repeating failed directions and to preserve historical
  context.
- The current route is documented in:
  `.planning/2026-06-07-pd-ppo-static-break-recalibration/pdppo_independent_particle_heavy_route.md`.

### Active Route After Independence Clarification
- Main profile:
  `particle_heavy_flux_v7`.
- Main scene:
  `windblown_sensors_physical_event_v16_surface_boundary.yaml`.
- Main learned probe:
  `pdppo_v16_particle_heavy_seed45_h082_oraclegreedy_20260609`.
- The route is not a v1 migration. It is a PD-PPO scenario/profile/teacher
  correction based on structural screens and learned-policy transfer audits
  inside `rl_sensor_scheduling_framework`.

### V16 Particle-Heavy Learned Probe Is Not Sufficient
- Seed45 particle-heavy oracle-greedy PD-PPO is behaviour-valid:
  `mid=8`, `always_on=0`, `always_off=0`, `warmup_abort=0`,
  `switches_per_step=0.038187`.
- It beats the deployable selected-static replay:
  `0.432414` vs `0.436687`.
- It does not pass the main gate:
  - loses best deployable static / best duty non-PD-PPO:
    `0.432414` vs `0.431815`;
  - loses best original dynamic:
    `0.432414` vs `0.418746`;
  - loses raw feasible static:
    `0.432414` vs `0.391799`.
- Event/calm audit:
  - PD-PPO event `0.708512`, non-event `0.277366`;
  - duty feasible static event `0.696706`, non-event `0.283061`;
  - round-robin event `0.722932`, non-event `0.247924`;
  - raw feasible static event `0.652193`, non-event `0.245569`.
- The learned policy keeps `laser_disdrometer` and `fc4_flux` near the duty
  lower bound, so the intended episodic high-cost-channel mechanism is not yet
  learned.
- Decision:
  do not expand v16 particle-heavy PPO to seeds 41--45. Test structural
  generator change under v17 first.

### V17 Gate Must Use Current Constraints
- The old v17 micro-particle gate is stale for the current route because it used
  `micro_particle_v6`, harvest `0.75`, and did not explicitly enforce env dwell
  12 in the gate runner.
- Correct current-route v17 gate settings are:
  `particle_heavy_flux_v7`, harvest `0.82`, env dwell `12`, seed45 first.
- If this gate does not create dynamic headroom, additional PPO tuning is likely
  low-value until the scenario/profile changes more substantially.

### Corrected V17 Gate Result
- Corrected v17 particle-heavy seed45 at B=1.15 does not yet satisfy the
  final structural gate.
- The useful signal is split:
  - behaviour-valid dynamic schedule has a small event-side gain
    (`+0.000439`) but loses overall by `-0.001091`;
  - the strongest unrestricted dynamic schedule wins loss-wise
    (`0.648781` vs deployable static `0.663613`) but uses a degenerate duty
    pattern (`mid=3`, `always_on=2`, `always_off=3`).
- This is not a PPO-tuning problem yet. The structural design still trades off
  dynamic advantage against nondegenerate deployment behaviour.
- Next diagnostic should be a nearby budget scan, because the strict failure is
  small and may depend on whether the budget allows a behaviour-valid dynamic
  alternation without preserving the same static shortcut.

### V17 Budget Scan Creates A Valid Learned-Policy Target
- The targeted v17 particle-heavy budget scan on seed45 changed the structural
  decision from "do not train" to "train one probe".
- All tested budgets passed the structural gate under env dwell 12 and h0.82:
  - `B=1.05`: margin `+0.007317`, event margin `+0.006063`;
  - `B=1.10`: margin `+0.022911`, event margin `+0.028598`;
  - `B=1.20`: margin `+0.013654`, event margin `+0.014208`.
- `B=1.10` is the best next learned-policy target because it has the largest
  headroom while preserving an acceptable dynamic duty profile:
  `mid=6`, `always_on=1`, `always_off=1`, `switches_per_step=0.060059`.
- The best acceptable `B=1.10` dynamic action alternates between:
  - non-event:
    `met_station_core|surface_temp_ir|shielded_thermo_hygro|snow_particle_counter`;
  - event:
    `met_station_core|radiometer_basic|ultrasonic_anemometer_hd|fc4_flux`.
- This structure is aligned with the intended particle-heavy route: calm
  windows rely on surface/particle context, while event windows switch toward
  radiometer/ultrasonic/FC4 without requiring laser or multiple permanently
  active channels.
- Active test:
  `pdppo_v17_particle_heavy_b1p10_seed45_h082_oraclegreedy_20260610`.

### V17 B=1.10 Learned Probe Is A Partial, Not Expandable, Pass
- The B=1.10 learned probe satisfies the user's static-shortcut condition in
  the deployment-valid comparison:
  PD-PPO `0.456376` beats deployable selected static `0.468638` and best
  deployable static `0.463888`.
- It is not a full evidence branch:
  it loses selected / best static `0.415860`, AoI `0.441799`, and
  duty-constrained round-robin `0.441571`.
- The failure is not a behaviour collapse:
  `mid=8`, no always-on/off sensors, zero aborts, and switch rate `0.039164`.
- Mechanism:
  PD-PPO improves event-window loss relative to AoI/round-robin but gives that
  back in calm windows.
- Sensor-duty diagnosis:
  - `met_station_core`, `radiometer_basic`, and `snow_particle_counter` are
    kept near the upper duty bound;
  - `laser_disdrometer` and `fc4_flux` remain near the lower duty bound;
  - FC4 and ultrasonic do not increase during event windows.
- Event-pair replay shows that the structural-gate action 11/20 does not
  transfer to the split-run oracle/evaluation protocol (`0.492458`), and the
  best current-oracle event-pair replay is only `0.453114`.
- Decision:
  do not replicate B=1.10 or train fixed event-pair AWBC from these masks.
  Test the structurally passing budget bracket B=1.05/B=1.20 instead.

### V17 Budget Bracket Does Not Fix The Dynamic-Baseline Gap
- B=1.05 and B=1.20 both completed on seed45.
- B=1.05:
  PD-PPO `0.440043` beats deployable selected static `0.449926`, but loses
  best deployable static `0.434833`, round-robin `0.421760`, and
  duty-constrained round-robin `0.429548`.
- B=1.20:
  PD-PPO `0.446923` beats deployable selected static `0.452960`, but loses
  best deployable static `0.439660`, round-robin `0.429338`, and best duty
  non-PD-PPO `0.439660`.
- Across B=1.05/1.10/1.20:
  no budget point beats original dynamic or duty-constrained dynamic baselines.
- Since B=1.10 is the only point that beats best deployable static, the next
  correction should keep B=1.10 and change training distribution/reward
  emphasis rather than continue budget search.
- Active hypothesis:
  event-start sampling `0.90` plus event reward multiplier `3.0` overweights
  event windows; PD-PPO wins event loss but loses too much non-event loss.
- Active test:
  B=1.10 balanced training with event-start probability `0.65` and event
  reward multiplier `1.5`.

### V17 Balanced Training Does Not Fix The Non-Event Gap
- B=1.10 balanced training completed on seed45.
- PD-PPO remained behaviour-valid:
  `mid=8`, no always-on/off sensors, zero aborts, and switch rate `0.041911`.
- Loss did not improve:
  `custom_ppo=0.458406`, worse than the previous event-heavy B=1.10 probe
  (`0.456376`).
- It still only beats the deployable-static family, not the main dynamic
  baselines:
  - best deployable static `0.463114`;
  - best original dynamic, AoI `0.441903`;
  - best duty-constrained dynamic, duty-constrained round-robin `0.441375`.
- Mechanism:
  event loss is better than AoI and duty round-robin, but non-event loss is
  much worse (`0.320755` vs `0.286210` / `0.288367`).
- Therefore the immediate issue is not excessive event sampling alone.
  The next high-ROI probe should recover calm-window quality by adding a weak
  candidate-static prior while retaining hard duty constraints.

### Weak Candidate Prior Does Not Help Under Event-Heavy Training
- The B=1.10 event-heavy weak-prior probe completed with
  `custom_ppo=0.459842`, worse than both the no-prior event-heavy probe
  (`0.456376`) and the no-prior balanced probe (`0.458406`).
- Behaviour remained valid (`mid=8`, no always-on/off, zero aborts), so the
  failure is not a deployment-constraint violation.
- The branch still only beats deployable static (`0.461550`) and loses the
  dynamic baselines (`round_robin=0.439709`,
  `duty_constrained_round_robin=0.439123`).
- Mechanism:
  event loss is slightly better than dynamic baselines, but non-event loss is
  still much worse (`0.323429` vs `0.288497` / `0.286946`).
- Sensor duty became more static-like:
  `met_station_core` and `radiometer_basic` both stayed near `0.744`, while
  event/non-event duty differences were mostly near zero.
- Therefore, weak candidate prior is not useful when combined with the original
  event-heavy training emphasis. The only remaining value of this idea is the
  paired balanced-prior run.

### Balanced Candidate Prior Is Directionally Useful But Insufficient
- The paired balanced weak-prior probe is the best current v17 B=1.10 branch:
  `custom_ppo=0.450952`.
- It remains behaviour-valid:
  `mid=8`, no always-on/off sensors, zero aborts, and switch rate `0.040904`.
- It beats deployable static by a clear margin:
  `0.450952` vs `0.462892`.
- It still loses the relevant dynamic baselines:
  AoI `0.442024` and duty-constrained round-robin `0.441410`.
- The improvement is mechanistically interpretable:
  non-event loss improves from about `0.321` in the no-prior probes to
  `0.303890`, while event loss remains close to dynamic baselines.
- Sensor duty also becomes more reasonable:
  `radiometer_basic` rises during events, and `surface_temp_ir` /
  `snow_particle_counter` rise outside events.
- This gives one more justified probe: a stronger balanced prior
  (`candidate_prior_scale=1.0`, `prior_kl_coef=0.1`) under the same hard duty
  guard. If that does not close the dynamic-baseline gap, the next correction
  should be a scenario/objective change rather than more PPO/prior tuning.

### Stronger Candidate Prior Confirms A Scene/Objective Tradeoff
- The stronger balanced-prior probe produced `custom_ppo=0.455396`, worse than
  the weaker balanced-prior run (`0.450952`).
- It remains behaviour-valid (`mid=8`, no always-on/off, zero aborts), so the
  problem is objective allocation rather than deployment validity.
- Stronger prior improves event-window loss (`0.693108`) but worsens non-event
  loss (`0.321905`).
- Compared with duty-constrained round-robin:
  - event window: PD-PPO is better (`0.693108` vs `0.713484`);
  - non-event window: PD-PPO is worse (`0.321905` vs `0.288966`).
- This confirms that current v17 B=1.10 is not primarily a PPO/prior-strength
  issue. The algorithm can create event advantage, but current final-test
  event density and loss weighting do not make that advantage dominate.
- Next correction must be scenario/objective level:
  quantify event-density or event-weight thresholds from existing rollouts,
  then decide whether a v18 scenario should raise event dominance or alter
  calm-window predictability.

### V18 Event-Dominant Gate Creates Real Structural Headroom
- The v17 event-density analysis showed that current learned event advantage
  only dominates dynamic baselines when event windows occupy roughly
  `0.58--0.63` of the weighted evaluation.
- V18 raises event dominance directly instead of continuing PPO/prior tuning:
  event coverage `0.55`, event duration `12--36`, min gap `2`, and
  final-test selection by `event_fraction=0.65`.
- The structural gate passed cleanly:
  deployable static `0.373700` vs behaviour-valid dynamic `0.353753`.
- The dynamic advantage is large enough to justify training:
  `+0.053378` overall margin and `+0.055077` event-window margin.
- Behaviour is acceptable for a diagnostic target:
  `mid=7`, no always-on sensors, one always-off sensor, and
  `switches_per_step=0.039307`.
- Interpretation:
  v18 is not just another parameter retry. It changes the scenario/objective
  distribution in the direction implied by the v17 threshold analysis.

### Event-Fraction Evaluation Needs Non-Greedy Start Selection
- The first v18 PPO launch failed before learning because the initial
  `event_fraction_starts` implementation was greedy.
- Greedy selection can pick an early high-event start that blocks enough later
  non-overlapping windows, even when a feasible set exists.
- The fix is a bounded backtracking selector that first tries the desired
  number of event-rich windows, then relaxes that count only if geometry
  forces it.
- The v18 PPO runner now uses `512 x 8` evaluation windows, matching the
  structural gate's total evaluation coverage while making non-overlap more
  stable in the short final-test partition.

### V18 Learned Probe Breaks Static But Not Dynamic Yet
- The first v18 learned probe is the strongest static-break result so far:
  PD-PPO beats full-open, best static, selected static, deployable selected
  static, and best deployable static in the same seed.
- Behaviour is clean:
  `mid=8`, no always-on/off sensors, zero aborts, and
  `switches_per_step=0.038462`.
- The remaining failure is narrow and specific:
  PD-PPO loses AoI by `0.000401` and duty-constrained round-robin by
  `0.002083`.
- Event/calm audit reverses the old v17 failure mode:
  PD-PPO now has the best non-event loss (`0.260588`), but event loss
  (`0.542475`) is worse than AoI (`0.533319`) and duty round-robin
  (`0.533853`).
- Therefore the next probe should not change the scene or static constraints.
  It should modestly increase event emphasis from `0.65/1.5` to `0.75/2.0`.

### V18 Medium Event Emphasis Worsens The Learned Policy
- Increasing event-start probability and event reward from `0.65/1.5` to
  `0.75/2.0` worsened PD-PPO from `0.411854` to `0.418941`.
- The degradation is mainly event-window loss:
  `0.542475 -> 0.554086`; non-event loss also worsens slightly
  `0.260588 -> 0.262435`.
- The policy remains behaviour-valid (`mid=8`, no always-on/off, zero aborts),
  so this is not a duty-collapse failure.
- Sensor-duty audit shows the event-emphasis run still does not move the
  important event sensors enough: FC4 and laser remain near the low duty bound,
  while radiometer and SPC remain more active in non-event than event windows.
- Conclusion:
  do not keep increasing event sampling or event reward on v18.

### V18 Fixed Event-Pair Replay Is Not A Sufficient Teacher
- Saved-run replay of the structural gate pair and FC4/ultrasonic event
  alternatives did not beat the best learned branch.
- Best replay was `calm14_event20_l0=0.413351`, valid behaviour, but still
  worse than balanced40k PD-PPO (`0.411854`) and dynamic baselines.
- The direct structural pair `struct14_15_l0` transfers poorly to the split
  protocol (`0.422221`).
- Conclusion:
  do not spend a PPO run on fixed event-pair AWBC for v18. If balanced80k
  fails, the remaining problem is likely scenario/objective transfer rather
  than a missing imitation target.

### V18 Balanced80k Disproves The Simple Optimization-Limited Explanation
- Increasing balanced v18 training from `40000` to `80000` timesteps worsened
  PD-PPO from `0.411854` to `0.429545`.
- The run remained behaviour-valid (`mid=8`, no always-on/off sensors, zero
  aborts), so the failure is not deployment collapse.
- The degradation is broad:
  event loss worsened from `0.542475` to `0.565269`, and non-event loss
  worsened from `0.260588` to `0.272369`.
- Balanced80k no longer beats deployable selected static
  (`0.429545` vs `0.425651`) and loses the dynamic baselines by much larger
  margins.
- Conclusion:
  the remaining v18 gap is not solved by more PPO updates under the same
  objective. Balanced40k should be treated as the best learned v18 point; more
  same-setting optimization is not justified.

### V18 Balanced40k Is A Qualified Operational Positive, Not Full Dynamic Dominance
- The switch-limited audit shows balanced40k PD-PPO (`0.411854`) beats all
  dwell24/dwell36 operational dynamic variants, including
  `custom_ppo_dwell24=0.417325`, `custom_ppo_dwell36=0.419495`,
  `duty_dwell24_aoi=0.421253`, and `duty_dwell36_round_robin=0.424932`.
- It still narrowly loses the original high-frequency dynamic rows:
  AoI `0.411454` and duty-constrained round-robin `0.409771`.
- Therefore v18 establishes:
  static-family breakage plus a win against switch-limited operational
  dynamics under clean duty behaviour.
- It does not establish:
  dominance over the strongest unconstrained/high-frequency dynamic
  heuristics. Any final claim must keep those two baseline classes separate.

### V19 Should Target The SPC/Laser Boundary, Not More Event Weight
- V18 balanced40k's learned duty audit shows the policy keeps
  `snow_particle_counter` near the high duty bound and `laser_disdrometer`
  near the low duty bound.
- The v18 structural gate's best eligible dynamic, however, switches toward
  event-side laser and beats deployable static with a `0.053378` margin.
- Event-emphasis training worsened the learned policy, so the next lever
  should be the scenario feasibility boundary, not more event reward.
- V19 therefore changes only the `snow_particle_counter` cost:
  `0.52/0.68 -> 0.62/0.83`.
- This keeps both intended bundles feasible and tight under B=`1.10`:
  calm SPC bundle `1.09/1.45`, event laser bundle `1.10/1.49`.
- The gate result should determine whether the boundary change creates enough
  structural headroom to justify a learned PPO probe.

### V19 Boundary Change Does Not Improve Structural Headroom
- V19 still passes the structural gate, but its headroom is slightly worse than
  v18:
  overall margin `0.051004` vs v18 `0.053378`, and event margin `0.051576`
  vs v18 `0.055077`.
- Best dynamic remains the same family:
  `dynamic:auto_non14_event15_lead0`.
- The best deployable static also remains the SPC-heavy source family:
  `met_station_core|radiometer_basic|snow_particle_counter`.
- Conclusion:
  raising SPC cost alone does not solve the transfer problem. Do not spend a
  PPO run on v19.

### Candidate Prior May Be Suppressing Event-Laser Exploration
- The v18 balanced candidate-prior table ranks SPC/FC4 static masks highest;
  the top 12 prior rows contain no `laser_disdrometer`.
- The learned v18 balanced policy mirrors that bias:
  high duty on `met_station_core`, `radiometer_basic`, and
  `snow_particle_counter`, with `laser_disdrometer` near the low duty bound.
- The runner already enabled the event-gated actor, so a disabled-gating
  explanation is ruled out.
- One controlled no-candidate-prior ablation is justified before abandoning
  v18 as strictly incapable of original-dynamic dominance.

### No-Prior Ablation Falsifies The Candidate-Prior Explanation
- Disabling the candidate prior worsened PD-PPO from `0.411854` to `0.415339`.
- The event loss worsened from `0.542475` to `0.551928`; non-event loss
  improved from `0.260588` to `0.257161`, but not enough to offset event
  degradation.
- The policy still did not raise event-side laser:
  `laser_disdrometer` event duty only changed from `0.131938` to `0.134668`.
- The policy remains SPC-heavy:
  `snow_particle_counter` event/non-event duty `0.718380` / `0.752898`.
- Conclusion:
  the weak candidate prior is not the main suppressor. The remaining
  algorithmic suspect is strong oracle-greedy AWBC (`awbc_coef=0.40`) because
  no-prior training still reports `awbc_label_rate=1.000` and retains the same
  SPC-heavy allocation.

### Low-AWBC No-Prior Exhausts V18 Same-Scene Tuning
- Reducing `awbc_coef` from `0.40` to `0.05` with the candidate prior disabled
  worsened PD-PPO to `0.436716`.
- It fails best static, deployable selected static, best deployable static,
  original dynamic, and duty-constrained dynamic rows.
- The mechanism is not the desired event-laser transfer:
  event-side FC4 rises, but event-side laser falls and non-event loss collapses
  to `0.308671`.
- V18 learned branches now tested and rejected beyond balanced40k:
  medium event emphasis, fixed event-pair replay, balanced80k, no-prior, and
  low-AWBC/no-prior.
- Conclusion:
  stop same-scene v18 algorithm tuning. The best honest v18 result remains
  balanced40k: static-family break plus switch-limited operational dynamic win,
  but not strict original-dynamic dominance.

### V20 Profile Scan Is The Next Structural Test
- Same-scene v18 PPO tuning failed through the plausible controlled levers:
  event emphasis, fixed event-pair replay, longer training, no-prior, and
  low-AWBC/no-prior.
- V19's sensor-cost boundary did not improve structural headroom, so the next
  lower-cost structural axis is the existing target-profile family rather than
  another cost tweak or PPO recipe.
- The v20 gate holds the v18 event-dominant geometry fixed and scans:
  `particle_heavy_flux_v7`, `event_flux_particle_v7`, and
  `dual_flux_particle_v7`.
- Acceptance logic:
  launch learned PPO only if a profile improves behaviour-valid deployable
  static headroom and event-side margin beyond v18's completed structural gate.
  If no profile improves those margins, the honest branch remains v18
  balanced40k as an operational/static-family positive, not a strict
  original-dynamic dominance result.

### V20 Particle-Heavy Rerun Does Not Improve V18
- The first v20 profile, `particle_heavy_flux_v7_b1p10_p1p55`, passes the
  structural gate but falls slightly below the earlier v18 gate:
  overall margin `0.052366` vs `0.053378`, event margin `0.054219` vs
  `0.055077`.
- The best dynamic family is unchanged:
  `dynamic:auto_non14_event15_lead0`.
- This does not justify another PPO launch by itself. Continue the profile
  scan and only launch PPO if `event_flux_particle_v7` or
  `dual_flux_particle_v7` improves the behaviour-valid deployable-static and
  event margins.

### V20 Event-Flux Profile Improves Overall Margin But Not Event Margin
- `event_flux_particle_v7_b1p10_p1p55` passes and raises the overall
  behaviour-valid dynamic margin to `0.063723`, above v18's `0.053378`.
- Its event margin is only `0.051035`, below v18's `0.055077`.
- The best dynamic family remains `dynamic:auto_non14_event15_lead0`, while
  the best deployable static source shifts to
  `met_station_core|radiometer_basic|laser_disdrometer`.
- Interpretation:
  this profile may improve the strict overall static-break gate, but it does
  not cleanly improve event-side separation. Hold judgment until
  `dual_flux_particle_v7` finishes and compare event/calm decomposition.

### V20 Profile Scan Selects Event-Flux Only As A Diagnostic PPO Target
- Full scan result:
  `event_flux_particle_v7` is best on overall margin (`0.063723`), but not on
  event margin (`0.051035`).
- `dual_flux_particle_v7` and `particle_heavy_flux_v7` stay near v18 overall
  margin and also do not exceed v18's event margin.
- Therefore the only justified learned probe is a diagnostic reduced PPO on
  `event_flux_particle_v7`, not a profile promotion. The acceptance rule is:
  pass only if learned PD-PPO beats the static families and original/duty
  dynamic baselines with clean duty behaviour. If it fails, the profile scan
  does not rescue strict original-dynamic dominance.

### V20 Event-Flux PPO Fails By The Same Event-Side Mechanism
- The `event_flux_particle_v7` target-profile change produced a stronger TCN
  structural overall margin, but it did not transfer to learned PPO.
- Completed PPO loss:
  `custom_ppo=0.401974`.
- It loses:
  best static `0.398205`, deployable selected static `0.401011`, best
  deployable static `0.400316`, original round-robin `0.397568`, and duty
  round-robin `0.396908`.
- Behaviour is clean:
  `mid=8`, `always_on=0`, `always_off=0`, `warmup_abort=0`,
  switch `0.037057`.
- Event/calm decomposition identifies the same transfer failure:
  PD-PPO event `0.518869` vs duty round-robin event `0.508371`; PD-PPO
  non-event `0.266603` vs duty round-robin non-event `0.267827`.
- Learned sensor use remains SPC-heavy with low laser/FC4:
  `snow_particle_counter` event duty `0.718380`,
  `laser_disdrometer` event duty `0.134668`, and `fc4_flux` event duty
  `0.146952`.
- Conclusion:
  the profile scan does not rescue strict original-dynamic dominance. Do not
  run more same-recipe v20 PPO variants.

### V20 Event-Pair Replay Does Not Rescue The Branch
- Direct replay on the completed v20 event-flux split-run oracle tested both
  the structural laser pair, FC4-heavy action30 alternatives, and a broader
  top-auto scan of remaining behavior-valid single-pair candidates.
- The structural gate pair `auto_non14_event15` transfers only partially:
  lead0 gives `0.401146`, better than learned v20 PPO (`0.401974`) but still
  below the strict static/dynamic baselines.
- The best replayed pair is `eventflux_auto_non2_event15_l0 = 0.400381`,
  with valid behaviour (`mid=8`, no always-on/off sensors, zero aborts).
- That pair beats deployable selected static (`0.401011`) by `0.000630`, but
  still loses best deployable static (`0.400316`), best static (`0.398205`),
  original round-robin (`0.397568`), and duty-constrained round-robin
  (`0.396908`).
- Event/calm split shows why:
  best top-auto replay has event `0.511723` and non-event `0.271440`, while
  duty round-robin has event `0.508371` and non-event `0.267827`, and AoI has
  event `0.505721`.
- Conclusion:
  v20's structural margin is not enough under the actual split-run oracle and
  eval-start protocol. The failure is no longer just learned PPO underusing
  laser/FC4; even direct dynamic pairs miss the strict dynamic baselines. Close
  this branch and move to a new structural direction if strict original-dynamic
  dominance remains required.

### V21 Bursty Event Geometry Is Not A PPO Target
- V21 changed structure rather than just target weights:
  lower event coverage, shorter separated events, higher flux exponent, and
  stronger event microstructure.
- Final structural scan:
  - `particle_heavy_flux_v7`: overall margin `0.017244`, event margin
    `-0.023047`, formal pass only because overall loss improves;
  - `event_flux_particle_v7`: overall margin `-0.010957`, event margin
    `0.006708`;
  - `dual_flux_particle_v7`: overall margin `-0.022565`, event margin
    `-0.068546`.
- Conclusion:
  the bursty-event geometry separates event/calm behavior, but not in a usable
  direction. It either improves non-event loss while worsening event loss, or
  improves event loss while losing overall. Do not launch PPO from v21.

### V22 FC4 Boundary Restores Structural Headroom, With A Laser Caveat
- Raising only FC4 cost from `0.54/0.70` to `0.72/0.96` breaks the
  FC4-heavy static shortcut family seen in v20 validation candidates.
- All v22 profiles pass the deployable-static structural gate:
  `event_flux_particle_v7` is strongest with overall margin `0.059582` and
  event margin `0.044922`.
- However, the deployable static reference shifts to
  `met_station_core|radiometer_basic|laser_disdrometer`.
- Interpretation:
  v22 is worth exactly one learned PPO diagnostic because it creates both
  overall and event headroom. It is not yet a clean proof that all static
  shortcuts are gone; the learned policy must beat the laser static reference
  and dynamic baselines with valid duty behavior.

### V22 Learned PPO Does Not Transfer Structural Headroom
- The v22 event-flux PPO diagnostic completed with valid deployment behaviour:
  `mid=8`, no always-on/off sensors, zero aborts, and switch rate `0.041361`.
- It fails every strict comparison:
  `custom_ppo=0.411906` versus best static `0.394480`, deployable selected
  static `0.394044`, best deployable static `0.393007`, original round-robin
  `0.401172`, and best duty non-PD-PPO `0.393007`.
- The event/calm audit shows both sides are weak relative to the strongest
  references:
  PD-PPO event/non-event `0.529296/0.275961`, duty validation static
  `0.500483/0.270780`, duty feasible static `0.510280/0.257198`, duty
  round-robin `0.524250/0.257653`.
- Sensor duty explains the failed transfer:
  the policy remains met/radiometer/SPC-heavy, laser event duty is only
  `0.122384`, and FC4 event duty is lower than non-event duty
  (`0.174704` vs `0.242360`).
- Conclusion:
  the FC4 boundary alone is insufficient. It creates TCN structural headroom
  but does not make the learned policy beat static/dynamic baselines. Do not
  launch same-recipe v22 PPO variants. The only justified next diagnostic is a
  direct event-pair replay on the completed split oracle.

### V22 Direct Replay Shows The Final-Eval Shortcut Is Action 2
- Direct event-pair replay on the completed v22 split-run oracle improves over
  learned PPO but still misses strict static gates.
- Best direct pair:
  `v22_eventflux_auto_non2_event15_l0=0.396653`, event `0.513243`,
  non-event `0.261634`, valid behaviour.
- Behaviour-valid structural pair:
  `v22_eventflux_auto_non7_event15_l0=0.396882`, event `0.516283`,
  non-event `0.258608`, valid behaviour.
- These beat learned PPO (`0.411906`) and original round-robin (`0.401172`),
  but lose best static `0.394480`, deployable selected static `0.394044`, and
  best deployable static `0.393007`.
- Static-mask replay identifies the shortcut:
  `static_action2_core_surface_spc=0.394668` is strong, while pure laser
  static is weak (`static_action15_laser=0.420640`) and static FC4 is weaker
  (`static_action21_surface_fc4=0.435987`).
- Conclusion:
  v22 should be closed. The next structural test should not be another FC4 or
  laser-only tweak; it should make action 2 infeasible while preserving a
  feasible calm action 7 and event action 15.

### V23 Met/Laser Exchange Is The Next Structural Test
- V23 raises met and lowers laser by matched amounts:
  met `0.14/0.18 -> 0.33/0.38`, laser `0.86/1.18 -> 0.67/0.98`, FC4 held at
  v22 `0.72/0.96`.
- This makes the final-eval action-2 shortcut exceed B=`1.10`:
  `met+radiometer+surface+SPC = 1.11`.
- It preserves the intended behavior-valid dynamic alternatives:
  calm action 7 `radiometer+surface+ultrasonic+SPC = 0.94`; event action 15
  `met+radiometer+laser = 1.10`, peak `1.49`.
- Launch mode:
  CPU tmux because all GPUs had active Python allocations. This avoids
  interfering with other server jobs while the structural screen runs.

### V23 Gate Selects Dual-Flux For One Learned Diagnostic
- V23 completed with all three scanned profiles passing after action 2 was
  made infeasible.
- `particle_heavy_flux_v7` has the strongest structural margins:
  overall `0.058551`, event `0.067801`. It is not the cleanest learned target
  because its best dynamic row uses `always_on=1` and `always_off=2`.
- `dual_flux_particle_v7` has smaller but positive margins:
  overall `0.030123`, event `0.022259`. Its best row
  `dynamic:diverse_top5_lead6_dwell12` is behaviour-clean:
  loss `0.380097`, event `0.527918`, non-event `0.227583`, `mid=8`,
  no always-on/off sensors, and switch `0.030884`.
- `event_flux_particle_v7` also passes, but its best row still has one
  always-off sensor and a slightly smaller overall margin than dual-flux.
- Interpretation:
  V23 is the first recent gate that directly breaks the action-2 shortcut
  identified by v22 replay while preserving a clean dynamic target. It still
  needs learned-policy confirmation; no multi-seed expansion is justified
  until the seed45 PPO beats static, deployable-static, original dynamic, and
  duty-constrained baselines with clean duty behaviour.

### V23 Learned PPO Does Not Transfer The Structural Headroom
- The v23 dual-flux learned policy is deployable and behaviour-clean:
  `mid=8`, zero always-on/off sensors, zero warmup aborts, and switch rate
  `0.032234`.
- It breaks the ordinary static shortcut:
  PD-PPO `0.449127` beats best static / selected static `0.452356` and
  deployable selected static `0.485782`.
- It is not a strict success:
  it loses best deployable static / best duty non-PD-PPO `0.438596` by
  `0.010531`, and loses AoI `0.447516` by `0.001611`.
- The failure mechanism is again low event-instrument transfer:
  laser duty `0.140625`, FC4 duty `0.128662`, and the top mask is the
  met/radiometer/shielded/SPC bundle for `41.99%` of steps.
- Event/calm decomposition:
  PD-PPO nearly matches AoI on event loss (`0.576956` vs `0.577165`) but loses
  calm loss (`0.301093` vs `0.297375`); duty feasible static beats it on both
  event and calm (`0.567254/0.289602`).
- Conclusion:
  v23 is a structural pass but not a learned-policy pass. A direct event-pair
  replay on the completed split-run oracle is the next cheap diagnostic; no
  seed expansion is justified.

### V23 Split-Oracle Headroom Requires A Cyclic Mask-Pool Policy
- Single calm/event pairs do not transfer:
  best direct event-pair replay is `v23_dual_auto_non6_event21_l0=0.450856`,
  worse than learned PPO and below static/dynamic/duty baselines.
- Exact cyclic replay of the gate's pure-static top-mask pools does transfer:
  `v23_dual_diverse_top5_l6_dwell12=0.437728`.
- This cyclic policy is behaviour-clean:
  event loss `0.557965`, non-event loss `0.298486`, `mid=8`, zero
  always-on/off sensors, zero aborts, switch `0.034035`, and top-mask fraction
  only `32.06%`.
- It beats every relevant seed45 reference:
  best static `0.452356`, learned PPO `0.449127`, AoI `0.447516`, and best
  deployable static / best duty non-PD-PPO `0.438596`.
- Mechanistic interpretation:
  V23 has real adaptive headroom, but the useful policy is not a binary
  event-laser switch. It is a cyclic event-conditioned mask pool that rotates
  several near-feasible static masks and uses lead-6 event anticipation.
- Next implication:
  do not change scene again yet. Add a cyclic teacher / mask-pool AWBC mode
  and run one reduced PPO probe to test whether the policy class can learn the
  now-validated dynamic mechanism.

### Cyclic Teacher Is The Correct Learnability Probe
- Existing `event_pair` AWBC can only imitate one calm mask and one event mask.
  That is insufficient for V23 because the successful replay is a mask pool:
  top5 calm masks, top5 event masks, lead `6`, dwell `12`.
- Added `event_cyclic` as a narrow teacher extension rather than changing the
  reward or scene:
  it cycles through resolved candidate action indices by episode phase and
  falls back to oracle-greedy if a target action is infeasible.
- The first cyclic-teacher PPO probe started cleanly and reached update `40`
  with `awbc_label_rate=1.000`, so candidate resolution and teacher labeling
  are working.

### AWBC0.8 Cyclic Teacher Nearly Transfers But Still Misses Duty Baseline
- Learned policy:
  `custom_ppo=0.441380`, valid behaviour, zero aborts.
- It beats static and original dynamic references:
  best static `0.447070`, AoI `0.449137`, full-open `0.460805`.
- It misses the strongest duty/deployable reference:
  duty feasible static `0.440551`, gap `0.000829`.
- The event/calm split shows the remaining issue:
  PD-PPO has better calm loss than duty feasible static (`0.287274` vs
  `0.289114`) but worse event loss (`0.574452` vs `0.571319`).
- Compared with exact cyclic replay, the learned schedule is too concentrated:
  top mask `42.48%` versus exact replay `32.06%`; laser duty `0.241943`
  versus exact replay `0.285156`.
- Next implication:
  one stronger cyclic-imitation probe is technically justified because the
  scene and teacher work, the margin is small, and the remaining failure is
  teacher fidelity rather than scenario structure.

### AWBC1.2 Shows The Miss Is Not A Simple Imitation-Strength Problem
- Stronger cyclic imitation completed with valid behaviour:
  `custom_ppo=0.440397`, zero aborts, `mid=8`, switch `0.035714`.
- It improves total loss over AWBC0.8 (`0.440397` vs `0.441380`) but fails the
  strict gate by more:
  duty feasible static `0.436732` is better by `0.003665`, and
  duty-constrained round-robin `0.439321` is better by `0.001076`.
- The event/calm decomposition explains the regression:
  AWBC1.2 has strong calm loss (`0.277932`) but much worse event loss
  (`0.580687`) than duty feasible static (`0.564365`) and AWBC0.8
  (`0.574452`).
- Mask fractions moved closer to exact cyclic replay, but the learned event
  composition did not:
  top mask `34.79%` versus replay `32.06%`, laser duty `0.345215` versus replay
  `0.285156`, and event laser duty `0.323476` versus replay `0.217925`.
- Conclusion:
  V23 remains a real structural static-break scene because exact cyclic replay
  passes, but the current feed-forward PPO + cyclic AWBC recipe is a
  learnability blocker. Do not continue coefficient sweeps or seed expansion
  on this recipe.

### Phase Visibility Is The Next Distinct Learnability Test
- The successful exact replay is not just event-conditioned; it is
  episode-phase-conditioned:
  top5 calm/event pools, lead `6`, dwell `12`, effectively a 5-mask cycle over
  `60` steps.
- Before the phase probe, PPO could infer that cycle only indirectly from
  previous action, duty estimates, sensor freshness, and time-of-day. It did
  not observe `(current_idx - episode_start_idx) mod 60`.
- This makes phase exposure a different failure test than AWBC coefficient
  tuning:
  if phase60 still misses the strict duty/deployable reference, the blocker is
  likely not hidden cycle state alone.

### Phase60 Closes V23 As A Learned-PPO Route
- Phase-aware PPO failed more broadly than AWBC0.8:
  `custom_ppo=0.447119`, losing validation-selected static, AoI,
  duty-constrained round-robin, and duty feasible static.
- The global mask distribution was not the problem after phase exposure:
  phase60's top four mask fractions nearly matched exact replay
  (`32.01/22.97/15.09/10.72%` vs `32.06/20.31/13.48/11.72%`).
- The same-run exact replay control is the decisive caveat:
  `phase60_exact_diverse_top5_l6_dwell12=0.437319` is clean and beats learned
  phase60 PPO by `0.009800`, but loses the same-run duty feasible static
  (`0.437106`) by `0.000212`.
- Conclusion:
  V23 has useful dynamic structure, but its strict split-oracle margin is too
  small and sensitive to oracle retraining. Learned-PPO failure should not be
  chased further on this scene.

### Minor Cyclic Timing Tweaks Do Not Rescue V23
- Same-run replay sweep over the same top5 pools found:
  `l3/dwell12=0.439674`, `l6/dwell6=0.441866`,
  `l0/dwell12=0.445886`, `l6/dwell24=0.448719`.
- The best row is behaviour-clean but still worse than same-run duty feasible
  static `0.437106` by `0.002568`.
- Conclusion:
  the next branch cannot be another V23 timing/teacher variant. It must first
  create a larger same-run exact dynamic replay margin.

### Phase 14 Gate Must Use The Split-Run Oracle Reference
- V23 exposed a gate mismatch:
  the standalone TCN structural gate reported a clean dual-flux dynamic margin
  of `0.030123`, but replaying the same cyclic mechanism against a retrained
  split-run oracle gave only `0.437728` versus `0.438596` in the original run
  and `0.437319` versus same-run `duty_constrained_feasible_static_projected`
  `0.437106` in the phase60 control.
- The hard reference is therefore not ordinary static alone. It is the best
  same-run deployable/duty reference, especially
  `duty_constrained_feasible_static_projected`, because that baseline applies
  the same duty guard and projection machinery that repeatedly absorbs the
  apparent dynamic advantage.
- Acceptance for the next branch:
  first pass a TCN structural screen, then create a zero-PPO split-run source
  (`total_timesteps=0`, oracle and baselines only) and replay the exact dynamic
  policy on the same oracle/final-test starts. Do not launch PPO unless replay
  beats the best duty/deployable reference by at least `0.005` absolute loss or
  `1%` relative, whichever is larger.

### V24 Hypothesis: Make Laser Event-Selective
- V23 same-run replay decomposition:
  exact cyclic replay event loss `0.558937` beats duty feasible static
  `0.566329`, but non-event loss `0.296477` loses duty feasible static
  `0.287458`.
- The duty-static policy uses laser almost uniformly
  (`event duty 0.330`, non-event duty `0.326`), while exact cyclic replay
  unexpectedly uses laser more in non-event windows (`0.363`) than event
  windows (`0.218`) because the V23 normal laser noise is too good.
- V24 changes the information structure, not the cost boundary:
  it keeps V23 powers and degrades non-event laser particle noise to
  `0.16/0.45`, while event laser noise is `0.08/0.22` with event observation
  probability `0.88`.
- Expected gate signal:
  top non-event static masks should stop relying on laser, top event masks
  should still include laser, and split-run cyclic replay should widen the
  margin against `duty_constrained_feasible_static_projected`.

### V24 Particle-Heavy Stage-1 Pass Is Necessary But Not Sufficient
- Under the event-selective laser sensor config, the first Stage-1 profile
  `particle_heavy_flux_v7` passed the TCN structural gate:
  best valid dynamic `0.361329` versus deployable-static reference `0.393251`.
- The best valid dynamic row still has a borderline operational shape:
  `5` mid-duty sensors, `1` always-on sensor, `2` always-off sensors, and
  switch rate `0.028320`. This is acceptable for the structural screen but
  not yet a learned-policy claim.
- The important decision remains the Phase 14 same-run replay gate. V23 showed
  that a standalone TCN margin can disappear after split-oracle retraining, so
  no PPO seed expansion or paper-mainline migration is justified until
  `70_v31_split_replay_gate.py` confirms a margin over the same-run
  duty/deployable reference.

### V24 Particle-Heavy Is The First Phase-14 Pre-PPO Pass
- V24 `particle_heavy_flux_v7` passed the stricter same-run split-oracle replay
  gate that was introduced after V23 failed to transfer standalone TCN
  headroom.
- The gate's same-run best reference was `aoi=0.429470`, stronger than the
  duty feasible static row (`0.432382`) in this split source.
- Best replay was `split_top2_l6_dwell12=0.414078`, giving a margin of
  `0.015392` absolute / `3.58%` relative over the reference, with clean
  deployment behaviour (`mid=8`, no always-on/off sensors, switch `0.043712`,
  no warmup aborts).
- This is materially stronger than V23's exact replay margin, which was only
  `0.000868` in one source and vanished under phase60 same-run replay. V24
  therefore justifies exactly one reduced learned-PPO diagnostic.
- It is still not yet a paper-mainline learned PD-PPO result. The learned PPO
  must now reproduce the replay mechanism or beat the same-run references
  directly; otherwise V24 remains a scenario/replay finding rather than a
  validated PD-PPO contribution.

### V24 Seed45 Is The First Learned Single-Seed Pass In The V20+ Series
- The V24 particle-heavy cyclic-teacher PPO run transferred enough of the
  replay mechanism to beat all same-run reference families on seed45:
  `custom_ppo=0.451010`.
- Comparisons:
  best static / selected static `0.477724`, deployable selected static
  `0.513591`, best original dynamic `aoi=0.464753`, and best duty/deployable
  non-PD-PPO `duty_constrained_feasible_static_projected=0.453601`.
- Behaviour is clean:
  `mid=8`, zero always-on/off sensors, zero aborts, switch rate `0.039286`,
  duty range `0.226318--0.742188`.
- The strongest caution is margin size:
  the learned margin over the best duty/deployable reference is `0.002591`,
  much smaller than the pre-PPO exact replay margin `0.015392`. This may still
  be seed/oracle sensitive.
- Current answer to paper-mainline migration:
  not yet. V24 is now the first credible candidate that does not require
  changing the paper contribution framing, but it needs locked seed
  replication before it can be treated as an explicit mainline PD-PPO result.

### V24 Multi-Seed Replication Rejects Learned Mainline Promotion
- Locked seeds `41--45` completed under the same V24 event-selective laser,
  particle-heavy, B=`1.10`, h=`0.82`, dwell12, cyclic-teacher AWBC0.8
  configuration.
- The positive part is behaviour:
  `5/5` seeds satisfied the deployment gate. PD-PPO had `mid=8`, no
  always-on/off sensors, no warmup aborts, duty ranges within the intended
  hard bounds, and switch rates around `0.035--0.039`.
- The negative part is decisive:
  only seed45 beat all same-run reference families. Learned PD-PPO beat
  best original dynamic only `1/5` and best duty non-PD-PPO only `1/5`.
  It beat best deployable static only `2/5`.
- The aggregate comparison table is not paper-mainline-safe:
  mean baseline-minus-PD-PPO deltas were `-0.012423` for best deployable
  static, `-0.014409` for best original dynamic, and `-0.011868` for best duty
  non-PD-PPO. Negative means PD-PPO is worse on average.
- Seed-level failure pattern:
  seed41 lost to round-robin by `0.043684`; seed42 lost to
  validation-selected static by `0.042580`; seed43 lost to round-robin by
  `0.005401`; seed44 lost to round-robin by `0.015703`; seed45 was the only
  strict pass, with a narrow `0.002591` margin over
  `duty_constrained_feasible_static_projected`.
- Interpretation:
  V24 validates the stricter Phase-14 two-stage gate as a screening method
  and gives a useful single-seed learned diagnostic, but it does not provide a
  stable learned PD-PPO result that can be moved into the first-paper mainline
  without changing the contribution framing.
- Avoid repeating:
  do not add more seeds to this exact particle-heavy cyclic-teacher recipe.
  Any next PPO run needs a new mechanism or a new split-replay pass that
  directly addresses learned transfer across seeds.

### V24 Dual/Event Replay Gates Preserve Structural Headroom
- After particle-heavy learned transfer failed multi-seed, the remaining V24
  Stage-1-passing profiles were tested under the stricter same-run split
  replay gate.
- `dual_flux_particle_v7` passed:
  best replay `split_top2_l0_dwell12=0.410668` versus
  `validation_selected_static=0.417963`, margin `0.007295` absolute /
  `1.745%` relative. Behaviour is clean: `mid=8`, no always-on/off, switch
  `0.044048`.
- `event_flux_particle_v7` passed more strongly:
  best replay `split_top2_l0_dwell12=0.406600` versus `aoi=0.416698`,
  margin `0.010099` absolute / `2.423%` relative. Behaviour is the same clean
  lead-0 dwell-12 top-2 cyclic structure.
- The shared winning mask pools are:
  calm `radiometer+surface+shielded+laser` and
  `radiometer+shielded+SPC`;
  event `surface+shielded+FC4` and `surface+shielded+SPC`.
- Interpretation:
  V24 still has structural headroom outside particle-heavy, but learned
  evidence must be re-established. Event-flux is the better single diagnostic
  because its replay margin is larger and its teacher has no lead offset.

### V24 Event-Flux AWBC0.8 Misses Strict Learned Gate Narrowly
- The event-flux AWBC0.8 learned diagnostic preserved the desired deployment
  behaviour: `mid=8`, no always-on/off sensors, no warmup aborts, switch
  `0.035501`, duty range `0.226318--0.742188`.
- It beats the deployable/duty static comparison by a very small margin:
  `custom_ppo=0.418312` versus
  `duty_constrained_feasible_static_projected=0.418446`.
- It still fails the strict learned gate:
  it loses `aoi=0.416698` by `0.001614`, full-open `0.415783` by `0.002529`,
  and raw feasible static `0.418157` by `0.000155`.
- This is not a seed-expansion candidate. The only remaining minimal test is
  whether stronger imitation can transfer the known replay schedule more
  faithfully; otherwise V24 event-flux should be closed as another replay-only
  candidate.

### V24 Event-Flux AWBC1.2 Shows Stronger Imitation Is Not Enough
- The stronger-imitation diagnostic stayed behaviour-clean:
  `mid=8`, no always-on/off sensors, no warmup aborts, switch `0.037027`, duty
  range `0.238037--0.742188`.
- It does recover two reference wins in its own run:
  `custom_ppo=0.436344` beats full-open `0.441786` and AoI `0.440952`.
- It still fails the strict migration gate decisively:
  loses selected/best static `0.412144` by `0.024201`, deployable selected
  static / best deployable static `0.425520` by `0.010824`, and best duty
  non-PD-PPO `0.432757` by `0.003587`.
- Interpretation:
  the V24 event-flux replay margin is real, but current cyclic-teacher PPO
  does not learn a policy that beats the static and duty-constrained shortcuts.
  This closes same-recipe AWBC coefficient tuning as a paper-mainline path.

### V24 Event-Flux Phase24 Improves Dynamic Wins But Still Loses Static
- The phase-visible probe is behaviour-clean:
  `mid=8`, no always-on/off sensors, no warmup aborts, switch `0.042949`,
  duty range `0.239258--0.740479`.
- It improves the dynamic/full-open side of the learned result:
  `custom_ppo=0.423954` beats full-open by `0.011504`, AoI by `0.009732`,
  and best duty non-PD-PPO by `0.003156`.
- It still fails the strict static-break gate:
  loses selected/best static `0.408230` by `0.015724` and deployable selected
  static / best deployable static `0.419936` by `0.004017`.
- Mechanism comparison:
  AWBC0.8 was closest to deployable static but missed full-open/AoI; AWBC1.2
  and phase24 recover dynamic wins but remain worse against the static
  shortcut. The blocker is no longer just hidden cycle phase.
- Interpretation:
  V24 event-flux is not a seed-expansion candidate under current learned
  transfer. Any next V24 action must be justified by replay/static-reference
  audit, not by another small PPO tuning knob.

### Split-Replay Gates Must Enforce Replay-Local Static References
- V24 event-flux exposed a gate-contract bug:
  the old split-replay gate reported pass against AoI (`0.416698`) because
  best replay was `0.406600`, but the same replay-local static candidate table
  contains `static_action8=0.403818`.
- That means event-flux never had strict static-break headroom at the replay
  stage; learned PPO could not reasonably be expected to overcome a static
  reference stronger than the teacher replay.
- The corrected gate now requires both:
  source-run reference margin and replay-local best-static margin.
- Under the corrected gate:
  event-flux fails (`margin_abs_vs_static_reference=-0.002782`), while
  dual-flux passes (`best replay=0.410668`, replay-local best static
  `0.418077`, margin `0.007409`).
- Interpretation:
  close V24 event-flux, including AWBC0.8, AWBC1.2, and phase24. V24 dual-flux
  is the only remaining V24 profile with strict replay evidence and deserves
  the next learned confirmation probe.

### V24 Dual-Flux Phase24 Is The New Single-Seed Learned Candidate
- No-phase dual-flux learned PPO is behaviour-clean and beats many references,
  but still loses the best deployable/duty reference by `0.005498`; it should
  not be expanded.
- Phase24 dual-flux learned PPO is the first result after the strict-static
  gate fix that passes every same-run learned reference:
  `custom_ppo=0.440622`, full-open margin `0.016947`, best/selected static
  margin `0.014871`, deployable selected static margin `0.045126`, AoI margin
  `0.010888`, and best deployable / best duty non-PD-PPO margin `0.000790`.
- Behaviour is clean:
  `mid=8`, no always-on/off sensors, zero warmup aborts, switch `0.042369`,
  duty range `0.236572--0.741699`.
- The caution remains margin size:
  the decisive best duty/deployable margin is under `0.001`. This is a valid
  locked-seed expansion candidate, not yet a paper-mainline result.

### V24 Dual-Flux Phase24 Does Not Replicate As A Learned Mainline Result
- Locked seeds `41--45` completed under the same V24 dual-flux phase24
  AWBC0.8 setup. All five seeds retained the desired operational behaviour:
  `pdppo_valid_behavior=5/5`, `mid=8`, zero always-on/off sensors, and zero
  warmup aborts.
- The learned performance did not replicate. Strict win counts were:
  full-open `4/5`, best static `1/5`, selected static `1/5`, deployable
  selected static `2/5`, best deployable static `2/5`, best original dynamic
  `2/5`, and best duty non-PD-PPO `1/5`.
- Aggregate margins are negative for the references that matter most:
  best static `-0.023397`, selected static `-0.021260`, deployable selected
  static `-0.003243`, best deployable static `-0.014460`, best original
  dynamic `-0.015137`, and best duty non-PD-PPO `-0.012267`. Only full-open is
  positive on average (`+0.004442`), which is insufficient for the static-break
  claim.
- Interpretation:
  phase visibility and the corrected strict replay gate were useful
  diagnostics, but the single-seed seed45 success was not stable. The V20+
  series still has no learned PD-PPO result that can be migrated into the
  first-paper mainline without changing the contribution framing or adding a
  new structural/training mechanism.
- Avoid repeating:
  do not launch more same-recipe V24 dual-flux phase24 PPO seeds or AWBC
  coefficient tweaks. The next branch must start from a new mechanism that
  improves multi-seed learned transfer, not from another seed expansion of the
  current cyclic-teacher recipe.

### V25 Low-Budget Squeeze Is The Next Structural Test
- The V24 multi-seed failures are not caused by invalid deployment behaviour:
  all locked dual-flux phase24 seeds were behaviour-clean. The failure is that
  several seeds still have stronger static or high-frequency dynamic reference
  policies.
- The next structural lever should therefore change the feasible static set
  before another PPO run. Lowering B from `1.10` to `1.03--1.05` is a targeted
  test because:
  - event FC4 remains feasible at B=`1.03`
    (`surface_temp_ir + shielded_thermo_hygro + fc4_flux = 1.03`);
  - the calm-laser static bundle is excluded until B=`1.08`;
  - the met-laser static bundle remains excluded until B=`1.10`.
- This is a structural gate, not a learned-policy retry. Acceptance remains:
  TCN dynamic schedule must pass the strict behaviour filter and beat
  deployable/static references by material margin; any PPO launch still
  requires a later same-run split-replay pass.
### 2026-06-21 Active Goal Audit
- The API goal remains active but is not fully precise: it still names BO-1,
  which has already been pivoted away from. The tool does not allow editing the
  objective text in place, only marking the goal complete or blocked.
- The correct active research objective is the local `research-state.yaml`
  objective: continue autonomous PD-PPO strong-claim exploration until the
  evidence supports forecast-optimal, non-fixed, non-cyclic scheduling under the
  tested protocol.
- This objective matches the user's latest constraints:
  PPO remains the final scheduler; modifications may move beyond scene tuning
  into simulator/data, teacher/oracle, PPO features/auxiliary heads/memory,
  reward/evaluation, and moderate explainable sensor/noise variants; each
  direction has a 10-unit anti-stall limit.
- The live plan pointer previously targeted an ESWA terminology rewrite plan.
  That was safe for manuscript work but wrong for the current autonomous
  experiment loop. It has been restored to the PD-PPO static-break
  recalibration plan.
- SCENEBAL-1 is currently the active direction because it has effective
  improvement and multi-seed breakthrough evidence. It should not be abandoned
  merely because BO-1 was stopped; BO-1 is historical evidence and SCENEBAL-1 is
  the current simulator/target-generation branch.

### 2026-06-21 SCENEBAL-1 18-Seed Finding
- SCENEBAL-1 `93--110` is the first branch to reach `18/18` operational step,
  operational macro, strict explicit replay step/macro, and behavior gates.
- This satisfies the strongest currently defensible operational version of the
  user's target: PPO is forecast-best against validation-selected static and
  rule-dynamic baselines, and the learned behavior is not fixed or a simple
  cycle.
- The apparent true-static macro blocker was a metric-scale artifact in the
  oldclaim collector. After replay-normalized recomputation, learned true-static
  macro is `18/18`.
- The maximal version is still not complete only because true-static step is
  `17/18`: seed95 has a positive but sub-threshold margin against true fixed
  static. The next action should diagnose that strict-margin case rather than
  spend the next unit on blind seed expansion.
- Seed95 diagnosis confirms it is a strict-margin artifact rather than a sign
  failure. PPO beats the true fixed static reference on seed95 by
  `0.0017415271440766045`, but the configured relative-margin gate requires
  `0.003906182191737571`.

### 2026-06-21 Paper Claim Finding
- The canonical ESWA manuscript has been moved from the stale `14`-seed
  macro-only claim to the corrected SCENEBAL-1 `18`-seed claim.
- Manuscript-supported:
  operational step/macro `18/18`, explicit replay step/macro `18/18`, behavior
  `18/18`, replay-normalized true-static macro `18/18`, and positive
  true-static step margins `18/18`.
- Manuscript boundary:
  strict-margin true-static step is `17/18`, with seed95 positive but below the
  configured threshold.
- Do not return to the old text:
  `13/14` macro, `10/14` step, and ten-seed duty/dwell highlights are now stale
  and were removed from the checked main manuscript/highlights.

### 2026-06-21 Seed-Margin Risk Finding
- Seed95 is an isolated strict-margin boundary. It is the only seed below
  `0.005` true-static step margin and the only seed below `0.02`; the next
  lowest margin is seed98 at `0.020629`.
- Distribution over corrected 18-seed replay-local true-static step margins:
  min `0.001742`, median `0.082456`, mean `0.087145`, max `0.181463`.
- Stress wave `111--116` should be treated as robustness testing. Pivot only if
  it reveals repeated true-static sign failures, behavior collapse, or loss of
  explicit replay dynamic headroom.

### 2026-06-21 Manuscript Evidence Figure Finding
- The canonical ESWA PDF now contains a seed-level SCENEBAL-1 evidence figure
  in the results section.
- The figure makes the remaining boundary visually explicit: seed95 is positive
  against true fixed static but below the predefined strict margin, while the
  aggregate gates remain `18/18` except for the strict-margin true-static step
  gate at `17/18`.
- The figure strengthens the current paper posture because it separates the
  main claim from the boundary case: all-seed operational/replay/behavior/
  true-static macro evidence is positive, and the only caveat is a single
  sub-threshold ordinary step margin rather than a sign failure or behavior
  collapse.
- The local monitor scripts have also been repaired so the continuing
  stress-wave watch will not pollute logs with `printf` option errors. This is
  operational hygiene, not a change to evidence.

### 2026-09-09 Flexible-subset V539--V540
- V539 preserved the frozen forecaster and target process while using online alert proxies for state-dependent effective load. It produced subset-level opportunity but poor chronological transfer.
- Adding clock phase features in V540 did not solve transfer. The corrected no-leakage probe remained below the provisional `0.15` top-1 gate on every seed; the contaminated first probe is explicitly excluded.
- The current bottleneck is online identifiability, not PPO optimization: condition-dependent subset optima exist, but the present observations do not identify them across the chronological train/test boundary.
- V541 is the last bounded physical-input diagnostic in this branch. It uses observable humidity, wind, and radiation to form continuous heater demand without changing truth targets.
- V541 confirms the separation between geometry and transfer. The physical-input resource trace keeps positive opportunity and no epsilon=`0.01` static intersection, but the valid chronological probe passes the `0.15` top-1 gate for only one of four seeds. More PPO tuning would therefore be premature.

### 2026-09-09 Objective-Level Gate Revision
- `docs/09-02-01.md` identifies the remaining gap: row-wise channel-quality
  rank flips do not prove that downstream forecast-optimal complete subsets
  change. A static subset can combine several specialists and remain
  near-optimal across all regimes.
- The next decisive artifact must therefore be a subset-level loss matrix. For
  each feasible subset `S` and condition/block `c`, record frozen-forecaster
  loss `L_c(S)`, condition-optimal subsets, static regret, and the intersection
  of epsilon-optimal sets.
- The useful adaptive-opportunity ceiling is the gap between the selected
  static policy and a common-dwell/startup condition-adaptive diagnostic. This
  diagnostic is privileged and must not be promoted as a fair baseline.
- The V541-16 probe is allowed to finish because it tests whether broader train
  coverage plus a larger probe model repairs transfer without touching the
  scheduler. It is the final bounded observability intervention in this branch.
- If subset-loss geometry is weak, the correct conclusion is that the flexible
  high-budget regime removed the original incompatibility. If geometry is
  strong and online-observable but PPO still fails, the next clean algorithm
  intervention is decision-epoch/singleton-lock PPO semantics; no
  bandit-derived prior or reward patch is permitted.

### 2026-09-09 V541 Expanded Probe Result
- Increasing the chronological training coverage from 8 to 16 starts and the
  diagnostic probe width from 128 to 256 did not establish transfer. Test
  top-1 was `0.1489/0.0769/0.1758/0.0674`, mean `0.1173`; train top-1 was
  `0.9337/0.9986/0.9931/0.9959`.
- The train/test gap makes a capacity explanation unlikely. The current
  meteorology/resource process is nonstationary across the chronological split
  in a way that prevents stable identification of the forecast-optimal subset
  from the available online observations.
- This is a pre-PPO scene/observability failure. Do not loosen the gate or
  convert the diagnostic into a policy result.
- The next decisive test is objective-level subset geometry: evaluate all
  feasible subsets by operating block under the frozen forecaster, then inspect
  the epsilon-optimal intersection and a common-dwell adaptive opportunity
  ceiling. A new stationary/cyclic truth branch is considered only after this
  audit identifies whether the current resource regime itself has a useful
  forecast-loss crossover.

### 2026-09-09 V542 Geometry Launch
- V542 is the first direct test of the revised scientific question: whether
  complete feasible-subset forecast losses, not individual sensor quality,
  produce incompatible condition-specific optima.
- The audit is restricted to frozen V541 assets and chronological training
  blocks. It will not authorize a policy wave unless the epsilon-optimal static
  intersection is empty with material regret and the opportunity is executable.

### 2026-09-09 V542 Clipped-Oracle Audit Correction
- The first V542 run was stopped after inspecting its first three seed outputs.
  V541's frozen TCN metadata sets `loss_clip=100.0`; observed subset losses
  were concentrated near `95--100`, making absolute `epsilon=0.01/0.05`
  intersections numerically uninformative.
- Those clipped results are retained as an implementation/scale diagnostic,
  not as evidence of adaptive opportunity.
- A sensitivity audit with the same frozen model and `loss_clip=1e6` is now
  running under the explicit output directory
  `reports/analysis/v542_subset_forecast_geometry_train_unclipped_20260909/`.
  It uses predeclared absolute epsilons `1, 5, 10`; relative-regret summaries
  will be computed after completion. This does not alter any policy asset.

### 2026-09-09 V542 Unclipped Geometry Finding
- Removing the diagnostic loss clip revealed real but small subset-level
  crossover. Event-condition opportunity was `0.88%--1.71%`; robust
  operating-condition opportunity was `0.35%--1.04%` across the four seeds.
- The result does not meet the current scene-readiness standard because a
  5%-relative near-optimal intersection still contains multiple static subsets.
  The empty intersections from the clipped run were not valid evidence.
- The correct next test is a predeclared budget-phase screen. If a lower
  physical budget creates a larger objective-level opportunity without
  eliminating the specialist channels, it becomes the next candidate scene.
  Otherwise a new resource/truth branch is required before PPO.

### 2026-09-09 V543 Budget Phase Finding
- The short budget screen shows that resource geometry, not merely the
  meteorological signal, controls the available adaptive opportunity. `B=2.25`
  retains 22 feasible masks and produces the most consistent improvement over
  `B=4.0` in the four-seed screen, although one seed remains weak.
- `B=1.75` is not promoted because it approaches the laser/base-power
  breakpoint and risks removing the particle specialist from the action
  surface. `B=2.25` is the lowest screened regime that retains the laser
  channel while allowing multiple low-power combinations.
- A full geometry audit is required before any observability or PPO work.

### 2026-09-09 V544 Finding
- B=2.25 preserved a nominal 22-mask action family but did not produce a
  meaningful forecast-loss frontier. Across four seeds, condition-level
  relative opportunity was `0.0207%`, `0.9290%`, `0.0207%`, and `1.0090%`;
  operating-condition opportunity was `0%`, `0.9721%`, `0.2488%`, and `0%`
  after retaining groups with at least 1024 samples.
- The condition-level near-optimal static intersection at 5% remained
  non-empty for all seeds. Operating-condition intersections were also
  non-empty except seed 7242, where the opportunity was still below 1%.
- The failure is attributable to the physical breakpoint: the 50-W laser
  heater makes the laser unavailable under B=2.25, so the nominal arbitrary
  subset surface is effectively a low-power met/radiometer/IR/FC4 problem.
  This branch is closed as a candidate main scene, not as evidence against
  arbitrary-subset PD-PPO.

### 2026-09-09 V545 Launch
- Started policy-free geometry screens at B=12, 16, and 20 W. These budgets
  were selected from the declared resource trace occupancy (`21%`, `47%`, and
  `72%` laser feasibility), before inspecting any PPO result.
- All runs use frozen V541 truth/forecaster assets, two chronological starts,
  four seeds, and the unclipped diagnostic loss scale. The next decision is
  based on complete-subset forecast geometry, not channel-level quality.

### 2026-09-09 V545 Short-Screen Finding
- The short hardware-breakpoint phase confirms that the 50-W laser load creates
  meaningful resource regimes only above approximately B=8 W. At B=12, 16,
  and 20 W the laser is feasible for roughly 21%, 47%, and 72% of the trace.
- B=20 produced the strongest short-window condition-level crossover, but this
  is not sufficient evidence because the operating-state partition was sparse.
  Full temporal coverage is required before any transfer probe.
- B=12 and B=16 remain diagnostic records. They are not promoted to PPO and
  are not treated as failed policies.

### 2026-09-09 V546 Launch
- Started the full eight-window B=20 subset audit on the frozen V541 assets.
- The audit will test whether the apparent short-window crossover survives
  chronological coverage and whether a near-optimal static subset remains over
  common operating states. No online transfer or PPO has started.

### 2026-09-09 V546 Finding
- Full B=20 geometry preserved state-dependent condition winners, but the
  downstream forecast-loss opportunity remained small: `1.27%--2.52%` across
  condition views and `1.94%--2.71%` across common operating-state views.
- A 5%-relative near-optimal static intersection remained in three of four
  seeds. The physical resource trace alone therefore does not break the
  static shortcut strongly enough to justify transfer or PPO.

### 2026-09-09 V547--V548 Finding/Launch
- Applying the exposure-coupled heater-quality relation is a clean scenario
  intervention because it changes observation reliability through the same
  physical heater state while preserving target columns and event labels.
- The generated quality traces reduce mean GMX500/Parsivel quality to roughly
  `0.73/0.53--0.54`, with about `92%--94%` of rows below full quality. This is
  a truth/quality audit, not a policy result.
- The compatible asset-only preparation completed for all four seeds. V548 is
  now screening whether that physical quality coupling creates a material
  complete-subset forecast frontier before any policy work.

### 2026-09-09 V548 Finding
- The heater-quality relation materially increases condition-level subset
  separation relative to the resource-only branch, but the effect is not yet
  stable across four seeds. Two seeds retain a 5%-relative static shortcut.
- The operating-state audit is currently unavailable because the prepared
  asset truth retained effective resource costs but not heater-state columns.
  Any later operating-state gate must explicitly preserve those columns.

### 2026-09-09 V549 Launch
- Started a full eight-window audit for the quality-coupled B=20 scene. This
  is still a frozen-asset geometry test; no online or PPO evidence is claimed.

### 2026-09-09 V549 Finding and V550 Launch
- The quality restoration relation creates condition-dependent winners, but
  downstream separation remains small: relative condition-level opportunity
  is `0.7881%--1.6499%` across seeds `7241--7244`. The 5%-relative static
  intersection is empty in every seed, but this alone does not establish a
  strong adaptive opportunity. The operating-state view is unavailable because
  heater-state columns were omitted from the prepared asset truth.
- V550 now screens the predeclared `B=12` and `B=16` hardware breakpoints with
  the same quality-coupled truth. No final-test feedback, policy training, or
  target-label changes are used.

### 2026-09-09 V550 Short Finding
- B12 and B16 are materially stronger than B20 in short windows, but their
  seed spread is substantial and the operating-state partition is still absent.
  B12 is the only budget promoted to a full geometry audit; B16 is closed as a
  diagnostic breakpoint rather than a policy candidate.

### 2026-09-09 V550 Full Finding and V551 Launch
- B12 full coverage reduced the apparent short-screen opportunity to
  `0.7627%--1.6276%`, so the two-channel quality route is closed without
  transfer or PPO. Empty 1%/5% intersections alone are insufficient when the
  absolute downstream margin is this small.
- The V613 comparison identified a clean, previously implemented difference:
  its relation degrades all exposed channels under the declared risk state,
  whereas V547 only degraded GMX500 and Parsivel. V551 now tests that relation
  on the current seeds at B12 using the event-column branch, which is valid for
  the available resource trace and avoids fabricating a dew-point field.

### 2026-09-09 V551 Finding
- Applying the all-channel risk-state quality relation increased short-window
  separation for seeds `7241--7243`, but seed7244 fell to `0.087%`. The
  cross-seed instability means this is not a defensible geometry gate. No
  online transfer or PPO is authorized from V551.
- The discrepancy with V613 is attributable to scene/resource differences,
  not an established PD-PPO effect. V613 remains historical diagnostic context
  only and cannot be reused as current mainline evidence.
## 2026-09-09 V553 architecture finding and action-space correction

- V553's `candidate_count=16` at `B=1.15` was not a scene result. The
  candidate builder called `PowerProjector.project_mask()` during action-space
  construction, so infeasible requests were rewritten and duplicate projected
  masks were removed before the runtime state was known.
- The flexible-subset path now enumerates the declared subset space without
  power projection. Runtime `feasible_candidate_mask()` remains responsible for
  power, startup, coverage, dynamic-resource, and dwell feasibility.
- The legacy `build_projected_candidate_masks()` helper is retained for
  historical pipelines. Current custom-PPO preparation uses the new arbitrary
  subset helper; the geometry diagnostic uses the same non-projecting semantics.
- The old all-true fallback when no candidate was executable was removed. Such
  a state now raises an explicit consistency error instead of bypassing hard
  feasibility masking.
- For the current six-channel flexible configuration this produces 64 actions.
  A declared mandatory backbone would reduce the surface to 32 optional
  subsets; this distinction is now explicit rather than being an accidental
  consequence of budget projection.
## 2026-09-09 V554 protocol

- V554 reuses the V552 channel-specific quality truth and the V553 chronological
  partition. The only intended action-space change is retaining all subsets
  compatible with the mandatory `cr1000xe_backbone`; this is a controlled
  implementation correction, not a new scene fit.
- Any V554 geometry result must be compared with the pre-fix V553 numbers only
  as an implementation audit. It cannot be described as an improvement until
  the full candidate surface and runtime feasibility accounting are verified.
# 2026-09-09 V556 physics findings

- Existing V541 resource traces provide a genuine state-dependent effective
  power signal for GMX500 and Parsivel, but not for the radiometer, SI-111 or
  FC4 under the current evidence manifest.
- The low-budget screen separates geometry from downstream value. At B=2.15 W
  the laser is feasible during heater-off periods and infeasible during its
  high-load periods, while the feasible subset count remains nontrivial. This
  is a suitable next geometry point, but it does not justify PPO by itself.
- A fixed subset family without laser remains feasible in all rows at B=2.15;
  therefore the next audit must test whether forecast loss changes across the
  resource regimes and whether a universally near-optimal static subset still
  exists. The route must close if that forecast gate fails.

- V557's first geometry attempt exposed a reproducibility bug in the audit
  fallback: V541 training metadata used the 12-dimensional state implied by its
  uncertainty vector, but did not serialize `state_columns`; the audit assumed
  the newer 15-dimensional latent-augmented default. The correction is limited
  to metadata reconstruction and does not alter truth, oracle, or policy.
## 2026-09-09 V561 asset completion and V562 launch
- V561 asset preparation completed for seeds `7241--7244`; each seed retains
  32 declared subset candidates and a frozen TCN evaluator.
- The first V561 asset attempt failed only because the reusable launcher used
  the V559 truth suffix. Parameterizing the suffix and rerunning produced the
  valid assets; the failed attempt is not scientific evidence.
- V562 now audits downstream forecast geometry under the event-specific quality
  relation. No transfer or PPO result is available yet.

## 2026-09-09 V562 geometry finding
- The event-specific relation produces some subtype-dependent winners, but the
  downstream operating geometry is weak: gaps versus the best fixed subset are
  `0.000351--0.002146` across seeds `7241--7244`, below the predeclared
  `0.01` material gate.
- Condition-level gaps are also inconsistent (`0--0.008998`) and do not imply
  an online opportunity. The route therefore fails before observability and
  learner stages; no PPO result should be generated from it.
- This closes the heater-plus-event relation at B=`2.15`. A future route must
  change the physical complementarity/resource mapping, not add policy-side
  modules to this scene.

## 2026-09-09 physical-budget calibration finding
- The active resource manifest distinguishes an absolute controller budget of
  `1200 W`, a physical development effective budget of `55 W`, and the older
  normalized screening budget `2.15`. These values are not interchangeable.
- At the physical `55 W` scale, the current six-channel optional subset family
  is effectively always feasible in the available resource traces; at `2.15`,
  the low-cost non-laser subset dominates the forecast frontier. Neither scale
  currently supplies a defensible binding, state-dependent arbitrary-subset
  problem without an independently documented power-system budget.
- Do not start V563 assets or PPO. The next action is a budget-manifest audit or
  a new physically documented resource mechanism, selected before looking at
  policy results.
- The remote four-seed resource screen confirms this is not seed noise: mean
  feasible-subset counts are `12.35--12.39` at B=`2.15`, `18.82--18.86` at
  B=`12`, `26.59--26.70` at B=`20`, and `32.00` at B=`55`.

## 2026-09-09 V563/V564 persistent target finding
- Persistent transport/particle/thermal target modes created balanced mode
  support and changed several condition-level subset winners, so the route did
  alter forecast geometry without changing event labels or resource traces.
- The effect was not materially executable. Operating-state opportunity gaps
  were `0.008591`, `0.002358`, `0.005213`, and `0.000532` for seeds
  `7241--7244`, all below the `0.01` gate. Candidate 023 remained the best
  static shortcut in all four seeds and near-optimal across most states.
- The route is closed before online transfer and PPO. Increasing target gain
  solely to force a larger gap would be post hoc scene fitting; the next route
  must be justified by an independent physical/resource specification.

## 2026-09-09 V565/V566 frequency-cost finding
- Fixed sampling-frequency multipliers changed the feasible resource geometry
  without exposing frequency as a policy action. At B=`2.15`, only 16/32
  subsets were feasible and no subset contained more than two optional
  channels.
- This did not create downstream adaptive value. Operating-state opportunity
  gaps were `0.005465`, `0.000106`, `0.003823`, and `0.000000` for seeds
  `7241--7244`.
- The route is closed before online transfer and PPO. The remaining blocker is
  target-observation complementarity, not merely the count of feasible masks.

## 2026-09-09 V567 exposure-state finding
- The existing nowcast-coupled Stage-B generator has valid time alignment,
  bounded quality traces, and positive future correlation for particle/flux
  states, but its state support is unsuitable for scheduling experiments.
- Across seeds `7241--7244`, exposure >=0.5 occupied `0.9908--0.9925` of rows;
  GMX heater occupancy was `0.9825--0.9898`, with very long runs.
- This is a truth-only failure. No assets or policies were trained. Changing
  thresholds after seeing the occupancy would violate the frozen scene gate.

## 2026-09-09 V568--V574 geometry findings
- V568 quantile exposure produced non-degenerate truth support, but its two
  heater states yielded operating gaps `0.000834`, `0.014727`, `0.000189`, and
  `0.000000`; a 1% static intersection remained in 3/4 seeds.
- V572 applied the declared GMX500/Parsivel heater-quality restoration. Gaps
  were `0.000666`, `0.014552`, `0.000195`, and `0.000000`; the conclusion did
  not change.
- V574 used the documented absolute temperature/dew-point/surface-temperature
  controller. Its full test partition contained `00/01/11`, but the frozen
  evaluation windows all landed in `11`, giving zero operating opportunity.
- The geometry audit had two implementation gaps, both corrected before using
  any result: asset manifests now persist dynamic-resource mappings, and the
  geometry path now merges all manifest-referenced resource columns into truth.
- No V568--V574 route passes the predeclared all-seed materiality gate. No
  online transfer or PPO result may be reported from these routes.

## 2026-09-10 V576 independent-nowcast geometry finding

- The independent-nowcast heater controller created all four heater states in
  the full test partitions, and the frozen starts covered multiple states, but
  this did not create executable forecast opportunity.
- Operating gaps for seeds `7241--7244` were `0.000000`, `0.008172`,
  `0.000628`, and `0.000467`. The persistent low-cost non-laser subsets stayed
  feasible and best across the operating states.
- Seed `7242` illustrates the distinction: its condition-level gap was
  `0.129460`, while its operating gap was only `0.008172`. The resource
  mechanism therefore changes feasibility without changing the downstream
  static frontier enough to justify online transfer.
- V576 is closed before PPO. V577 is a predeclared budget phase screen, not a
  final budget selection experiment; it must be judged by whether the
  executable subset frontier changes across the same frozen windows.

## 2026-09-10 V577 resource phase-screen finding

- The budget screen covered `0.55--2.15` using ten predeclared effective
  budgets, the same four frozen windows per seed, and startup budget `2.60`.
- No budget produced an all-seed operating gap of at least `0.01`. The largest
  observed per-seed gap was `0.007416`; the gap vectors were identical across
  several budget intervals, showing that the changing feasible-set cardinality
  did not change the task-level frontier.
- At least three seeds retained a 1% near-optimal static intersection at every
  budget. This is evidence against continuing heater-only budget tuning.
- V577 is closed before online transfer and PPO. A new route must change the
  physically supported subset-value complementarity or resource accounting;
  changing only the budget is not sufficient.

## 2026-09-10 V578 persistent-target cross geometry finding

- Reusing the persistent target truth with the frozen independent-nowcast
  heater trace produced four resource-duty states in every seed and different
  state-wise best subsets. No seed retained a 1% near-optimal static
  intersection.
- The executable operating gaps were `0.054767`, `0.003613`, `0.001656`, and
  `0.005529` for seeds `7241--7244`. Thus the route has real state-dependent
  geometry, but the effect is not stable enough across the predeclared seeds
  for online transfer or PPO.
- V578 is closed before learner training. The result supports a narrower
  diagnosis: resource-state complementarity is necessary but insufficient;
  all-seed downstream operating value must also be material.

## 2026-09-10 V579 persistent-target budget finding

- The V579 screen found a useful separation: budgets `0.85--1.85` removed the
  1% static intersection in all four seeds and changed the state-wise best
  candidate in all four, but the largest operating gap was only `0.004778`.
- B=`2.15` increased the seed7241 gap to `0.054767`, but the other three
  seeds remained below `0.006`. No budget provided a stable four-seed PPO gate.
- V579 is closed before learner training. The current physics/resource route
  should not be extended by more budget tuning; a new route must alter the
  documented sensor-quality complementarity or add an independently supported
state-dependent load for currently dominant low-cost channels.

## V581 predeclared bounded follow-up (2026-09-10)

V581 tests only the identified V580 failure mechanism: absolute wind/humidity
and temperature/sunlight clipping created an imbalanced specialist state under
the source nowcast distribution. Training-prefix quantile scaling is an
ordinary calibration operation and is fitted before the policy-training
partition. It does not use event labels, target values, test windows, or
latent states. No PPO is allowed unless V581 passes the existing support,
persistence, and four-seed downstream geometry gates.

## V580 predeclared hypothesis (2026-09-10)

V576--V579 changed resource occupancy without reliably changing the
forecast-optimal executable subset.  V580 tests the next allowed Stage-B
mechanism: persistent, specialist-separated transport/thermal states derived
from noisy nowcasts jointly modulate future target dynamics and observation
quality, while an independent nowcast controller supplies heater/resource
loads.  The route is valid only if state support is non-degenerate, runs are
long enough for the six-step dwell, all specialist states occur in the fixed
development windows, and the subsequent complete-subset operating geometry
passes the existing all-seed materiality gate.  The route is closed before
PPO if any of those gates fail.

## V580 truth-only finding (2026-09-10)

The resource mechanism had adequate support, but the specialist target/quality
mechanism did not.  Across seeds `7177--7180`, the particle state occupied
`0.893--0.949` of the test partition and the flux state occupied `0--0.005`.
The failure is attributed to absolute driver clipping under the source
nowcast distribution, not to PPO or subset geometry.  V580 is closed.  The
only remaining bounded Stage-B adjustment is training-prefix quantile
normalization of the wind, humidity, coldness, and solar drivers before their
persistent specialist combinations are formed.

## V581 truth-only finding (2026-09-10)

The q10/q90 transform itself is valid, but flux support disappeared from the
frozen test suffix for every seed.  The source nowcast sequence therefore
does not support the high-wind/high-humidity branch used by the V580/V581
flux relation.  V582 is the final bounded adjustment: FC4 demand is modeled
as wind-dominant with humidity as a secondary modifier.  If the fixed test
partition still lacks a material flux state or downstream geometry, the
Stage-B route is closed rather than tuned further.

## V582 truth-only finding (2026-09-10)

The wind-dominant flux relation did not recover flux support in the frozen
test suffix.  This rules out further coefficient tuning on the existing four
truth files.  V583 therefore changes only the predeclared data-generation
seeds, not the physical relation, resource mapping, evaluation starts, or
policy observables.  The fresh-seed gate is the final truth-generation check
before the route is either promoted to subset geometry or closed.

## V583 truth-only finding (2026-09-10)

Fresh generation seeds did not solve the problem under `rho=0.992`: raw
transport-driver support existed in the test suffix, but the filtered state
did not cross the activation threshold.  This isolates persistence response
time as the remaining Stage-B hypothesis.  V584 is the final bounded check
using `rho=0.95`; no other relation or selection rule changes.

## V584 truth-only finding (2026-09-10)

The response-time change increased full-test flux support but did not place a
material flux state in the fixed evaluation windows.  This closes the current
Stage-B route.  The evidence does not justify starting PPO: the evaluator
would mostly see particle/thermal windows, and changing starts after this
observation would invalidate the frozen protocol.  A future scene redesign
must change the base meteorological truth or predeclare a new evaluation
protocol before any learner experiment.

## V585 protocol decision (2026-09-10)

The fixed legacy starts under-cover the newly generated scene family, but
replacing them after reading V584 would be invalid.  V585 locks an eight-
window protocol from an index-only RNG before generating fresh truth seeds.
It is the final attempt to separate protocol coverage from scene support
while preserving chronological and online-observability boundaries.

## V585 truth-only finding (2026-09-10)

The protocol audit separates window under-coverage from scene support.  Even
with eight index-only starts and fresh seeds, the flux state was absent from
most windows and from every window of one seed.  The current source generator
therefore cannot support a fair all-seed arbitrary-subset scheduling test.
The Stage-B route is closed.  A future attempt must change the base
meteorological/state generator and freeze a new protocol before generation;
the existing heater, threshold, persistence, and window variants must not be
reused as further tuning axes.

## V586 structural correction (2026-09-10)

The prior Stage-B route incorrectly treated flux demand as zero outside a
hysteresis-active interval.  V586 preserves the causal nowcast state and uses
the continuous filtered transport factor directly.  This is the last
truth-only correction before subset geometry; it does not add labels,
post-test selection, or policy-side assistance.

## V586 asset-stage decision (2026-09-10)

Because Stage B is a continuous-factor scene, binary flux-dominance coverage
is not used as the final truth gate.  V586 proceeds to frozen asset and
complete-subset geometry evaluation, where the actual downstream forecast
loss determines whether the continuous factor is material.  No learner is
trained before that geometry gate.

## V586 asset schema correction (2026-09-10)

The first asset-only execution failed before fitting because the transformed
truth did not include the explicit quality metadata column required for the
mandatory CR1000Xe backbone.  This was an input-schema defect, not a scene or
algorithm result.  The builder now emits a constant quality value of `1.0`
for that mandatory backbone.  The corrected asset preparation was relaunched
under a fresh output root; the previous failed directory is not used as
evidence.

The first relaunch exposed a stale-input issue: the asset runner still read
the original V586 truth CSVs generated before the backbone-quality field was
added.  It stopped before fitting at the same schema check, so it is not a
science result.  Truth and resource traces must be regenerated under a fresh
root and passed explicitly to the asset runner.

The regenerated V586 truth/resource root passed the truth gate for all four
seeds.  It contains the required backbone-quality metadata and preserves the
predeclared protocol.  Test-partition flux, particle, and thermal occupancy
ranges are `0.0747--0.1153`, `0.5261--0.6406`, and `0.5756--0.6028`,
respectively; specialist future-target correlations are positive throughout.
This supports proceeding to frozen assets, but does not yet establish
forecast geometry or learner value.

## V586 budget-screen audit correction (2026-09-10)

The completed V586 B=1.85/2.35 screen cannot be treated as a clean budget
comparison. The geometry runner used one override for both the normalized
`PowerProjector` action-cost budget and the dynamic effective-resource budget,
although the sensor YAML and heater trace are on different declared scales.
The dynamic resource guard also had `fixed_power_w=0.0`, so the mandatory
CR1000Xe backbone was not included in that guard even though it was included
in the static projector. These are implementation/protocol defects in the
budget screen, not evidence that the scene fails. A corrected audit must
separate the two budgets and include the backbone fixed load before deciding
whether V586 geometry passes.

## V587 consistent-resource geometry launch (2026-09-10)

The geometry audit was extended with explicit independent overrides for the
normalized action-cost budget, dynamic effective-resource budget, and fixed
dynamic load. V587 reuses the locked V586 truth, frozen evaluators, and eight
starts, disables the normalized action-cost bottleneck for geometry (`1000`),
and evaluates the physical effective-resource budget at `2.15` including the
`0.4104` backbone load. This is a diagnostic correction; no PPO training is
allowed until its all-seed operating geometry is reviewed.

## V587 result (2026-09-10)

The corrected B=`2.15` audit produced operating gaps `0.011506`, `0.016133`,
`0.006384`, and `0.008015` for seeds `7401--7404`. The 1% near-optimal static
intersection was empty for all four seeds, but the all-seed materiality gate
still failed. The corrected resource semantics improve the geometry and expose
the remaining seed-dependent weakness; they do not justify online transfer or
PPO at this budget.

## V588 corrected budget screen (2026-09-10)

The predeclared B=`1.85` and B=`2.35` comparison is being rerun with static
projector budgets `1000/1000`, independent dynamic budgets at the tested value,
and backbone fixed load `0.4104`. All eight seed/budget jobs use the same V586
truth, evaluators, and starts. Results remain blocked from learner use until
both budget points are fully audited.

## V588 result and V589 rationale (2026-09-10)

Under corrected semantics, B=`1.85` produced operating gaps
`0.011506`, `0.016133`, `0.006384`, `0.008015`, and B=`2.35` produced
`0.010751`, `0.015123`, `0.005973`, `0.007792` for seeds `7401--7404`.
Both budgets had empty 1% near-optimal static intersections but failed the
all-seed materiality gate. A resource-only phase screen showed that B=`2.15`
supports 20 candidate masks and B=`2.35` supports 28, while B=`3.0` is the
first tested point at which all 32 masks have nonzero support for every seed.
This motivates one bounded B=`3.0` diagnostic geometry run as a resource-
frontier breakpoint check, not as final-budget selection.

## V589 breakpoint geometry launch (2026-09-10)

The four-seed B=`3.0` audit is running with the corrected independent budget
semantics. B=`3.0` was selected only because the resource-only phase screen
identified it as the first tested point with support for all 32 masks; it is
not being treated as a confirmatory result or as evidence selected from PPO
performance.

The first V590 geometry launch stopped before science output because the new
asset copier stored `oracle_path` relative to each asset directory while the
geometry audit resolved it only from the repository working directory. The
audit now resolves relative oracle paths against `run_dir`; the V590 geometry
run was relaunched with the same assets and protocol.

## V590 window-count audit correction (2026-09-10)

The first pooled-controller geometry run also inherited `--max-rollouts 4`.
Because the audit truncates explicit starts to `max_rollouts`, it used only the
first four of the eight locked starts. Those outputs are invalid as eight-
window evidence. The runner now uses `--max-rollouts 8`; V590 was relaunched
against the same pooled assets and all eight starts.

## V590 corrected eight-window geometry result (2026-09-10)

The valid V590 B=`3.0` geometry audit used all eight starts and passed the
materiality gate for every seed. Operating opportunity gaps were
`0.062288`, `0.056644`, `0.046725`, and `0.045861` for seeds `7401--7404`.
The 1% near-optimal static intersection was empty for all four seeds and all
32 candidate masks were declared with nonzero support. This clears geometry,
but not chronological online transfer or PPO.

## V591 transfer audit launch (2026-09-10)

The transfer probe uses only heater-state flags and frozen fixed-mask losses:
state-conditioned candidate rankings are fitted on eight fixed training
windows and evaluated on the eight locked test windows. The mandatory
backbone cost is included in feasibility. This is a non-learning diagnostic;
PPO remains blocked until transfer improvement is positive for all seeds.

## V590 pooled-controller scene correction (2026-09-10)

V589 still had seed7403 concentrated in one heater state. The controller was
therefore corrected at the scene-definition level: temperature and wind
quantiles are now fitted once on the pooled training prefixes of seeds
`7401--7404`, then the same observable hysteresis controller is applied to all
seeds. The resulting state counts are balanced across seeds (`00` about
20--21k, `01` about 12--13k, `10` about 36--37k, `11` about 19--20k over
90k steps). This changes no target, event label, or scheduler input and is a
deployment-consistent calibration correction. New frozen assets were prepared
at B=`3.0`; pooled-controller geometry is running before any learner work.

## V591--V594 online transfer failure (2026-09-10)

The corrected chronological transfer audit used eight training starts and
eight locked test starts, with the mandatory CR1000Xe load included in every
dynamic feasibility check. V591's heater-state-only lookup improved over the
test static schedule in only seeds `7403` and `7404`. V592 added all declared
online context and resource features to train-only candidate regressors;
V593 used same-time backbone-relative candidate loss; V594 added six-step
history means. Their transfer-minus-static values remained positive for seeds
`7401` and `7402` in every variant. Thus the failure is not explained by a
missing single context feature or by absolute loss drift alone.

This is a useful negative gate result: the frozen assets expose a strong
operating geometry gap, but the candidate ranking is not chronologically
recoverable from the available online state. Starting PPO here would confound
learner behavior with a failed observability/transfer protocol, so the scene
is closed before learner training.

## V595 pooled-context transfer closure (2026-09-10)

One shared regressor was fitted on all four development training partitions,
while each seed kept its own training-selected static comparator. The
transfer-minus-static losses were `+0.085574`, `+0.077100`, `-0.026971`, and
`-0.017979`. Pooling training data did not remove the two-seed failure. This
confirms that the current issue is not merely insufficient per-seed sample
size; the B=`3.0` heater scene has a material feasible-frontier gap but no
stable online value-transfer evidence.
## V596 full observation boundary (2026-09-10)

The next diagnostic must use the actual pre-action state exposed to the policy,
not only manually selected resource/context columns. V596 uses the environment
rollout's `agent_observations` but reconstructs the environment with
`include_event_flag_in_state=False`. This preserves the online-observation
test while preventing exact simulator event labels from leaking into the
transfer model. The output records the exclusion explicitly and remains a
non-learning diagnostic; PPO remains gated on its all-seed result.
## V598 resource phase rationale (2026-09-10)

The pooled heater trace has discrete support breakpoints independent of PPO
performance: 16 supported masks at B=`1.2--1.5`, 18 at B=`1.6--1.8`, 24 at
B=`2.0--2.15`, 30 at B=`2.35--2.6`, and 32 at B=`3.0`. This motivates a
bounded B=`1.4`/`1.6` geometry screen as an objective-level resource test. It
does not alter the scene after learner results and will not be promoted to PPO
without the predeclared all-seed geometry and online-transfer gates.
## V598 B=1.4 geometry result (2026-09-10)

The lower resource phase restores objective-level subset competition without
changing target generation or using final-test feedback. The four operating
gaps are all positive and exceed the configured `0.01` materiality threshold;
the 1% near-optimal static intersection is empty for every seed, and the
three-specialist union is infeasible. This is the first current heater route
that clears geometry below the B=`3.0` support breakpoint. It still requires
online transfer evidence.

The attempted B=`1.6` sub-batch has no valid output and is excluded from all
decisions; no result is inferred from its empty logs.
## V600 B=1.4 transfer closure (2026-09-10)

Lowering the physical budget restored a large operating-condition opportunity
gap, but the exact deployment observation could not recover candidate value
chronologically: only seed `7403` beat its training-selected static schedule.
The negative result is not caused by a missing event flag because that flag was
excluded by construction, and it is not evidence that PPO is weak. The current
heater controller/resource trace is closed as geometry-positive but
observability-negative. A new scene must make the effective power and sensing
quality depend on causal nowcast/alert variables that remain informative over
the forecast horizon.
## V601 alert-coupled resource scene (2026-09-10)

The previous pooled heater controller was weakly aligned with the available
alert signals, so its resource geometry did not transfer chronologically. V601
uses a documented observable controller: met-core heating follows the thermal
alert and laser heating follows the particle alert, both at threshold `0.5`;
unchanged channels retain their source trace. This is a controlled scene
variant with `truth_targets_changed=false`, `event_labels_changed=false`, and
no exact event flag in the scheduler observation. It must pass the same
geometry and online-transfer gates before any learner experiment.
## V604 alert-coupled transfer closure (2026-09-10)

All four seeds lost to the train-selected static comparator under the
alert-coupled resource trace. This isolates a causal inconsistency: effective
power was made alert-driven, but sensor-quality columns and the frozen oracle
were inherited from the previous heater scene. The route is closed before PPO.
Any next scene must regenerate the quality relation and refit the forecaster as
one frozen asset bundle; reusing the old oracle would be invalid.

## V605 planned repair

The next candidate must not reuse V601/V604's frozen oracle. V605 copies the
alert-coupled asset bundle, derives each specialist quality trace from the same
alert inputs that drive effective resource load, and refits the TCN on the
oracle partition. This preserves target and event labels and keeps the exact
event flag out of the scheduler observation while removing the previous causal
inconsistency. V605 is still a geometry/transfer prerequisite, not PPO
evidence.

The first corrected-path smoke also found stale resource columns in the oracle
training truth. Replay removes and remerges these columns, so fitting on the
copied table produced an input-width mismatch. The asset builder now constructs
the same resource-merged truth view used by replay before refitting each TCN.

V606's repaired seed7401 geometry is negative for the intended purpose. The
alert-linked quality relation produces different best candidates in some
conditions, but the weighted operating gap is only `0.0001358` and the 1%
near-optimal static intersection contains eight masks. This is not sufficient
evidence of adaptive forecast opportunity; the route must close rather than
expand to four seeds or PPO.

V613 artifacts from an earlier branch are not promoted: their geometry passes
only some seeds and their manifests include a privileged event flag in at least
one asset family. V607 therefore reuses only the repaired, label-free V605
bundle and changes the budget as a predeclared resource-competition screen.

V606 initially failed before loading an oracle because metadata paths were
double-prefixed during bundle replay. The failure is procedural only. The
correct contract is bundle-local filenames for truth, resource trace, and
oracle paths; the V605 builder and existing metadata are being repaired before
the geometry decision.

V607 confirms that the V606 failure is not only caused by the B=`2.15` static
specialist bundle. At B=`1.4`, the feasible family is more restricted but the
forecast-loss frontier remains nearly static. A useful next scene must make
the forecast target/observation relation condition-specific at the channel or
variable level; resource-only and scalar quality-only coupling are closed.
## 2026-09-10 V608 causal-specialist findings

The heater-only route is insufficient: V606/V607 produced dynamic resource
frontiers but did not establish forecast-value transfer. V527 also failed its
observable target relation, especially for particle and transport.

V608 fixes the causal construction rather than increasing a post hoc target
amplitude. A six-step-ahead noisy mode proxy is generated first. At time `t`,
the proxy from `t-6` drives the current target increment, while specialist
quality is coupled to the same physical mode. Against the source truth, the
event-window correlations for all four seeds were:

| increment | active-window Spearman range |
|---|---:|
| transport / mass flux | 0.853--0.854 |
| particle / velocity | 0.839--0.845 |
| thermal / surface temperature | 0.853--0.857 |

Raw target correlations are not used for this gate because the source weather
process is a confounder; the audit uses V608 minus V525 target increments.

V608 resource occupancy at `B=2.15` has 32 candidate masks, 8--20 feasible
masks per step, and 8 always-feasible masks in seed 7177. This is useful
state-dependent resource evidence but not sufficient evidence for adaptive
forecast value. The refitted TCN geometry remains the decisive gate.

The first completed V608 frozen-forecaster geometry audit (seed 7177) gives a
condition gap of `0.048444` and no 1% condition-level intersection. However,
the operating-heater gap is only `0.002181`, with seven masks in the 1%
operating near-optimal intersection. Thus the new target relation works at the
condition level, but the existing V525 heater trace is not causally aligned
with that relation strongly enough for a deployable adaptive claim. The other
three seed audits remain pending in `v608_geometry2`; no online transfer or PPO
is authorized.

V608 is now closed. Across all four seeds, condition-level gaps were
`0.048444`, `0.024835`, `0.000469`, and `0.000016`, while operating-heater
gaps were `0.002181`, `0`, `0.000053`, and `0.000089`. Every seed retained the
same seven-mask operating 1% intersection. The causal target relation alone is
therefore insufficient for a deployable adaptive claim when the resource state
is independent of it.

V609 couples the documented GMX500 and Parsivel heater loads to the same causal
mode. Transport mode activates both exposed loads; particle mode activates
Parsivel; the other three channels retain the manifest's fixed-load definitions.
This is evaluated at the manifest's 55 W development effective budget, not a
post-hoc budget sweep. The resource-only screen gives 24--32 feasible masks and
24 always-feasible masks in each seed. This route is still only a candidate
until refit-forecaster operating geometry passes all four seeds.

## 2026-09-10 V610 resource geometry

The existing frequency-cost manifest provides a fixed effective acquisition
cost for each channel while preserving the minimum scheduling epoch. V610 adds
the V609 heater increments after conversion from the declared 55 W physical
reference into the existing `2.15` effective-unit scale. The resulting
resource trace has 11--16 feasible masks per step, 11 masks feasible over the
full trace, and consecutive-frontier Jaccard as low as `0.6875`. This is a
stronger resource-only screen than V609, but it does not establish downstream
forecast value. Frozen TCN assets are being rebuilt before geometry is
audited.

V610 geometry closed for seeds `7177--7180`. Condition gaps were `0`,
`0.001559`, `0.003910`, and `0.002090`; operating gaps were `0.0000007`,
`0`, `0`, and `0.0000198`. The operating 1% intersection contained seven
candidates in every seed. The effective resource frontier is dynamic, but the
current ordinary target aggregation still selects a near-universal static
subset after the deployable resource partition.

The next diagnostic will use a predeclared group-balanced target objective:
particle, flux, and thermal specialist groups receive equal total weight,
while the mandatory-backbone targets remain unchanged. This tests whether the
failure is caused by target-group domination, not by tuning the policy or
resource trace. The result remains geometry-only until the all-seed operating
gate passes.

The four V610 frozen assets completed with 32 candidates each. A four-seed,
four-start, 256-step geometry audit is running remotely in tmux
`v610_geometry`; no online transfer or PPO result exists yet.

## 2026-09-10 V609 mode-coupled geometry closure

The refitted V609 assets were audited remotely at `B=55 W` for seeds
`7177--7180`, with 256-step held-out rollouts and all 32 declared masks. The
condition-level gaps were `0.013069`, `0.002938`, `0.004986`, and `0.004636`.
After grouping by the resource states actually available to a deployable
policy, the gaps fell to `0.0000057`, `0.0000378`, `0`, and `0.0001899`.
The operating 1% near-optimal intersections each contained eight masks, and
the specialist union was infeasible for every seed.

This is a geometry failure, not a learner failure. At `55 W`, the physical
heater loads exclude only part of the large subsets during heated states while
leaving the low-cost subset family feasible across the trace. The causal target
relation is present at condition level, but it is not converted into a
material, state-dependent executable forecast frontier. Online transfer and
PPO remain blocked. Any successor must predeclare a physically justified
controller budget/load model that makes the operating frontier binding; it must
not reuse the condition-only gap as evidence.

## 2026-09-10 V611 group-balanced geometry closure

The group-balanced frozen assets were evaluated with the unchanged V610
resource trace. Operating gaps for seeds `7177--7180` were `0`, `0`, `0`, and
`0.000124`; each seed retained seven candidates in the operating 1%
near-optimal intersection. A secondary audit grouped losses by the noisy
forecast-mode proxy available to the scheduler and produced gaps of
`+0.000798`, `+0.000150`, `-0.000250`, and `+0.000127`. Only seed 7180 had a
different proxy-conditioned best candidate. V611 is closed before online
transfer and PPO. The next scene must strengthen the mode-specific
specialist-quality relation; more budget or weight tuning is not justified.

## 2026-09-10 V612 exclusive-quality geometry closure

The stronger mode-specific quality relation was evaluated with the unchanged
V610 resource trace and ordinary target aggregation. Operating gaps for seeds
`7177--7180` were `0.000001`, `0`, `0`, and `0.000020`; all four seeds kept
seven candidates in the 1% operating near-optimal intersection. This rules out
the current quality contrast as the missing bridge. The route is closed before
online transfer and PPO. Repeatedly changing quality floors, target weights,
or budgets after this result would be post-hoc scene fitting rather than a
defensible physical calibration.

## 2026-09-10 SOC provenance check and V613 result

- `src/v2/env.py` contains a reusable energy account with capacity, reserve,
  harvest, external load, and SOC observations. Its presence is execution
  capability, not evidence that the physical entity has a battery or energy
  harvesting subsystem.
- `scripts/127_audit_entity_energy_trajectory.py` audits fixed schedules with
  `capacity_wh=8640`, fixed auxiliary load `6.01 W`, and a hysteretic external
  profile whose heater branch is `600 W`. It has 25 hourly rows and marks the
  72 h/168 h requests as truncated; it cannot support a cumulative-energy
  deployment claim.
- V613 remote logs show operating opportunity gaps of `0.002415`, `0.014562`,
  `0.031601`, and `0.033118` for seeds 7181--7184. Online top-1 transfer was
  `0.2456`, `0.4418`, `0.1605`, and `0.2595`. Geometry is not uniformly
  above the predeclared `0.01` threshold, so this is diagnostic only.
- Decision: close the SOC and V613 routes. Continue with a predeclared,
  persistent deployable operating-state design; do not tune thresholds or
  resource budgets in response to these failed gates.

## 2026-09-10 V614 finding: empirical cold availability plus physical-watt trace

- Independent hardware-test evidence supports a temperature-dependent quality
  relation for the Modbus weather channel: reported invalid-wind rates increase
  from `4.8%` at `-30..-10 C` to `98.1%` below `-46 C`. This is used only as a
  calibration reference; it is not presented as Antarctic field telemetry.
- V614 uses the nowcast air-temperature column as the sole online driver. The
  exact test labels and latent event columns are not policy inputs.
- The original V614 launch mixed physical-watt resource traces with the legacy
  normalized budget `2.15`, making every heated all-optional row infeasible.
  That output is discarded. A separate physical-watt manifest with a declared
  `55 W` development budget was added, and the four-seed truth/resource run was
  regenerated successfully.
- The corrected resource trace has joint heater occupancy in all four states,
  total power `2.646--57.146 W`, and all-optional feasibility fraction
  `0.7759--0.7824`. This is a valid resource/quality screen, not yet evidence
  of downstream forecast-value crossover.
- Next action: prepare frozen forecasters from the corrected V614 truth and
  physical-watt traces, then run the predeclared 32-subset operating geometry
  audit. Online transfer and PPO remain blocked until the geometry gate passes.

## 2026-09-10 V615 geometry closure

- All four corrected V614 frozen assets completed successfully. The geometry
  audit used 32 candidates, eight held-out starts, normalized interface budget
  `2.15`, physical dynamic-resource budget `55 W`, and the same frozen TCN per
  seed.
- Operating opportunity gaps were `0.004463`, `0.000824`, `0.001400`, and `0`
  for seeds `7401--7404`; the all-seed `>=0.01` gate failed.
- The best static candidate was `candidate_005` in all seeds:
  `met_station_core + surface_temp_ir + cr1000xe_backbone`. The operating 1%
  intersection was empty in three seeds and contained only that candidate in
  seed7404.
- Condition-wise best candidates changed in some heater states, but the
  frequency-weighted operating loss advantage was not material. The route is
  closed before online transfer and PPO.
- The first geometry attempt had only a path-resolution failure, not a
  scientific failure. `scripts/109_v32_audit_subset_forecast_geometry.py` now
  resolves relative oracle paths against both the run directory and project
  root; the corrected audit completed all four seeds.

## 2026-09-10 V616 rationale

The V615 asset configuration used `sensor_quality_availability_floor=0.2`.
The environment defines this as a lower bound on observation probability, so
it replaced the measured `0.019`--`1.0` availability relation with
`0.2`--`1.0`. V616 sets the floor to `0.0`, the direct interpretation of the
independent failure-rate calibration. This is a semantic fidelity correction,
not a result-selected hyperparameter search. The same four seeds, resource
manifest, 55 W budget, normalized interface budget, partitions, starts, and
32-mask family are retained.

## 2026-09-10 V616 geometry finding

V616 completed the planned subset-level forecast audit after correcting the
availability floor. The physical trace is nontrivial: all four heater states
occur, total load varies, and the condition-wise best subset is not identical
in every state. This is insufficient for the main claim because the dominant
operating state is still `heater_10000`, and `candidate_005` remains the best
fixed subset after the actual time distribution is applied.

The exact operating gaps are `0.0019531069`, `0.0010018449`,
`0.0010458259`, and `0` for seeds `7401--7404`. The 1% near-optimal static
intersection is empty in the first three seeds and contains `candidate_005`
in seed 7404. Equal weighting of observed heater states is diagnostic only:
the corresponding gaps of the actual best static subset are `0.02641`,
`0.00937`, `0.01601`, and `0` and therefore do not establish an all-seed
adaptive opportunity under a declared deployment distribution.

This closes the empirical-cold heater route before online transfer and PPO.
The next admissible route is a truth-only, predeclared occupancy/persistence
screen: define the desired temperature/icing state support from the hardware
controller and scenario protocol before fitting assets, then require the
resulting state distribution to produce a material complete-subset forecast
gap under the actual weights. No threshold, start, or aggregation may be
chosen from the V616 losses.

## 2026-09-10 V617 full-test geometry finding

The complete final-partition audit confirms that V616's result is not caused
by sparse evaluation windows. A fixed 27-window grid over `[76500,90000)`
produced operating gaps of `0.0028669913`, `0`, `0`, and `0.0017416359` for
seeds `7401--7404`. Only two seeds had any positive gap, and the mean was
`0.0011521568`, far below the materiality threshold.

`candidate_005` remained the best fixed subset for every seed. The operating
state winners changed only in seeds 7401 and 7404; seeds 7402 and 7403 kept
the same subset across all heater states. This establishes a reproducible
static shortcut under the current empirical cold availability and heater
model. It also means that starting online transfer or PPO here would test
learner noise against an environment with no material adaptive opportunity.

The heater route is therefore closed. A successor should use a predeclared
budget phase screen that changes which complete subsets are feasible, with
the actual entity-mapped costs and the same 32-mask geometry audit at every
budget. The phase screen must be evaluated as a family before selecting any
budget; no point may be chosen from its downstream loss after the fact.

## 2026-09-10 V618 budget phase finding

The complete predeclared budget screen did not identify a viable operating
interval. At B=`1.25`, `1.50`, and `1.75`, the four-seed operating gaps were
`[0, 0.031827, 0, 0]`; at B=`1.90` they were
`[0.008346, 0.001095, 0, 0]`; at B=`2.05` they were
`[0.008491, 0.001113, 0, 0.001009]`; and at B=`2.15` they were
`[0.008473, 0.001132, 0, 0.001006]`. None satisfies the all-seed materiality
gate. The favorable seed at the three lowest budgets is isolated and does not
justify selecting one of those budgets for training.

This closes the empirical-cold/heater route without an online transfer or
policy result. Resource frontier movement alone is not enough when the
dominant forecast-optimal subset remains stable across the actual operating
distribution. The next screen must test observability and downstream value on
the stronger V557 causal specialist-separated geometry, with any added context
derived from variables available at decision time and frozen before evaluation.

## 2026-09-10 V619 observability repair rationale

The V541 truth files contain eight deployable context columns: four weather
nowcasts and four delayed noisy alert proxies. However, the V541 metadata
reports `agent_context_columns: []`; its 521-dimensional observation therefore
excluded those columns. The subsequent V558 transfer audit consequently tested
the causal scene without its intended online context and obtained low test
top-1 coverage (`0.094--0.144`) and mean action regret (`0.128--0.184`).

V619 repairs this protocol omission without changing the scene, labels,
resource trace, budget, partition, or candidate family. The context columns
are generated before asset fitting and are available at decision time. The
repair must first pass the same complete-subset geometry gate; only then may
online transfer and PPO be considered.

## 2026-09-10 V619 interpretation boundary

The repaired geometry is materially adaptive in three seeds but not in all
four. Because each asset wave refits a stochastic TCN, the new oracle hashes
cannot be compared as if context inclusion were the sole intervention. V619
is therefore evidence that the context-aware asset protocol can preserve
strong geometry, not evidence that context caused the seed-level changes.
V620 is the paired no-context control needed to separate those effects. No
online transfer or PPO result is promoted from V619 alone.

The V620 control asset wave completed all four manifests with empty context
columns. Its first geometry launch was discarded as an execution-path error:
the shell did not change to the remote repository root, so no Python audit
started and only an empty seed log was created. The corrected rerun uses a
fresh output directory and the exact matched geometry command.
## 2026-09-10 V619/V620 paired comparison

V620's no-context control produced operating gaps `0.092539614`,
`0.101896249`, `0.126597234`, and `0.084154605` for seeds `7241--7244`,
with mean `0.101296926`; all four had empty 1% near-optimal static
intersections. V619 with context produced `0.099816328`, `0.078951824`,
`0.000011251`, and `0.058387964`, with mean `0.059291842`; seed 7243 kept a
non-empty 1% static intersection.

The V619 and V620 oracle hashes differ for every seed, so refitting
randomness is a confounder. The result cannot attribute the difference to
context inclusion or removal. A valid follow-up must reuse the V541 oracle and
change only the observation metadata and merged truth columns.

## 2026-09-10 V621/V622 frozen-context result

The metadata-only V621 repair is a controlled intervention: its oracle hashes
and per-seed geometry are identical to V541, while the observation contract
contains eight decision-time context columns.

V622 did not pass the held-out transfer gate. Static-minus-transfer margins
were `0.003665516`, `0.009662347`, `-0.038180794`, and `-0.020078376` for
seeds `7241--7244`, giving mean `-0.011232827` and `2/4` positive seeds. The
selected transfer schedule remained dynamic, with switch rates `0.242--0.369`
per row, so the failure is value-ranking transfer error rather than a no-switch
artifact. Privileged geometry remains a diagnostic opportunity, not deployable
policy evidence; PPO is blocked on this route.

## 2026-09-10 V626 coverage-probe result

The round-robin observation probe was a controlled diagnostic for the
candidate-000 observation coverage concern. It retained the corrected V624
losses, frozen oracle, context metadata, resource trace, and feasibility
budget, while changing only the deployable probe schedule. Transfer remained
negative in every seed-level mean: `-15221.665`, `-6177.913`, `-12414.541`,
and `-9641.401`, with `4/16` positive folds. The observation-coverage
hypothesis is rejected for this route. No PPO training is justified.

## 2026-09-10 Transfer diagnostic invalidation

The V622/V623 transfer outputs cannot be used for a scientific gate. Their
input geometry files were generated with `oracle_loss_clip=100`, and an audit
found exact-100 clipping in `91.96%--93.93%` of rows. The transfer regressors
therefore learned an almost constant target; a negative static-minus-transfer
margin in that setting does not establish failure of the online context.

The V622/V623 route is reopened as a diagnostic. A corrected geometry audit
with the same V621 frozen oracle and protocol but a nonbinding loss clip is
required before any observability or PPO decision.

## 2026-09-10 V624/V625 corrected result

The corrected V624 geometry removed the artificial loss clipping while keeping
the frozen V621 oracle, resource trace, budget, starts, and context metadata.
All four operating gaps remained positive and no 1% static intersection was
present. V625 then repeated the full leave-one-start-out transfer audit on
these losses. Per-seed mean margins were `-16823.536`, `-5875.860`,
`-10634.562`, and `-9520.044`; only `3/16` folds were positive. The negative
result is therefore not a clipping artifact. The route is closed before PPO:
privileged subset-value geometry is present, but the current decision-time
observation does not support reliable value transfer.

## 2026-09-10 V623 transfer sensitivity

The leave-one-start-out audit used the same V621 frozen oracle, context
metadata, candidate losses, budget, and resource trace. It removed the small
training-sample concern from V622 by using three starts for each training fit
and one held-out start per fold. Nevertheless, the four seed-level mean
margins remained negative: `-0.015383`, `-0.005356`, `-0.031197`, and
`-0.004197`. No seed met the required positive mean margin and at-least-three
positive-fold rule. The transfer route is therefore closed before PPO.
## V627 route definition (2026-09-10)

The previous V619--V626 transfer probes were closed because the observation
trajectory and candidate-loss labels were not action-conditionally aligned;
they are not PPO evidence. The replacement route tests the physical chain
first. `scripts/180_build_observable_physical_resource_trace.py` applies only
the manifest-declared heater hysteresis rules to weather variables available at
decision time. It ignores generator modes, event labels, and future targets.
The remote screen must establish heater occupancy and a changing feasible
frontier before any frozen forecaster or policy is created.

V629 passed the resource-frontier prerequisite but failed the downstream
forecast-geometry gate. The failure is informative: the V608 target-mode
process and the nowcast heater controller were independent, so changing the
feasible masks did not consistently change the forecast-optimal subset. The
next admissible generator must share a deployable operating-factor chain
between target innovation, sensor quality, and effective power. It must remain
causal and must not export the latent factor, event label, or candidate loss to
the scheduler.

## 2026-09-10 Observable physical-resource route

The heater/resource implementation is connected to the executable-mask guard.
The environment reads the per-channel `resource_effective_power_*` columns
through the dynamic-resource mapping and applies them to feasibility. The
summary's fixed `steady_cost` fields are only legacy normalized action costs;
they do not represent the dynamic-resource cost.

The full V633 time axis has 11 to 16 feasible optional-subset masks under the
2.15 effective budget. The laser channel is feasible for about 15% of rows
because its declared heating increment removes it from the frontier during
heating. This is genuine resource-geometry change, but V635 sampled only
simultaneous-heater starts. V636 is a coverage correction: starts are fixed by
the first occurrences of resource states before forecast losses are inspected.

The V630-V635 causal factor coupling still produced small condition-wise
forecast gaps (`0.000000--0.005268`), so V636 must pass the downstream gate
before this route can admit PPO training. If V636 also fails, this physical
scene family will be closed without PPO.

## 2026-09-10 Sensor-specific quality coupling

V639 exposed a modeling omission: the first physical resource trace changed
feasibility but not the measurement quality of the heated channels. V637
corrected this by applying a bounded quality improvement when the declared
GMX500/Parsivel heater is active. This is a physical observation-model change,
not a policy or reward adjustment.

The remaining seed-level failure was a radiometer subset that stayed optimal
across all operating labels. V640 therefore adds one bounded signal-quality
relation for the radiometer, using only decision-time solar irradiance. V642
is the final geometry check for this scene family; no PPO evidence will be
promoted from V637-V641 without its result.

## 2026-09-10 V642 closeout

The final physical-observation correction did not remove the seed-specific
static shortcut. Operating gaps were `0.046997`, `0.119788`, `0.017873`, and
`0.000000`; the mean is positive, but the predeclared all-seed gate fails.
The zero-gap seed has the same radiometer subset as the best candidate in all
three operating bins and retains a 1% near-optimal static intersection.

This is a useful negative result about scene readiness, not PPO performance:
the resource constraint changes executable subsets, while the current frozen
forecaster does not consistently value those changes across seeds. PPO must
not be trained on this route because it would confound a policy result with a
scene-design failure.
## 2026-09-10 Energy route correction

The repository contains a usable finite-energy environment, but the existing
entity audit is not a defensible scene: it samples 25 hourly truth rows and
uses a 600 W external hysteretic heater that is explicitly non-controllable.
The no-heater trace leaves the 8640 Wh account almost unchanged, while the
heater trace rapidly triggers the guard. Reusing historical `harvest_per_step`
and `capacity` constants would be an arbitrary intervention. The SOC route is
therefore diagnostic-only and closed for the current work unit. Continue with
hardware-derived budget breakpoints or obtain a traceable supply/storage
trajectory before any SOC-based PPO experiment.

## 2026-09-10 Real hardware supply-field audit

The SEUAWS room/freezer test archive supplies 115 hours of voltage telemetry:
Modbus battery voltage averages 12.6829 V and Parsivel2 supply voltage
averages 12.0328 V. It also records heating current and state, but nonzero
heating is rare (`74/17,720` rows), and no system-level current or charging
trace is present. These data validate acquisition fields only. They do not
justify a synthetic Antarctic battery/SOC process or extrapolated heater duty.
See `reports/entity_supply_validation_20260910.md`.

## V616 task-level geometry finding

The V616 resource/quality route is not an all-or-nothing failure. Under the
32-subset frozen evaluator, operating-condition winners vary across three of
four seeds, but the attainable loss reduction over the best static subset is
only `0.00100--0.00195`, and one seed has zero gap. A changing feasible
frontier therefore does not by itself establish a useful downstream forecast
opportunity. The gate must remain at the complete-subset forecast level.

## V645 admissible intervention

The next intervention is restricted to the target side of the causal chain.
It uses the already measured cold-availability quality proxy to scale a
delayed persistent wind innovation. The audit column is not an observation
feature, exact failure labels are not used, and no PPO-facing architecture or
reward changes are allowed. If this truth-only screen does not produce stable
subset forecast geometry, the empirical-cold route will be closed instead of
being tuned around the geometry result.

## V653 audit-definition failure

V653 correctly loaded the persistent Stage-B factor columns, but the first
correction still used an argmax label. The V535 generator intentionally gives
the thermal state a persistent baseline while the flux and particle states
share the transport load. An argmax therefore assigns nearly every row to
thermal even though transport and thermal activity vary jointly. This makes
the reported operating gap meaningless and cannot be used to accept or reject
the scene. The audit must preserve the multi-factor state using fixed causal
thresholds and explicit combinations; V654 is the corrected rerun.

## V654 Stage-B closeout

The corrected multi-factor labels removed the audit artifact, but did not
rescue the scene. Held-out operating gaps were `0`, `0`, `0`, and `0.001529`,
well below the `0.01` materiality requirement. The conditionwise gaps were
`0`, `0.011166`, `0.000879`, and `0.004581`, so the apparent opportunity was
not stable across seeds or chronological starts. The normalized 20 W guard
also left all 32 candidate masks supported in the sampled windows, which means
the intended resource geometry was not binding. This route is closed before
online transfer and PPO. The next route must repair the physical budget
interface or introduce an independently justified state-dependent cost before
any learner is trained.

## V645 closeout finding

The delayed cold-risk wind innovation increased target perturbation in low
quality periods, but it did not consistently change the downstream best subset.
One seed gained a material `0.015164` conditionwise opportunity, while the
other three were `0.000000`, `0.000944`, and `0.004598`; three seeds retained a
1% near-optimal static candidate. This separates target variability from
forecast-value variability: making a target harder under cold risk is not
enough to create a deployable adaptive scheduling opportunity. The route is
closed without PPO.

## V616 task-level geometry finding

The V616 resource/quality route is not an all-or-nothing failure. Under the
32-subset frozen evaluator, operating-condition winners vary across three of
four seeds, but the attainable loss reduction over the best static subset is
only `0.00100--0.00195`, and one seed has zero gap. A changing feasible
frontier therefore does not by itself establish a useful downstream forecast
opportunity. The gate must remain at the complete-subset forecast level.

## V645 admissible intervention

The next intervention is restricted to the target side of the causal chain.
It uses the already measured cold-availability quality proxy to scale a
delayed persistent wind innovation. The audit column is not an observation
feature, exact failure labels are not used, and no PPO-facing architecture or
reward changes are allowed. If this truth-only screen does not produce stable
subset forecast geometry, the empirical-cold route will be closed instead of
being tuned around the geometry result.
## 2026-09-10 V610 geometry conclusion superseded

The first V610 report is not a valid operating-state result. The resource
builder adds `resource_frequency_mode_id`, but the audit selected heater-state
columns first and reported `heater_00000`, `heater_01000`, and `heater_11000`.
Because the reported operating gap and near-optimal intersection depend on
that partition, the V610 closeout is withdrawn pending V668. The correction
changes only truth-side evaluation stratification; the scheduler never sees
the mode id.
## 2026-09-10 V668 corrected frequency-mode closeout

The corrected V668 audit confirms that the original V610 failure was partly
an audit-label bug, but correcting it does not rescue the route. The operating
forecast gaps for seeds `7177--7180` are `0`, `0.0000262`, `0`, and `0`; every
seed retains a nonempty 1% near-optimal static intersection. The resource
frequency mode changes feasibility, but the downstream forecast objective
still admits a static shortcut after deployable state stratification. This is
a scene-geometry failure, not a PPO failure. No online transfer or PPO run is
permitted on V610/V668.
## 2026-09-10 V669 candidate scene

The first candidate after V668 uses shared observable weather drivers instead
of a randomly generated operating mode. Wind and humidity drive transport and
particle demand, while cold/dew-point/radiation conditions drive thermal
and icing demand. The same drivers are applied with a declared six-step lag
to target innovation and channel quality, and the existing hardware-derived
hysteresis controller generates the resource trace.

The truth-only screen is promising but not evidence of adaptive scheduling:
all four seeds have balanced driver support, `11--16` feasible masks per row,
`11/32` always-feasible masks, and about `85%` frontier changes. The required
next test is a frozen 32-subset forecast geometry audit. No PPO should be
started unless operating condition gaps are material, the near-optimal static
intersection is empty, and a train-only deployable observation policy transfers
through the dwell-aware environment.

## 2026-09-10 V670 schema correction

The first V670 geometry attempt stopped before rollout because the new truth
generator omitted the mandatory `agent_context_quality_cr1000xe_backbone`
column. This is an environment schema error, not a geometry result. The
generator now sets the backbone quality to `1.0`; V670 must be regenerated and
rerun with the same assets, starts, and budgets.
## 2026-09-11 V670 final-window support failure

V669's shared nowcast relation produced balanced driver support, but its
weather-derived heater controller saturated in the final partition. V670's
four geometry audits all saw only `heater_11000`, with zero deployable
operating opportunity and a nonempty static intersection. The condition-only
gaps (`0.00018`, `0.00002`, `0`, `0.00124`) cannot be used because the policy
cannot observe a condition that is absent from the final windows.

A conditional controller removing the unconditional cold trigger was then
screened locally. It produced approximately zero heater duty because the
dew-point risk inputs were too weak in the available truth, so it is rejected
as a scene candidate rather than threshold-tuned to force occupancy. The next
route must obtain full final-window support from a traceable physical driver
before any new assets are built.
## 2026-09-11 V672 resource support

The shared-driver controller restores state support in the final partition
without changing target truth or fixed frequency multipliers. Each seed has
all four core/laser heater combinations in the final partition, with 11--16
feasible subsets per row. This passes the resource occupancy gate only; the
forecast evaluator must still demonstrate condition-specific subset value and
chronological online transfer.
## V672 shared-driver geometry (2026-09-11)

The shared-driver resource controller successfully created four supported
operating states and a changing feasible frontier, but the forecast objective
did not consistently inherit that variation. Seed7177 had a material operating
opportunity gap (`0.03418140`) and an empty 1% near-optimal static
intersection. Seeds7178 and 7179 had effectively zero operating gaps and seven
near-optimal static candidates; seed7180 had a small gap (`0.00234366`) and two
near-optimal candidates. This separates resource occupancy from task-level
adaptive opportunity and closes V672 before online transfer or PPO.

## V673 heater-quality geometry (2026-09-11)

Applying the existing heater-quality relation made the measurement-quality
state explicitly change with the same operating risk and heater state. It did
not change the decision: only seed7177 had a material operating gap
(`0.03421241`) with an empty 1% static intersection. Seeds7178--7180 had
operating gaps `0`, `0.00009367`, and `0.00010058`, with 7, 7, and 2
near-optimal static candidates. The next design must alter the shared target,
quality, and resource persistence/innovation chain; further PPO tuning is not
justified before that policy-free gate passes.

## V676 binding-budget geometry closeout (2026-09-11)

The predeclared lower-budget probe did not rescue the scene. At normalized
budget `1.45`, all four operating forecast gaps were below `0.001`, far below
the `0.01` materiality gate, and every seed retained a nonempty 1% near-optimal
static intersection. The smaller feasible frontier therefore made the action
space more constrained without making condition-specific subset value
separate. Further budget-only sweeps are not justified. The next diagnostic
must identify the static winning subset and measure its condition/block regret
before any new truth generation or policy training.

## V678 specialist-pair geometry closeout (2026-09-11)

Budget `2.50` corrected the mechanical limitation identified by V677: all
three specialist pairs can now be startup-feasible while their union remains
infeasible. However, only seed7177 passed the downstream operating geometry
gate. Seeds7178--7180 had operating gaps `0.0000063937`, `0.0001480981`, and
`0.0026746816`, with nonempty 1% static intersections. The remaining blocker
is therefore the cross-seed target/quality innovation process, not the
arbitrary-subset action representation or PPO capacity. No transfer or PPO
was run.

## V679 audit correction (2026-09-11)

An initial read-only target-quality audit used the stale resource columns still
present in the V675 truth CSV and therefore reported only `1100` heater state.
That output is invalid and is not promoted. The corrected audit follows the
geometry implementation by dropping stale resource columns and merging the
manifest-declared V672 resource trace. It recovers all four states (`00000`,
`01000`, `10000`, `11000`) across all four seeds. This was an audit-label
correction only; it changes no V678 geometry result.

## V680 cross-scene mismatch closeout (2026-09-11)

V680 reused the V527-r2 observable target/quality truth with the V672 resource
trace. The geometry failed in three of four seeds, with operating gaps `0`,
`0.0009802611`, and `0` and nonempty static intersections. Because the
resource trace was generated from a different target process, this route does
not isolate the V527-r2 truth relation. The next route must generate resource
loads from the same observable drivers and mode proxy.
## V681 matched resource design (2026-09-11)

The V527-r2 truth exposes causal, deployable forecast-mode scores and
low-pass wind/thermal drivers. The new resource controller uses:

```text
core  = 0.25 + 0.45*thermal_score + 0.15*transport_score + 0.15*cold_load
laser = 0.20 + 0.40*particle_score + 0.25*transport_score + 0.15*wind_load
```

with fixed hysteresis thresholds on=`0.50` and off=`0.35`. It does not read
`generator_persistent_mode_id` or `generator_independent_target_innovation`.
Local final-window screening gives all four heater states in all four seeds
and distinct feasible frontiers with 16, 11, 16, and 11 masks at budget 2.50.
This is a resource-geometry pass only; downstream forecast geometry remains
untested and PPO remains blocked.

## V681 geometry closeout (2026-09-11)

Matched frozen assets and the 32-subset geometry audit completed for all four
seeds. Operating gaps were `0.0020498832`, `0`, `0.0003399052`, and
`0.0000071095` for seeds 7177--7180. The 1% operating near-optimal static
intersections were nonempty in every seed. The resource chain is therefore
causal and state-supported, but its variation still does not create material
forecast-value separation. V681 is closed before online transfer and PPO.
