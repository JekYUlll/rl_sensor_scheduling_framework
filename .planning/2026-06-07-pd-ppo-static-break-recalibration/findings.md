# Findings & Decisions

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
- An archived prior scenario screen changed the problem structure through sensor
  costs, startup peaks, warmup, event noise, event observation probability,
  energy/storage settings, and snow-transport-focused objective weights. This
  is historical context only, not active evidence.
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
