# 2026-06-20 Met+Specialist Paper Narrative Audit

## Rewrite Status

The paper mainline has now been rewritten around the supported
static-normalized event-regime macro claim.

Updated files include:

- `paper/main.tex`;
- `paper/sections/01_introduction.tex`;
- `paper/sections/02_related_work.tex`;
- `paper/sections/03_problem_formulation.tex`;
- `paper/sections/04_framework_protocol.tex`;
- `paper/sections/05_simulation_setup.tex`;
- `paper/sections/06_results.tex`;
- `paper/sections/07_discussion_future_work.tex`;
- `paper/sections/08_conclusion.tex`;
- `paper/tables/sensor_specs.tex`;
- `paper/tables/metpair_staticnorm_macro_summary.tex`.

The rewritten manuscript states:

- main supported claim: static-normalized event-regime macro forecast loss,
  `13/14` seed gates;
- behaviour gate: `14/14`;
- macro sign-test: `p=0.00091552734375`;
- strict step-weighted fixed-static gate: `10/14`, reported as a limitation.

Local and `remote-gpu` builds both complete with `latexmk`. The pre-rewrite
paper was archived at
`paper/_archive/pre_staticnorm_macro_rewrite_20260620_232507/`.

## Current Paper Entry Point

- Canonical source is `paper/main.tex`, not `paper/paper.tex`.
- `paper/main.tex` states it was promoted from `pdppo_crst_rewrite.tex` and
  canonical edits should be made there.

## Main Finding

The current manuscript still tells the older V3.1 operational-constraint story:
eight logical channels, ten seeds, duty-cycle/static-priority comparisons, and
candidate-policy regularization. The new validated evidence is different:

- final candidate: `v31_metpair_stronglatent_seed45_h075_20260620`;
- six-sensor met+specialist-pair contract;
- one stable meteorological backbone plus one state-dependent specialist slot;
- strict no-duty-guard fixed-static replay is the decisive static-shortcut gate;
- learned PPO is verified on seed 45, with optional multiseed robustness still
  pending.

This requires a narrative rewrite, not a light wording pass.

## 2026-06-20 Re-Audit After Multiseed Failure

The earlier seed45 met+specialist result is no longer sufficient for the paper
mainline. Subsequent seed pools showed that the single-seed "key progress" was
not a robust strong-claim result:

- old metpair 7-seed pool: not supported;
- backbone-context 7-seed pool: behaviour improved but strict static gates still
  failed on several seeds;
- ortholinear strong-teacher 14-seed pool: behaviour gate passed in all seeds,
  but strict step-weighted and raw macro gates remained below the required
  robustness threshold.

The current active direction is a stricter, more explicit claim:

> PD-PPO learns a state-dependent specialist scheduler that improves a
> static-normalized event-regime macro forecast objective under a fixed
> meteorological backbone and one-specialist budget.

This claim now has posthoc static-normalized multi-seed support from the
ortholinear strong-teacher 14-seed pool:

```text
reports/aggregate/metpair_ortholinear_strongteacher_14seed_staticnorm_replay_20260620/
complete seeds: 14
macro seed gate: 13/14
behaviour gate: 14/14
macro claim strength: strong_macro_multiseed
macro sign-test p: 0.00091552734375
strict step seed gate: 10/14
```

The manuscript can be rewritten around the static-normalized event-regime macro
claim. It still must not claim broad step-weighted optimality or unconditional
multi-seed superiority over true fixed static schedules, because the strict step
gate fails in 4/14 seeds.

Narrative consequences:

- Seed45 numbers should appear only as a historical calibration point or be
  replaced by the 14-seed aggregate.
- The results section should be rebuilt around the static-normalized macro
  contract, with seed48 shown as the single macro failure.
- The queued reward-aligned balanced-objective branch is useful robustness
  evidence, but the current manuscript rewrite no longer has to wait for it if
  the claim is stated as static-normalized event-regime macro scheduling.

## Must Change

### Abstract

Affected source: `paper/main.tex:72`.

Problems:
- Claims "all eight logical channels" and no permanently on/off channels.
- Claims ten independent final-test seeds.
- Frames success against dynamic and duty-cycle-constrained baselines while
  explicitly excluding the validation-selected static schedule.

Required rewrite:
- State the main result as a calibrated forecast-oriented contextual specialist
  scheduling scene.
- Report the actual static gate:
  `custom_ppo=0.485635` vs `validation_selected_static=0.491597`; explicit
  dynamic replay `0.482174` vs true fixed static `0.492351`.
- Describe the behaviour as fixed meteorological backbone plus state-dependent
  specialist slot, not all-channel balanced use.
- Either mark the result as single-seed/calibration-stage evidence or run
  additional seeds before claiming multiseed robustness.

### Introduction

Affected source: `paper/sections/01_introduction.tex:41`.

Problems:
- The benchmark is described as eight logical channels.
- The main result claims all eight channels in use and ten final-test seeds.
- The contribution list still foregrounds duty-cycle and candidate-policy
  regularization rather than static-shortcut removal.

Required rewrite:
- Introduce the static-shortcut problem explicitly: a scenario is not useful if
  a fixed subset or a duty-guard-induced rotation wins.
- Present the met+specialist-pair design: one cheap meteorological backbone can
  pair with exactly one costly specialist, while two specialists are infeasible.
- Replace the result paragraph with the metpair evidence and behaviour audit.
- Update contributions to include the strict no-duty-guard static replay gate
  and corrected behaviour-complexity audit.

### Related Work

Affected source: `paper/sections/02_related_work.tex:54`.

Problems:
- Says the benchmark maps to eight logical channels.
- Does not explain why fixed static designs are a methodological threat for
  forecast-oriented scheduling.

Required rewrite:
- Replace eight-channel benchmark wording with the six-sensor specialist setup.
- Add one paragraph on static shortcut evaluation: learned scheduling must beat
  true fixed-static references, not only cycling/random baselines.

### Problem Formulation

Affected source: `paper/sections/03_problem_formulation.tex:22`.

Problems:
- Terminology distinguishes validation-selected static and static-priority
  schedules, but the decisive new baseline is true fixed static with duty guard
  disabled.
- Duty-cycle constraints and energy-account analysis are foregrounded, but the
  final metpair gate uses no static duty guard and does not rely on the
  energy-account branch.

Required rewrite:
- Add "true fixed-static replay" or "replay-local fixed-static reference" as a
  named baseline.
- Clarify that duty-guard rotations are not valid evidence for breaking the
  static shortcut.
- Move energy-account material to future work or appendix unless a validated
  energy-account result is added.

### Method

Affected source: `paper/sections/04_framework_protocol.tex:67`.

Problems:
- The method currently claims two forecast-guided regularization terms:
  AWBC and candidate-policy KL.
- The final metpair run has `prior_kl_coef=0.0` and `use_candidate_prior=false`;
  the active learning supports are subtype-teacher AWBC, BC pretraining,
  subtype auxiliary loss, and subtype router.

Required rewrite:
- Make candidate-policy KL optional or remove it from the main method claim.
- Add the observable/training-only subtype supervision path used by the final
  run.
- Update the chronological evaluation protocol from old 1024-step final windows
  to the metpair 512-step event-rich windows.
- Update baselines to include validation static, true fixed static no-duty-guard
  replay, explicit subtype replay as a diagnostic upper bound, and standard
  dynamic baselines.

### Simulation Setup

Affected source: `paper/sections/05_simulation_setup.tex:5`.

Problems:
- Sensor table and simulator parameters describe the old eight-channel setup.
- Budget is old `B in {1.65, 1.70, 1.75}` rather than final `B=0.75`,
  startup peak `0.95`, and max active `2`.
- Sequence length is old 90,000 epochs, while final metpair run uses 70,000.

Required rewrite:
- Replace `tables/sensor_specs.tex` or create a new metpair sensor table:
  `met_station_core`, `radiometer_basic`, `shielded_thermo_hygro`,
  `surface_temp_ir`, `laser_disdrometer`, `fc4_flux`.
- Document the key feasibility contract:
  `met + specialist = 0.74 <= 0.75`; two specialists are infeasible.
- Update simulator parameters: 70,000 truth steps, split ratios
  `[0.35, 0.50, 0.075, 0.075]`, final-test event-rich windows, TCN oracle.

### Results

Affected source: `paper/sections/06_results.tex:3`.

Problems:
- Entire main results section reports old ten-seed operational results.
- It explicitly says static-priority is slightly better on mean in places, which
  conflicts with the new main claim.
- Budget/dwell sensitivity, candidate-prior ablation, fixed-budget, and
  energy-account sections are stale for the metpair mainline.

Required rewrite:
- Replace the main result with a metpair gate table:
  learned PPO, router-confidence PPO, validation-selected static, replay-local
  true fixed static, explicit dynamic replay, round-robin, AoI, random, and
  diagnostic full-open/static-projected rows if retained.
- Add a behaviour table/figure showing the four learned masks:
  `met+laser`, `met+fc4`, `met+shielded`, `met+surface`.
- Report event-conditioned sensor-duty deltas:
  shielded mostly non-event, laser/fc4 event-sensitive, surface thermal.
- Move old V3.1 ten-seed operational tables to appendix or remove them from the
  main text.
- Remove candidate-prior ablation from main text unless the method is restored
  in the final metpair run.
- Remove energy-account analysis from main results or label it explicitly as
  exploratory and not part of the validated mainline.

### Discussion And Conclusion

Affected sources:
- `paper/sections/07_discussion_future_work.tex:4`
- `paper/sections/08_conclusion.tex:10`

Problems:
- Claims all channels remain at intermediate duty.
- Frames strongest evidence as dynamic-baseline wins, not static-shortcut
  breaking.
- Does not present the actual learned deployment pattern.

Required rewrite:
- Main design implication should be:
  forecast-oriented RL is useful when the hardware budget permits a stable
  context backbone plus one context-dependent specialist, but does not permit
  all specialists.
- Be explicit that one backbone sensor is always active.
- State the limitation: additional seeds/nearby budgets are robustness work
  before final submission.

## Tables And Figures To Replace Or Move

Main-text stale assets:
- `paper/tables/sensor_specs.tex`
- `paper/tables/simulator_parameters.tex`
- `paper/tables/env_dwell12_operational_results.tex`
- `paper/tables/env_dwell12_event_diagnostics.tex`
- `paper/tables/env_dwell12_budget_sensitivity.tex`
- `paper/tables/env_dwell12_dwell_sensitivity.tex`
- `paper/tables/env_dwell12_candidate_prior_ablation.tex`
- `paper/tables/main_results_v31.tex`
- `paper/tables/energy_account_storm_oracle.tex`

Likely stale figures:
- `figure_operational_summary.png`
- `figure_operational_behavior.png`
- `figure_fixed_budget_power_error.png`
- existing framework figure if it still shows eight channels or candidate KL
  as mandatory.

## Recommended Rewrite Order

1. Replace sensor/specification tables with the metpair scenario.
2. Replace results section first, because all claims should flow from the new
   evidence table and behaviour audit.
3. Rewrite abstract, introduction, and conclusion after the result section is
   stable.
4. Then adjust method and problem formulation so they describe the actual final
   run rather than the older V3.1 protocol.
5. Compile and run a claim audit for forbidden stale claims:
   `eight logical channels`, `all eight`, `ten independent final-test seeds`,
   `candidate-policy regularizer`, `static-priority`, `energy-account` in main
   results.
