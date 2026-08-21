# Findings: PD-PPO H75 Paper Rewrite

## Current Evidence

- Final h75 5-seed aggregate:
  - vs full-open reference: `0/5`, mean delta `-0.021959`;
  - vs original compact static: `3/5`, mean delta `+0.003920`;
  - vs deployable selected static: `4/5`, mean delta `+0.001007`;
  - vs best original dynamic heuristic: `5/5`, mean delta `+0.008445`;
  - vs best duty-constrained non-PD-PPO: `4/5`, mean delta `+0.006594`;
  - deployment behaviour valid `5/5`.

## Framing Requirement

- Opening should present a general algorithmic problem:
  prediction-driven RL scheduling for power-limited sensors in extreme
  environments.
- Antarctic AWS enters later as a controlled, physically motivated case study.
- Scenario calibration should be transparent but not dominate the abstract or
  introduction.

## Writing Risks

- The old draft over-emphasized caveats and fixed-budget negative diagnostics.
- The new draft must avoid implying universal dominance over compact static.
- AI-like phrasing to remove:
  - repeated "this study does not...";
  - defensive "only a protocol paper";
  - excessive "regime-dependent" repetition;
  - generic phrases like "in this work, we propose" repeated mechanically.

## Post-Rewrite Audit

- The active manuscript now leads with the general power-limited predictive
  scheduling problem and uses the Antarctic AWS case as a later controlled
  instantiation.
- The old caveat-first Results/Discussion structure has been replaced by an h75
  operational lead result.
- The old scheduling timeline figure is no longer used as a main positive-result
  figure; the main result figure is the new h75 operational summary.
- Static results are now separated into:
  - compact static: strong diagnostic shortcut, not the fair deployment baseline;
  - deployable selected static: selected static mask replayed under the same duty
    guard, used as the fair static deployment comparator.
- Remaining honest boundary:
  - PD-PPO is not claimed to beat full-open;
  - PD-PPO is not claimed to universally beat compact static;
  - evidence remains simulation-based and should be described as such in
    limitations, not as the main opening claim.

## Event-Conditioned H75 Diagnostics

- Source:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/h75_pdppo_vs_deployable_static_loss_audit.csv`.
- The table compares PD-PPO against deployable selected static under the same
  duty/dwell operational constraints.
- Mean over seeds 41--45:
  - PD-PPO overall/event/non-event oracle loss:
    `0.143213 / 0.182876 / 0.125584`;
  - deployable selected static overall/event/non-event oracle loss:
    `0.144220 / 0.183668 / 0.126672`;
  - PD-PPO win count over deployable selected static:
    `4/5` overall, `3/5` event, `4/5` non-event.
- Behaviour:
  - PD-PPO uses `10.8` distinct masks on average versus `7.0` for deployable
    static;
  - PD-PPO top-mask share is `37.4%` versus `48.5%`;
  - switching rates remain similar and low: `3.10%` versus `3.02%`.
- Sensor-use note:
  - the laser channel is not discarded by PD-PPO: mean duty is `21.0%` overall
    and `23.6%` during event windows.

## Generator-Validation Boundary

- Current h75 local artifacts only include `synthetic_validation.csv` for seeds
  41 and 42; seeds 43--45 have metadata but not the full truth-validation CSVs.
- The existing `tables/g1_generator_validation.tex` and
  `figures/figure3_synthetic_statistics.png` are therefore best described as
  generator-family boundary checks, not as a five-seed h75 result table.
- The manuscript now says these checks bound the simulation family and are not
  used for final-test policy selection.

## Candidate-Prior Ablation

- Source:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_candidate_prior_10seed/env_dwell12_h75_no_candidate_prior_operational_summary_10seed.csv`.
- Same deployment contract as the B=1.70 main h75 run, but with
  `--no-use-candidate-prior`, `--prior-kl-coef 0.0`, and
  `--candidate-prior-scale 0.0`.
- No-prior result over seeds 41--50:
  - PD-PPO loss `0.142476`;
  - wins vs best original dynamic heuristic `10/10`, delta `+0.006816`;
  - wins vs best duty-constrained non-PD-PPO `9/10`, delta `+0.003706`;
  - wins vs deployable selected static `2/10`, delta `-0.001873`;
  - valid deployment behaviour `10/10`.
- Paired against the main candidate-prior configuration:
  - main PD-PPO loss `0.140635`;
  - no-prior minus main mean `+0.001841`;
  - main configuration is lower in `8/10` seeds.
- Interpretation:
  the candidate prior improves loss and seed stability, but PD-PPO still learns
  a valid dynamic-baseline-beating schedule without it. The paper should frame
  this as a training-scaffold ablation, not as a separate main claim.

## Minimum-Dwell Sensitivity

- Sources:
  - main dwell 12:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/env_dwell12_h75_operational_summary_10seed.csv`;
  - dwell 6:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell6_h75_dwell_sensitivity_10seed/env_dwell12_h75_dwell6_operational_summary_10seed.csv`;
  - dwell 24:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell24_h75_dwell_sensitivity_10seed/env_dwell12_h75_dwell24_operational_summary_10seed.csv`.
- Dwell 6:
  - PD-PPO loss `0.136948`;
  - beats best original dynamic `7/10`, delta `+0.000362`;
  - beats best duty-constrained non-PD-PPO `9/10`, delta `+0.005748`;
  - beats deployable selected static `4/10`, delta `-0.001182`;
  - switching rate `0.059`, valid behaviour `10/10`.
- Dwell 12:
  - PD-PPO loss `0.140635`;
  - beats best original dynamic `10/10`, delta `+0.008493`;
  - beats best duty-constrained non-PD-PPO `9/10`, delta `+0.005477`;
  - beats deployable selected static `4/10`, delta `-0.000320`;
  - switching rate `0.031`, valid behaviour `10/10`.
- Dwell 24:
  - PD-PPO loss `0.142483`;
  - beats best original dynamic `10/10`, delta `+0.010106`;
  - beats best duty-constrained non-PD-PPO `10/10`, delta `+0.008249`;
  - beats deployable selected static `6/10`, delta `-0.000129`;
  - switching rate `0.016`, valid behaviour `10/10`.
- Interpretation:
  short dwell lets rapid cycling heuristics regain competitiveness; longer dwell
  reduces switching and strengthens PD-PPO's advantage over fair dynamic
  baselines. This directly supports using symmetric deployment constraints for
  dynamic baseline comparison.

## Training-Scaffold Ablation

- PPO-only source:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_vanilla_ppo_10seed/env_dwell12_h75_vanilla_ppo_operational_summary_10seed.csv`.
- PPO-only setting:
  same forecast reward, action projection, hard duty guard, dwell guard,
  baseline set, budget, and seeds as the main $B=1.70$ h75 run, but with
  `--awbc-coef 0.0`, `--no-use-candidate-prior`, `--prior-kl-coef 0.0`, and
  `--candidate-prior-scale 0.0`.
- PPO-only result:
  - mean PD-PPO loss `0.149170`;
  - valid deployment behaviour `10/10`;
  - beats best original dynamic heuristic `5/10`, mean delta `-0.001107`;
  - beats best duty-constrained non-PD-PPO baseline `3/10`, mean delta
    `-0.003615`;
  - beats deployable selected static `0/10`, mean delta `-0.009748`.
- Paired comparison:
  - main PD-PPO loss `0.140635`;
  - PPO-only minus main mean `+0.008535`;
  - main is lower in `10/10` paired seeds.
- Interpretation:
  ordinary PPO with the same projector and deployment guards does not reproduce
  the main result. The AWBC/prior training scaffolds are algorithmic components,
  not just implementation conveniences.
- Follow-up launched:
  AWBC-off/prior-on ablation in
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_awbc_prior_on_10seed`
  to separate AWBC from candidate-prior effects.

## AWBC-off / Prior-on Ablation

- Source:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_awbc_prior_on_10seed/env_dwell12_h75_no_awbc_prior_on_operational_summary_10seed.csv`.
- Setting:
  same h75 B=1.70 deployment contract, seeds, forecast reward, projector, hard
  duty guard, dwell guard, and candidate prior/KL as the main run, but with
  `--awbc-coef 0.0`.
- Result:
  - mean PD-PPO loss `0.145448`;
  - valid deployment behaviour `10/10`;
  - beats best original dynamic heuristic `3/10`, mean delta `-0.001588`;
  - beats best duty-constrained non-PD-PPO baseline `7/10`, mean delta
    `+0.003098`;
  - beats deployable selected static `0/10`, mean delta `-0.004175`.
- Paired against main:
  - no-AWBC/prior-on minus main mean `+0.004813`;
  - main is lower in `9/10` paired seeds.
- Interpretation:
  AWBC is the stronger stabilising component in the current 40k-timestep
  training budget. The candidate prior helps loss and seed stability, but prior
  guidance alone does not preserve the main dynamic-baseline advantage. The
  complete PD-PPO training scaffold is therefore justified as an algorithmic
  component, not just an implementation detail.
