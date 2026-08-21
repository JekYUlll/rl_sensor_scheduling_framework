# Response Analysis for `docs/05-13-01.md`

## 1. Main Judgment

The review notes are directionally useful, but one premise needs correction:

The current V2 public-weather generator is **not** an independent sampler. It already uses DFT phase randomization for the AntAWS-anchored base variables in `src/data_sources/public_weather_synthesis.py`. The more precise problem is that the later blowing-snow event construction forcibly modifies `wind_speed_ms` through event floors and precursors. This can break the DFT-preserved marginal/PSD/ACF properties that were initially produced for wind speed.

Therefore, the next step should not be "add DFT" from scratch. The next step should be:

1. Preserve the DFT/AntAWS base process more faithfully.
2. Generate blowing-snow events as a conditional process derived from wind-speed persistence, not by arbitrary independent interval injection.
3. Validate event-conditioned temporal structure, not only base-variable ACF.

## 2. Highest-Priority Experiments to Add

### E1-fix: Condition-Stratified Evaluation

Current issue:

The existing E1 condition sampler selected identical start indices for `mixed` and `event`, so those rows are not independent evidence.

Required fix:

- Update `scripts/32_v2_condition_eval.py`.
- Enforce disjoint start pools:
  - calm: event fraction `< 0.20`
  - mixed: event fraction in `[0.35, 0.65]`
  - event-heavy: event fraction `> 0.75`
- If a pool has too few candidates, the script should fail loudly or write an explicit warning instead of silently reusing the nearest windows.
- Reduce stride if needed, e.g. from `steps // 4` to `steps // 8`, to increase candidate windows.

Run cost:

No retraining is needed. Reuse existing trained PD-PPO checkpoints and rerun condition evaluation only.

Why it matters:

This is the fastest way to repair a real weakness in the current supplementary results.

### E2: Oracle Robustness Under Partial Observation

Reviewer concern:

The TCN oracle is used as the reward source under partial observations. We need to show that this reward signal remains meaningful when the scheduler observes only a subset of sensors.

Recommended experiment:

- Evaluate the trained TCN oracle under controlled observation subsets.
- For `k = 1, 2, 3, 4, 5` active sensors:
  - enumerate all feasible subsets if count is small;
  - otherwise sample a fixed number of subsets;
  - replay each subset over the same truth windows;
  - report FW-MAE mean/std.
- Also report:
  - full observation;
  - random feasible subset;
  - static projection;
  - PD-PPO.

Acceptance criterion:

The oracle should degrade smoothly as active-channel count decreases. The reward must not collapse to a near-constant signal across feasible subsets.

Why it matters:

This directly validates the reward mechanism and is more convincing than only citing the gap between `full_open` and `random`.

### P1: Physical-Unit Error Table

Reviewer concern:

FW-MAE needs physical interpretation.

Recommended experiment/analysis:

- Do not assume FW-MAE equals degrees Celsius.
- Extract or compute per-target raw MAE from existing `v2_eval_by_variable.csv`.
- Report raw MAE for at least:
  - `air_temperature_c`;
  - `wind_speed_ms`;
  - `snow_mass_flux_kg_m2_s`.
- Provide a small table mapping normalized FW-MAE to physical-unit raw MAE.

Why it matters:

This prevents a fragile claim such as "0.391 FW-MAE = 0.391 C" if the metric is normalized or target-weighted.

## 3. Medium-Priority Experiments

### A1-focused: Component Ablation

The A2 staged diagnostic already shows the likely story:

- EventAwareCritic alone is weak.
- AWBC matters.
- Oracle prior matters.

A full A1 grid is expensive and not immediately necessary. If run, make it focused:

- budget: `1.70`
- seeds: `41-45`
- variants:
  - full PD-PPO
  - minus AWBC
  - minus oracle prior
  - minus action mask
  - minus action embedding
  - minus event-aware critic

### H1-small: Sensitivity Around Influential Components

Only sweep the components that A2 says matter:

- `awbc_coef`: `0.05, 0.10, 0.20`
- `prior_kl_coef`: `0.5, 1.0, 2.0`
- budget: `1.70`
- seeds: `41-45`

Do not run broad H1 until E1 and E2 are fixed.

## 4. Simulation Generator Changes

### 4.1 Current Generator Behavior

Current file:

- `src/data_sources/public_weather_synthesis.py`

Current base-variable path:

- load AntAWS station records;
- regularize;
- apply DFT phase randomization;
- optionally match empirical distributions.

Current event path:

- create clustered event intervals independently of the wind process;
- force wind-speed precursors and event floors;
- generate flux/diameter/velocity from the modified wind speed.

The weak point is not the initial DFT synthesis. The weak point is the event override after DFT synthesis.

### 4.2 Recommended V3 Generator Design

Replace "event intervals overwrite wind" with "events emerge from persistent wind regimes".

Proposed design:

1. Generate AntAWS-anchored base variables with DFT phase randomization.
2. Generate a smooth storm-regime latent variable `r_t` using an AR(1), semi-Markov, or thresholded DFT-derived process.
3. Modulate wind speed with a smooth additive storm anomaly:
   - no hard discontinuous event floor unless absolutely necessary;
   - if distribution shifts, re-apply quantile mapping or validate against AntAWS after modulation.
4. Define blowing-snow event probability using a CRED-style function:
   - `p_event = sigmoid((u_t - u0) / tau)` or hysteresis around `8 m/s` and `12 m/s`;
   - optionally require minimum duration after threshold crossing.
5. Generate blowing-snow variables conditionally:
   - flux: `a * max(u_t - u_thr, 0)^alpha * lognormal_AR_noise`;
   - diameter: negatively correlated with wind speed;
   - particle velocity: positively correlated with wind speed;
   - inactive periods: flux `0`, particle variables `0` plus explicit availability/event masks.
6. Validate:
   - base-variable KS/PSD/ACF;
   - event fraction;
   - event duration distribution;
   - event gap distribution;
   - conditional wind distribution during event/non-event;
   - flux-vs-wind log-log slope;
   - particle diameter/velocity conditional correlations.

### 4.3 Validation Figure 3 Should Change

Current Figure 3 should not only show base marginal/ACF checks. It should include:

- base variable distribution comparison;
- wind-speed ACF/PSD;
- event duration histogram;
- log-log flux versus wind-speed relationship;
- condition-wise event/non-event wind distribution.

This makes the dataset claim defensible to Cold Regions S&T readers.

## 5. What Not to Claim Yet

Avoid the following until the corresponding checks are done:

- "FW-MAE of 0.391 corresponds to 0.391 C" unless raw physical-unit MAE confirms it.
- "mixed and event-heavy E1 both support the claim" because current mixed/event windows overlap.
- "DFT guarantees all synthetic-process fidelity" because event modification can break the DFT-preserved wind structure.
- "EventAwareCritic is strongly responsible for gains" because A2 shows it contributes little by itself.

## 6. Recommended Next Implementation Order

1. Fix E1 condition sampler and rerun E1 only.
2. Add E2 oracle robustness evaluation.
3. Add raw physical-unit metric extraction/table.
4. Patch generator validation so event-conditioned statistics are reported.
5. Redesign V3 event generation so events emerge from wind regimes rather than overwriting wind.
6. Run V3 dataset-fidelity smoke tests.
7. Only after validation passes, rerun S1 on V3 data.
8. Then run focused A1 and H1 if still needed.

## 7. Expected Paper Impact

For the current V2 paper text:

- Use S1 and A2 confidently.
- Use E1 only after sampler repair.
- Present static projection as a strong static reference, not a naive baseline.
- State that PD-PPO beats standard heuristics and approaches static projection.

For V3:

- The stronger scientific story should shift from "RL beats heuristics on a synthetic benchmark" to "prediction-driven scheduling remains effective under AntAWS-anchored, event-conditioned temporal dynamics."

