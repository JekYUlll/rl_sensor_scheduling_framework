# V2 Supplement Experiment Report - 2026-05-13

## 1. Execution Summary

The supplementary experiment batch was completed on the GPU server after switching from a serial runner to a conservative 3-GPU parallel runner under the server power limits.

- Server: `remote-gpu`
- GPU limits during run: `nvidia-smi -pl 150`, graphics clock capped to `300,1200`
- Parallel runner: `scripts/33_v2_run_supplement_parallel.py`
- Main tmux session: `v2_supp_parallel_20260513`
- Main log: `reports/logs/v2_supp_parallel_20260513.log`
- Per-task logs: `reports/logs/v2_supplement_parallel/`
- Main model: PD-PPO / `custom_ppo`
- Forecast oracle: TCN oracle
- Main budgets: `1.65`, `1.70`, `1.75`
- Seeds: `41-50`
- Training length per S1 run: `100000` timesteps
- Truth length: `8192`
- Startup peak budget: `3.2`

The completed batch contains:

- S1 main grid: `3 budgets x 10 seeds`
- A2 staged diagnostic: `4 stages x 5 seeds`
- E1 condition evaluation: `calm/mixed/event x 30 trained runs`
- A1 full ablation: not run in this batch
- H1 hyperparameter sensitivity: not run in this batch

## 2. Synced Local Artifacts

The following results have been synced back to the local workstation.

- Summary tables and figures: `reports/v2_supplement_assets/`
- Raw S1 lightweight CSV/JSON outputs: `reports/v2_forecast_eval_grid_prior_kl1/`
- Raw A2/E1 lightweight CSV/JSON outputs: `reports/v2_supplement_experiments/`
- Main and per-task logs: `reports/logs/v2_supp_parallel_20260513.log`, `reports/logs/v2_supplement_parallel/`

Large model checkpoints and rollout arrays were intentionally not pulled in full. The synced files are enough for statistical summaries, paper plots, and metric-level debugging.

## 3. Aggregation Fix

The first remote aggregation accidentally used the stale `reports/v2_paper_tables_prior_kl1/overall_long.csv`, which only contained three seeds per budget. This produced an incorrect `exp_s1_main_stats.csv` with `n=3`.

The aggregation logic in `scripts/31_v2_build_supplement_assets.py` was fixed so that, when raw grid outputs contain more filtered `(budget, seed, policy)` rows than the stale paper table, the script uses the raw grid instead. The corrected local and remote `reports/v2_supplement_assets/` now report `n=10` for S1.

## 4. S1 Main Result

Metric: `forecast_weighted_mae_overall`, lower is better.

| Budget | Full observation | Static projection | PD-PPO | Round-robin | AoI | Random |
|---:|---:|---:|---:|---:|---:|---:|
| 1.65 | 0.3685 | 0.3906 | 0.3979 | 0.4057 | 0.4136 | 0.4300 |
| 1.70 | 0.3689 | 0.3912 | 0.3911 | 0.4041 | 0.4244 | 0.4321 |
| 1.75 | 0.3691 | 0.3896 | 0.3907 | 0.4042 | 0.4304 | 0.4364 |

Interpretation:

- Full observation remains the expected unconstrained upper bound.
- PD-PPO is consistently better than random, AoI, and round-robin across all three budgets.
- PD-PPO is very close to static projection at budgets `1.70` and `1.75`; the difference is not statistically significant there.
- At the tightest budget `1.65`, static projection is still significantly better than PD-PPO.
- PD-PPO improves over round-robin by about `1.9%`, `3.2%`, and `3.3%` at budgets `1.65`, `1.70`, and `1.75`.
- PD-PPO improves over random by about `7.5%`, `9.5%`, and `10.5%` at budgets `1.65`, `1.70`, and `1.75`.

Statistical notes:

- Bonferroni alpha in the generated table: `0.00833`.
- PD-PPO is significantly better than random at all budgets.
- PD-PPO is significantly better than round-robin at `1.70` and `1.75`, but not at `1.65`.
- PD-PPO is significantly better than AoI at `1.70` and `1.75`, while `1.65` is close but not Bonferroni-significant.
- Full observation is significantly better than PD-PPO at all budgets, as expected.

## 5. A2 Component Diagnostic

A2 was run at budget `1.70` with seeds `41-45`.

| Stage | Mean FW-MAE | Std | N | Interpretation |
|---|---:|---:|---:|---|
| D1 MaskedActor + ActionEmbedding | 0.4224 | 0.0326 | 5 | basic feasible action policy |
| D2 + EventAwareCritic | 0.4219 | 0.0412 | 5 | almost unchanged |
| D3 + AWBC | 0.4080 | 0.0284 | 5 | clear improvement |
| D4 + oracle prior | 0.3942 | 0.0384 | 5 | strongest staged variant |

Interpretation:

- EventAwareCritic alone contributes little in this setup.
- AWBC provides a meaningful improvement over the masked actor baseline.
- The oracle-calibrated prior provides an additional meaningful improvement.
- The staged diagnostic supports the current PD-PPO design story: the strongest gains come from warmup-aware imitation and oracle-calibrated cold-start stabilization, while event-aware value conditioning is weaker by itself.

## 6. E1 Condition Evaluation

Metric: `forecast_weighted_mae_overall`, lower is better. Each condition table has `n=30`.

| Condition | Full observation | Static projection | PD-PPO | Round-robin | AoI | Random |
|---|---:|---:|---:|---:|---:|---:|
| calm | 0.3448 | 0.3643 | 0.3669 | 0.3816 | 0.3939 | 0.4054 |
| mixed | 0.3681 | 0.3907 | 0.3941 | 0.4108 | 0.4239 | 0.4378 |
| event | 0.3681 | 0.3907 | 0.3941 | 0.4108 | 0.4239 | 0.4378 |

Interpretation:

- PD-PPO remains better than round-robin, AoI, and random in both calm and event-heavy settings.
- PD-PPO remains very close to static projection, but does not surpass it.
- Full observation remains best, consistent with its role as an unconstrained upper bound.

Important caveat:

The `mixed` and `event` rows are identical. Inspecting metadata for `budget=1.70, seed=41` shows that mixed and event selected the same start indices:

- `mixed`: `[0, 4096, 4864, 5376, 5632, 6144]`
- `event`: `[0, 4096, 4864, 5376, 5632, 6144]`

This means E1 currently validates a calm-vs-event-heavy contrast, but it does not provide a clean independent mixed-regime conclusion. The condition sampler should be revised before using E1 as a main paper result.

## 7. Current Scientific Takeaway

The new results are much more aligned with the intended paper narrative than earlier runs:

- Full observation is the best unconstrained reference.
- PD-PPO is the best learned/adaptive policy among the practical schedulers evaluated here.
- PD-PPO consistently beats random, AoI, and round-robin under power constraints.
- PD-PPO approaches the static projection baseline at moderate budgets.
- The component diagnostic gives a plausible mechanism for PD-PPO's advantage: AWBC and oracle prior matter most.

However, the result should be worded carefully:

- We should not claim that PD-PPO beats every baseline if static projection is included as a baseline.
- A safer and more accurate claim is that PD-PPO outperforms standard heuristic schedulers and approaches the best static feasible projection.
- Static projection should be described as a strong static feasible reference, not as a naive heuristic.

## 8. Recommended Next Experiments

1. Fix E1 condition sampling.

   Require non-overlapping calm, mixed, and event start pools. For example:
   - calm: event fraction `< 0.20`
   - mixed: event fraction in `[0.35, 0.65]`
   - event: event fraction `> 0.75`
   If the current truth length cannot supply enough mixed/event windows, increase truth length or generate event-balanced validation sequences.

2. Run a focused A1 ablation, not the full expensive grid.

   Suggested budget: `1.70`.
   Suggested seeds: `41-45`.
   Variants:
   - Full PD-PPO
   - minus AWBC
   - minus oracle prior
   - minus action mask
   - minus action embedding
   - minus event critic

3. Run a small H1 sensitivity sweep only around the influential components.

   Suggested:
   - `awbc_coef`: `0.05, 0.10, 0.20`
   - `prior_kl_coef`: `0.5, 1.0, 2.0`
   - budget fixed at `1.70`
   - seeds `41-45`

4. Decide whether static projection belongs in the main comparison table.

   If included, present it as a strong static reference. If the paper's main message is adaptive policy optimization, the primary heuristic comparison should focus on round-robin, AoI, and random, with static projection as an upper static reference.

5. Consider lowering compute further for future reruns.

   With the server power instability, `3` GPUs under the `150W` cap was stable enough for this batch. Avoid using all six GPUs unless the machine room power stabilizes.
