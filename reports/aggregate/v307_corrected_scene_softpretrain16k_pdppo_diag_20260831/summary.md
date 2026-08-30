# V307 corrected-scene 16k pretraining diagnostic

- Run: 16,384 soft forecast-value pretraining, `TOTAL_TIMESTEPS=0` (no PPO updates).
- Scene seeds: 6811 and 6812; policy seeds: 6931 and 6932.
- Purpose: test whether greater teacher-label coverage transfers to deployable forecast scheduling.

## Exact result
- Seed 6811: custom PPO ordinary loss `0.452047919`, static-normalized macro `0.907622531`; behavior: abort `0`, always-on `0`, always-off `0`, mid-duty `6`, switches/step `0.018960776`.
- Seed 6812: custom PPO ordinary loss `0.579324141`, static-normalized macro `1.327050296`; behavior: abort `0`, always-on `0`, always-off `0`, mid-duty `5`, switches/step `0.014908091`.

## Win counts (lower loss is better)
- `validation_selected_static`: ordinary 0/2, mean margin (baseline - custom) `-0.082863868`; macro 0/2, mean margin `-0.214862281`.
- `feasible_static_projected`: ordinary 0/2, mean margin (baseline - custom) `-0.058727491`; macro 0/2, mean margin `-0.128945788`.
- `aoi`: ordinary 0/2, mean margin (baseline - custom) `-0.059796739`; macro 0/2, mean margin `-0.180917374`.
- `random`: ordinary 0/2, mean margin (baseline - custom) `-0.042309889`; macro 0/2, mean margin `-0.106596019`.
- `round_robin`: ordinary 0/2, mean margin (baseline - custom) `-0.036965691`; macro 0/2, mean margin `-0.132441143`.
- `full_open_unconstrained`: ordinary 0/2, mean margin (baseline - custom) `-0.072497747`; macro 0/2, mean margin `-0.205173262`.

## Decision
- **Rejected as a primary improvement.** The 16k pretraining coverage increased action-label accuracy to about 0.83 but did not transfer to forecast quality: ordinary loss lost to validation-selected static in 2/2 scenes and to every tested dynamic baseline in seed6811; the two-scene mean was worse than static and dynamic references.
- Behavior feasibility itself was clean in both scenes (zero aborts), but seed6811 had two always-on and four always-off channels and seed6812 had two always-on and four always-off channels for the static references; custom PPO had no always-on/off channels in seed6811 and seed6812, with five/six mid-duty channels. This does not rescue the predictive failure.
- Do not expand pretraining again. The remaining method-consistent investigation is return/credit or state-distribution alignment, not more teacher coverage or bandit-dependent components.
