# V302 validation-frozen subtype-normalized reward result

## Configuration

- Corrected six-channel quality scene, budget 1.75, startup peak budget 2.15.
- Development seeds 6811 and 6812; policy seeds 6881 and 6882.
- V300 decision-only actor updates retained.
- Forecast-loss reward was divided by subtype medians computed from the
  control-source validation candidate set. No bandit-dependent objective or
  test-time label was introduced.

## Outcome

V302 is retained as the strongest current development candidate, but it does
not pass the static-transfer gate. Ordinary-loss wins were validation static
`0/2`, feasible static `1/2`, full-open `1/2`, AoI `1/2`, random `2/2`, and
round-robin `2/2`. Mean ordinary margins were `-0.022041`, `+0.002096`,
`-0.011675`, `+0.001026`, `+0.018513`, and `+0.023857`, respectively.

Static-normalized macro wins were validation static `1/2`, feasible static
`2/2`, full-open `1/2`, AoI `1/2`, random `2/2`, and round-robin `2/2`.
Mean macro margins were `-0.013303`, `+0.072613`, `-0.003614`, `+0.020642`,
`+0.094963`, and `+0.069118`, respectively.

## Behavior

Both seeds had zero warm-up aborts and zero always-on channels. Seed 6811 had
one always-off channel and three mid-duty channels; seed 6812 had no always-off
channels and five mid-duty channels. Switching rates were `0.017224` and
`0.039224` per channel-step.

## Interpretation

Validation-frozen subtype scaling materially improves the dynamic and
feasible-static comparisons, supporting reward-scale conditioning as a real
training issue. It does not yet remove the static shortcut or guarantee all
channels remain active. The next step is a bounded confirmation using this
candidate and an explicit behavior audit, not a claim of final superiority.

## Provenance

- Raw runs: `reports/v302_corrected_scene_decision_only_staticnorm_pdppo_dev_seed6811_b1p75_20260822/`
  and `reports/v302_corrected_scene_decision_only_staticnorm_pdppo_dev_seed6812_b1p75_20260822/`.
- Aggregate source: `seed_metrics.csv` in this directory.
- Remote launcher: `scripts/run_v302_corrected_scene_decision_only_staticnorm_pdppo_dev_20260831.sh`.
