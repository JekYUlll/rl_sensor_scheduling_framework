# V311 train/evaluation state-distribution audit

This audit compares the configured policy-training partition `[12600, 30081]`
with the six final evaluation windows (384 steps per window). It is a
partition-level diagnostic; it does not claim to reconstruct the exact
pretraining batch composition.

| Scene seed | Split | n | blowing-snow | particle subtype | flux subtype | thermal subtype | event alert | mean mass flux |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6811 | training partition | 17482 | 0.537582 | 0.194257 | 0.193342 | 0.149983 | 0.248298 | 3.5e-05 |
| 6811 | final windows | 2304 | 0.369792 | 0.134115 | 0.134983 | 0.100694 | 0.181046 | 1.3e-05 |
| 6812 | training partition | 17482 | 0.502860 | 0.169889 | 0.186764 | 0.146208 | 0.230302 | 2.2e-05 |
| 6812 | final windows | 2304 | 0.704861 | 0.232639 | 0.297743 | 0.174479 | 0.329273 | 3.4e-05 |

The final windows therefore move in opposite directions relative to the
training partition: scene 6811 is less event-heavy, whereas scene 6812 is
more event-heavy. Mean event-alert prevalence shifts from `0.248298` to
`0.181046` for 6811 and from `0.230302` to `0.329273` for 6812. The same
pattern is visible in the subtype proportions and mass flux. This provides a
concrete state-distribution-shift hypothesis for V311's seed divergence, but
does not by itself prove that the exact optimizer failure is caused by this
shift.

## Consequence

The next method-consistent diagnostic should log the actual pretraining-state
statistics and compare them with held-out states, then test a training-only
stratified start distribution using frozen partition boundaries. No final
test labels or baseline-dependent signal should enter that experiment.

Command used:

```text
python3 - <<'PY' ... truth_v31_split.csv ...
```

The source files were the V311 metadata ranges and the two matched
`truth_v31_split.csv` files. Exact per-seed metrics remain in `seed_metrics.csv`.
