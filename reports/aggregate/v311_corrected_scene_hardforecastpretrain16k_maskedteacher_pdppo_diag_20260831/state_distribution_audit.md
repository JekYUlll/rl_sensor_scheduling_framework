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

## Reconstructed teacher-batch distribution

The configured pretraining call uses `event_start_prob=1.0`, episode length
512, and 16,384 steps, so it produces 32 deterministic training episodes.
Reconstructing the exact start-index sampler from the V311 configuration gives
batch event rates `0.707153` (seed 6811) and `0.665161` (seed 6812). The
corresponding mean event-alert rates are `0.320624` and `0.292957`. Thus the
actual teacher batches are substantially more event-heavy than the raw
training-partition averages, and seed 6811's final windows (`0.369792` event
rate) are far less event-heavy than the states used for pretraining. Seed
6812's final windows (`0.704861`) are close to the event-heavy pretraining
distribution. This is stronger evidence for a state-distribution explanation
of the asymmetric V311 transfer, while remaining a deterministic sampler
reconstruction rather than a logged batch trace.

The V311 manifest also records `min_dwell_steps=6` for the custom environment.
Therefore its four-step forecast-value teacher horizon does not cover a full
minimum execution block. This is a separate action-target/closed-loop mismatch
that must be tested before attributing all of the transfer failure to state
distribution.

## Consequence

The next method-consistent diagnostic should log the actual pretraining-state
statistics and test a training-only balanced start distribution using frozen
partition boundaries. No final
test labels or baseline-dependent signal should enter that experiment.

Command used:

```text
python3 - <<'PY' ... truth_v31_split.csv ...
```

The source files were the V311 metadata ranges and the two matched
`truth_v31_split.csv` files. Exact per-seed metrics remain in `seed_metrics.csv`.
