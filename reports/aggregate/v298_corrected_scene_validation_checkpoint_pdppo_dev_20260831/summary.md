# V298 corrected-scene validation-checkpoint development screen

Seeds: `6811`, `6812`. The policy seeds were fresh, but the validation source
directories and truth sequences were reused from V294 for this bounded
diagnostic; this is not an independent final evaluation.

## Result

The checkpoint-selection path was active: each run produced ten validation
checkpoint rows from the supplied V294 `validation_static_candidates.csv`.
Nevertheless, the selected policies did not recover the static-shortcut
transfer observed to be missing in V294.

| Comparator | Ordinary-loss wins | Mean margin (positive is better) | Macro wins | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 1/2 | -0.033568 | 1/2 | -0.055507 |
| feasible static projected | 1/2 | -0.009432 | 1/2 | +0.030410 |
| full-open unconstrained | 0/2 | -0.023202 | 0/2 | -0.045818 |
| AoI | 1/2 | -0.010501 | 1/2 | -0.021562 |
| random | 1/2 | +0.006986 | 2/2 | +0.052759 |
| round-robin | 1/2 | +0.012330 | 1/2 | +0.026914 |

## Operational audit

Both seeds had zero warm-up aborts. Seed 6811 had zero always-on, one
always-off, and four mid-duty channels; seed 6812 had zero always-on, one
always-off, and five mid-duty channels. Switching rates were `0.021566` and
`0.034882` per step. The behavior gate therefore passed except for the
remaining single always-off channel in each selected policy.

## Decision

Reject V298 as a mainline improvement. Validation checkpoint selection was
actually executed, but it did not improve ordinary static transfer or recover
the missing dynamic advantage. The next design decision should address the
training/evaluation objective or state distribution, not add another
checkpoint-selection wrapper.
