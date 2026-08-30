# V286 soft forecast action-value auxiliary

Remote development run, seeds 6801--6802. The auxiliary used frozen
forecaster candidate costs at genuine decision rows only. Margins below are
custom PPO minus comparator; negative values favor custom PPO.

| Comparator | Seed 6801 | Seed 6802 | Wins |
|---|---:|---:|---:|
| Validation-selected static | +0.060065 | +0.022030 | 0/2 |
| Feasible static | +0.007257 | +0.022931 | 0/2 |
| AoI | -0.007853 | -0.016771 | 2/2 |
| Random | -0.009441 | -0.017224 | 2/2 |
| Round-robin | -0.022807 | -0.020358 | 2/2 |
| Full-open unconstrained | +0.017025 | +0.047473 | 0/2 |

Both seeds had zero warm-up aborts, zero always-on/always-off sensors, and six
mid-duty sensors. The auxiliary improves the conventional dynamic comparison
but leaves a clear static-shortcut deficit, so it is not promoted to the
mainline.
