# V285 action-aligned dwell-block reward

Remote development run, seeds 6801--6802. The reward at each executable
decision was the negative mean frozen-forecaster loss over the selected
six-step dwell block. Margins below are custom PPO minus comparator, so
negative values favor custom PPO.

| Comparator | Seed 6801 | Seed 6802 | Wins |
|---|---:|---:|---:|
| Validation-selected static | +0.075184 | +0.057762 | 0/2 |
| Feasible static | +0.022376 | +0.058663 | 0/2 |
| AoI | +0.007265 | +0.018961 | 0/2 |
| Random | +0.005677 | +0.018508 | 0/2 |
| Round-robin | -0.007689 | +0.015374 | 1/2 |
| Full-open unconstrained | +0.032143 | +0.083205 | 0/2 |

Both seeds had zero warm-up aborts and zero always-on channels. Seed6801 had
six mid-duty channels; seed6802 had four. The block-aligned reward did not
recover predictive transfer and is not promoted to the mainline.
