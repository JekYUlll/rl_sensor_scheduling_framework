# V284 learner diagnostics

Remote matched development run, seeds 6801--6802. The diagnostic source was
the decision-only PPO update at genuine `decision_available` rows. Values are
ordinary oracle-loss margins, computed as custom PPO minus the comparator;
negative is better for custom PPO.

| Comparator | Seed 6801 | Seed 6802 | Wins |
|---|---:|---:|---:|
| AoI | +0.002172 | -0.038243 | 1/2 |
| Random | +0.000584 | -0.038696 | 1/2 |
| Round-robin | -0.012782 | -0.041830 | 2/2 |
| Validation-selected static | +0.070090 | +0.000559 | 2/2 |
| Feasible static | +0.017282 | +0.001460 | 2/2 |
| Full-open unconstrained | +0.027050 | +0.026002 | 2/2 |

The custom policy had zero warm-up aborts and zero always-on sensors in both
seeds. It used six mid-duty sensors in seed6801 and five mid-duty plus one
always-off sensor in seed6802. V284 does not support promotion of the learner
configuration to the mainline.
