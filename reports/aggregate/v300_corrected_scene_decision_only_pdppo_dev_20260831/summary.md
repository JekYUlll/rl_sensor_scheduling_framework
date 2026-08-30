# V300 corrected-scene decision-only policy-update screen

Seeds: `6811`, `6812`. Truth and validation source artifacts were reused from
V294; policy seeds were fresh. This is a bounded development diagnostic, not
independent final evidence.

V300 enabled the existing semi-Markov decision-only policy-update path. Actor
gradients were restricted to epochs where a new mask could be executed; the
critic and reward remained unchanged.

| Comparator | Ordinary-loss wins | Mean margin | Macro wins | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 0/2 | -0.022906 | 1/2 | -0.020483 |
| feasible static projected | 1/2 | +0.001231 | 2/2 | +0.065434 |
| full-open unconstrained | 1/2 | -0.012539 | 1/2 | -0.010794 |
| AoI | 1/2 | +0.000162 | 1/2 | +0.013462 |
| random | 1/2 | +0.017648 | 2/2 | +0.087784 |
| round-robin | 2/2 | +0.022993 | 2/2 | +0.061939 |

Both seeds had zero warm-up aborts and zero always-on channels. Seed 6811 had
one always-off and five mid-duty channels; seed 6812 had no constant channels
and five mid-duty channels. Switching rates were `0.061659` and `0.033435`.

Decision: retain as a method-correction diagnostic, but do not promote as a
complete mainline result. The update-interface correction improves dynamic
learning and feasible-static macro performance, yet ordinary validation-static
transfer remains `0/2`.
