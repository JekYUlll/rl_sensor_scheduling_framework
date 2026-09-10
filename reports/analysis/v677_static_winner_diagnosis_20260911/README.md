# V677 static-winner diagnosis

## Purpose

Explain why the V676 lower-budget probe cannot test the intended arbitrary-
subset problem. This is a read-only analysis of the frozen V675/V676 geometry
artifacts; no truth, forecaster, or policy was changed.

## Findings

The normalized budget `1.45` admits the mandatory backbone and individual
optional channels, but not a specialist pair under the declared startup
costs. In the V676 power rows:

| Candidate | Optional channels | startup cost |
|---|---|---:|
| candidate_002 | radiometer | 1.2104 |
| candidate_004 | surface-temperature IR | 1.3304 |
| candidate_008 | laser disdrometer | 1.4904 |
| candidate_016 | FC4 flux | 1.3904 |
| candidate_006 | radiometer + IR | 2.1304 |
| candidate_012 | IR + laser | 2.4104 |
| candidate_020 | IR + FC4 | 2.3104 |
| candidate_024 | laser + FC4 | 2.4704 |

Consequently, `B=1.45` is a one-optional-channel geometry, despite being
represented as a 32-subset action space. The V676 operating winners are also
unchanged across heater states for seeds 7178 and 7179, and the four operating
gaps are all below `0.001`.

At the unchanged V675 budget `2.15`, the three specialist pair startup costs
are still above budget (`2.3104`, `2.4104`, and `2.4704`). A predeclared budget
near `2.50` would be the first resource regime that admits all three specialist
pairs while excluding their three-channel union, whose startup cost is above
`4.0` in the current manifest. This is a geometry hypothesis, not a result;
it must first pass the same four-seed frozen forecast audit.

## Decision

Do not train PPO from V676. Prepare a single binding-pair geometry probe at
normalized budget `2.50`, reusing the V675 truth and frozen evaluator assets.
The probe is justified by the declared hardware cost thresholds, not by a
selected policy result. It will be closed before transfer if any seed retains
a material static shortcut or fails the operating opportunity gate.
