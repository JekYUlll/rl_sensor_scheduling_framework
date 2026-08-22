# V89 V83a entropy correction

V89 retains the V86 calibration-selected subtype teacher, constant AWBC 0.05,
and subtype action CE 0.05, while increasing candidate-action entropy from 0.02
to 0.04. It uses data seeds 1101--1105 and independent policy seeds 2601--2605.

## Result

- Joint wins against the strongest static family: 4/5.
- Joint wins against conventional dynamic policies: 5/5.
- Mean margins against static: +0.005183 ordinary, +0.061899 macro.
- Mean margins against dynamic: +0.044553 ordinary, +0.179862 macro.
- Feasibility and warm-up abort checks pass in all seeds.
- The channel-level behavior gate passes only 3/5 seeds. Seed 1101 has two
  always-off channels and seed 1105 has three; their switch rates are 0.003184
  and 0.005645 per step.

## Decision

V89 is rejected. Higher categorical action entropy preserves V86's favorable
forecast performance but does not prevent channel-level occupancy collapse.
Candidate-action entropy is therefore not an adequate proxy for flexible use of
the physical channels. Further entropy-coefficient tuning is closed.
