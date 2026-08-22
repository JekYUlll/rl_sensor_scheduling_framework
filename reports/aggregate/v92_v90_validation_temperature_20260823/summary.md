# V92 validation-selected execution temperature

Each frozen V90 checkpoint selected among temperatures 0, 0.05, 0.1, 0.2, and
0.5 using only validation replay. Temperature zero is deterministic argmax.

- Selected temperatures: 0, 0, 0.2, 0.05, and 0 for seeds 1101--1105.
- Strongest-static joint wins: 3/5.
- Conventional-dynamic joint wins: 5/5.
- Behavior gate: 2/5.
- Mean static margins: -0.002748 ordinary and +0.034970 macro.

Validation selection correctly rejects sampling for most scenes, but the two
nonzero temperatures do not improve the aggregate behavior gate. Execution
temperature is closed as a remedy. The next training comparison limits retained
teacher guidance to event windows, leaving calm and transition decisions to the
forecast-driven PPO objective.
