# V84 complete PD-PPO on frozen V83a scenes

- Seeds: 1101--1105; independent policy seeds: 2101--2105.
- Strongest-static joint wins: 2/5.
- Conventional-dynamic joint wins: 5/5.
- Mean ordinary/static-normalized macro margins versus static:
  -0.009387/+0.004910.
- Mean ordinary/static-normalized macro margins versus conventional dynamic:
  +0.029983/+0.122873.
- Behavior: zero always-on channels in 5/5, at most one always-off channel,
  nonzero switching, and zero warm-up aborts in all seeds.

The frozen scene passed both online-context and privileged receding gates, but
V84 retained the historical per-subtype static teacher. The next bounded
comparison replaces only that teacher action map with the complete-policy map
selected by constrained calibration replay.
