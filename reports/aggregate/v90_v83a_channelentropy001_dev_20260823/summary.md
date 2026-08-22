# V90 channel-marginal exploration result

V90 adds a training-only channel-marginal entropy coefficient of 0.01 to the
V86 configuration. All scene, evaluator, teacher, reward, feasibility, and
evaluation settings remain frozen.

- Strongest-static joint wins: 3/5.
- Conventional-dynamic joint wins: 5/5.
- Mean static margins: -0.002272 ordinary, +0.036843 macro.
- Mean dynamic margins: +0.037098 ordinary, +0.154806 macro.
- Behavior gate: 2/5. Seed 1103 has one always-on and two always-off channels;
  seeds 1101 and 1105 retain multiple always-off channels.

V90 is rejected. High channel-marginal entropy in the sampled training policy
does not ensure state-dependent channel use under deterministic argmax replay.
No coefficient tuning is authorized. The next diagnostic evaluates the frozen
stochastic policy that PPO actually optimizes, alongside the deterministic
deployment result.
