# V85 calibrated-teacher PD-PPO

- Strongest-static joint wins: 1/5.
- Conventional-dynamic joint wins: 4/5.
- Mean static margins: -0.026645/-0.019071.
- Mean conventional-dynamic margins: +0.012725/+0.098892.
- Behavior remained valid in all seeds.

Replacing the warm-start teacher masks alone did not transfer the calibrated
context policy into final PD-PPO behavior. V86 tests continued retention with
the existing AWBC mechanism at coefficient 0.05.
