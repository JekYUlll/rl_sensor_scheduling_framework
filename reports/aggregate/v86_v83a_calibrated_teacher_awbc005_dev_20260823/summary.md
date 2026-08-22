# V86 calibrated teacher with constant AWBC 0.05

- Strongest-static joint wins: 4/5.
- Conventional-dynamic joint wins: 5/5.
- Mean static margins: +0.008486/+0.066459.
- Mean conventional-dynamic margins: +0.047856/+0.184422.
- Zero aborts and no always-on channels, but seeds 1101 and 1105 had two and
  three always-off channels respectively.

Constant AWBC repairs guidance retention but over-constrains channel diversity.
V87 linearly decays the same coefficient to zero over training.
