# V94 linear subset representation summary

V94 retains the effective V93 objective, guidance scope, frozen V83a assets,
and policy seeds. The only method change is replacing the nonlinear subset
encoder with a linear additive action representation.

- Strongest-static joint wins: 3/5.
- Conventional-dynamic joint wins: 4/5.
- Complete behavior-gate passes: 3/5.
- Mean strongest-static margins: +0.006727 ordinary and +0.034033 macro.
- Mean conventional-dynamic margins: +0.046097 ordinary and +0.151996 macro.

The additive representation reduces paired performance stability and does not
repair the multi-channel omissions in seeds 1101 and 1105. It is rejected as
the primary representation. V95 restores the nonlinear representation and
executes the originally intended scope test with both AWBC and subtype-action
inclusion restricted to event samples. No architecture, reward, scene, or
policy-seed setting changes in V95.
