# V95 full event-only guidance summary

V95 corrects the V93 launcher overwrite and restricts both retained AWBC and
subtype-action inclusion supervision to event samples. All other V93 settings,
including the nonlinear subset representation, remain fixed.

- Strongest-static joint wins: 3/5.
- Conventional-dynamic joint wins: 5/5.
- Complete behavior-gate passes: 3/5.
- Mean strongest-static margins: +0.005756 ordinary and +0.057132 macro.
- Mean conventional-dynamic margins: +0.045126 ordinary and +0.175095 macro.

The corrected scope change repairs seed 1101's channel coverage but leaves two
channels unused in seeds 1103 and 1105 and reduces static paired stability.
Guidance-scope tuning is closed. Per-variable loss decomposition shows that the
base objective is dominated by snow-transport targets even in calm periods.
The next stage rebalances the normalized base target objective and screens the
scene with deployable and privileged dynamic policies before any PPO training.
