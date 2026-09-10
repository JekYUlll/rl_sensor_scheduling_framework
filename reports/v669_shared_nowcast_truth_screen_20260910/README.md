# V669 shared-nowcast truth-only screen

V669 is the next causal-scene candidate after the corrected V668 closeout. It
uses the existing weather backbone and derives three bounded drivers from
decision-time wind, humidity, temperature, and radiation nowcasts. The same
drivers are used for a six-step-lagged target relation, specialist quality,
and the subsequent documented heater controller. Exact mode labels are not
created or exposed to the scheduler.

The local truth-only screen passed the pre-asset support checks for seeds
7177--7180:

| check | result |
|---|---|
| dominant driver support | transport 28.5--28.9%, particle 29.2--29.4%, thermal 32.0--32.2% |
| resource heater duty | core 69.3--69.6%, laser 84.8--85.1% |
| optional feasible masks per row | 11--16 of 32 |
| always-feasible masks | 11 of 32 for every seed |
| changed frontier fraction | 84.8--85.1% of rows |

This is only a truth/resource occupancy screen. No frozen forecaster, policy,
online transfer result, or PPO checkpoint has been generated from V669.
The next gate is complete-subset forecast geometry under the generated trace,
followed by chronological executable transfer.
