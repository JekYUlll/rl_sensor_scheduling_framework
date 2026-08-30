# V287 stronger soft action-value auxiliary

Configuration: frozen V279 scene/evaluator, `forecast_decision`, decision-only
PPO, soft forecast-derived action-ranking auxiliary with coefficient `1.0`,
temperature `0.5`, stride `8`, and `30,000` requested timesteps (`30,720`
collected per seed). Seeds `6801--6802` ran remotely on GPU1/GPU2.

Ordinary oracle-loss results are in `summary.csv`. Custom PPO beat AoI,
random, and round-robin in both seeds. It beat feasible static only for
seed6801 and validation-selected static in neither seed; it also lost to the
full-open unconstrained reference in both seeds. Mean ordinary margins
(custom minus baseline) were `-0.024527` vs AoI, `-0.025548` vs random,
`-0.033798` vs round-robin, `+0.028833` vs validation-selected static, and
`+0.020034` vs full-open. The static shortcut therefore remains unresolved.

Behavior was valid for warm-up (`0` aborts) and had no always-on sensors.
Seed6801 used six mid-duty sensors with no always-off channels; seed6802 used
four mid-duty and two always-off channels. Switching rates were `0.057606`
and `0.034303` per step.

Decision: retain as a bounded learner-side sensitivity result; do not promote
to the PD-PPO mainline.
