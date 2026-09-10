# V668 corrected frequency-mode geometry audit

This matched audit reuses the V610 truth, effective resource trace, frozen
forecaster assets, budget, and four chronological starts. The only change is
that the geometry audit stratifies operating states by the truth-only
`resource_frequency_mode_id`, which is the state used to construct the
frequency-coupled resource trace. The mode id is not a scheduler observation.

| seed | condition gap | operating gap | 1% operating intersection |
|---:|---:|---:|---:|
| 7177 | 0.0035718131 | 0.0000000000 | 7 candidates |
| 7178 | 0.0024615728 | 0.0000261984 | 7 candidates |
| 7179 | 0.0000000000 | 0.0000000000 | 2 candidates |
| 7180 | 0.0000000000 | 0.0000000000 | 1 candidate |

The resource trace changes the instantaneous feasible frontier, but the
downstream forecast opportunity does not survive the deployable operating
partition. V668 therefore fails the pre-PPO geometry gate and does not justify
online transfer or policy training. The original V610 operating result is
superseded because it used heater-state labels instead of the explicit
frequency-mode state.
