# V306 pretraining-only PD-PPO diagnostic

Configuration: 4096-step training-partition soft forecast-value pretraining, `TOTAL_TIMESTEPS=0`; no PPO updates. Scene seeds 6811/6812; policy seeds 6921/6922. This is a diagnostic, not a primary-method result.

## Exact comparison

### ordinary forecast loss

| baseline | wins (custom lower) | mean margin (baseline - custom) |
|---|---:|---:|
| `validation_selected_static` | 0/2 | -0.052098 |
| `feasible_static_projected` | 0/2 | -0.027961 |
| `full_open_unconstrained` | 0/2 | -0.041732 |
| `aoi` | 1/2 | -0.029031 |
| `random` | 1/2 | -0.011544 |
| `round_robin` | 1/2 | -0.006200 |

### static-normalized event macro

| baseline | wins (custom lower) | mean margin (baseline - custom) |
|---|---:|---:|
| `validation_selected_static` | 1/2 | -0.048959 |
| `feasible_static_projected` | 1/2 | +0.036958 |
| `full_open_unconstrained` | 0/2 | -0.039270 |
| `aoi` | 1/2 | -0.015014 |
| `random` | 2/2 | +0.059307 |
| `round_robin` | 1/2 | +0.033462 |

## Behavior

| seed | always-on | always-off | mid-duty | switches/step | aborts |
|---:|---:|---:|---:|---:|---:|
| 6811 | 0 | 0 | 5 | 0.020553 | 0 |
| 6812 | 0 | 1 | 5 | 0.041395 | 0 |

## Decision

Pretraining-only does not recover either ordinary forecast transfer or static-normalized macro transfer. It is rejected as a primary improvement. Seed 6812 also has one always-off channel, so the behavior gate is not clean. The result supports the diagnosis that pretraining can fit oracle actions but does not by itself produce a deployable policy; the next work should not add another pretraining/auxiliary patch without changing the PPO return/credit interface.
