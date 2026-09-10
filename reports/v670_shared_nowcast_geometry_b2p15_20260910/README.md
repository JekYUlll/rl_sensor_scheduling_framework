# V670 shared-nowcast geometry closeout

V670 uses the corrected V669 shared-nowcast truth, four regenerated frozen
TCN assets, budget `2.15`, and starts `82600, 83000, 84800, 85500`.

| seed | condition gap | operating states in final windows | operating gap | 1% operating intersection |
|---:|---:|---|---:|---:|
| 7177 | 0.0001816 | `heater_11000` | 0 | 7 candidates |
| 7178 | 0.0000209 | `heater_11000` | 0 | 7 candidates |
| 7179 | 0 | `heater_11000` | 0 | 7 candidates |
| 7180 | 0.0012399 | `heater_11000` | 0 | 3 candidates |

The complete final partition is dominated by simultaneous core and laser
heating. The resource trace has multiple states over the full truth sequence,
but those states are absent from the fixed final evaluation windows. V670 is
therefore closed before executable transfer and PPO. Condition-only gaps are
not promoted as deployable evidence.
