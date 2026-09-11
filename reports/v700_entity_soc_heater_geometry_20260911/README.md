# V700 Entity SOC and Heater Geometry

V700 extends the corrected V699 cumulative-energy screen with the documented
GMX500 and Parsivel2 heater traces. Heater states use deployment-observable
temperature/dew-point hysteresis rules. Dynamic channel power enters both the
55 W effective resource guard and battery energy consumption. No policy output,
event label, or forecast loss is used to build the trace or select windows.

| Seed | Opportunity gap | 1% static intersection | Conditions | Min--max support rows |
|---:|---:|---:|---:|---:|
| 7177 | 0.002591 | 3 | 7 | 134--2048 |
| 7178 | 0.002090 | 2 | 7 | 94--2048 |
| 7179 | 0.006981 | 0 | 7 | 134--2048 |
| 7180 | 0.023518 | 0 | 7 | 99--2048 |

The resource frontier is state-dependent, but the predeclared downstream gate
requires an opportunity gap above 0.01 across the screened seeds and no
persistent 1% near-optimal static candidate. V700 is therefore closed before
online transfer and PPO. The JSON summaries and per-seed CSV files in this
directory are authoritative.

The audit used the current entity sensor configuration
`configs/sensors/windblown_sensors_entity_six_channel_v1.yaml`; stale channel
identifiers in copied source metadata were corrected before the audit was
accepted. Future asset preparation now synchronizes `sensor_ids` from the
selected sensor configuration.
