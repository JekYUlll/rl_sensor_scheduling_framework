# Entity Supply and Heater Field Validation (2026-09-10)

## Scope

This is a hardware-log validation, not an Antarctic deployment trace and not
a policy experiment. The source is the room-temperature/freezer test archive
under `/home/horeb/_Data/SEUAWS/rs485-reader/`.

## Observed fields

The synchronized test logs cover approximately `115.09 h` from 2026-05-24 to
2026-05-29.

| Source | Field | Result |
|---|---|---:|
| Modbus weather-station log | `Batt_volt_Min` | 13,920 valid rows; 12.6829 V mean; 12.0--12.7113 V observed |
| Parsivel2 status log | `supply_voltage` | 11.9--12.1 V; 12.0328 V mean |
| Parsivel2 status log | `heating_current` | 19--27 (field units); 24.7966 mean |
| Parsivel2 status log | `heating_state` | 17,646/17,720 rows at state 0; 72 at state 3; 2 at state 2 |

The logs verify that supply voltage and instrument heating telemetry are
available in the real acquisition path. They do not provide a complete system
current measurement, photovoltaic charging trace, battery state model, or
Antarctic operating distribution.

## Consequence for the flexible-subset scene

These records may calibrate device-voltage and heater-status representations
and support a later hardware-validation appendix. They cannot set
`harvest_per_step`, battery capacity, or a state-dependent SOC constraint for
the synthetic Antarctic scene. The rare nonzero heating states do not justify
extrapolating a persistent 50--100 W heater load.

The mainline remains blocked from SOC-based PPO. A valid SOC extension needs
system-level current or power, a documented supply/storage configuration, and
a longer trace whose temporal distribution is appropriate for the simulated
deployment. Until then, use the hardware manifest for instantaneous resource
geometry and keep this log as independent field validation.
