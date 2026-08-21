# 2026-05-22 Energy-Account Design Gate

## Why this is needed

The v4 saturation diagnostics repair the upstream sensor-value problem: the laser disdrometer has positive event-conditioned oracle value under a physically motivated dense-event saturation model for the low-cost snow particle counter.

However, the fixed per-step budget still prefers a static high-value subset at `B=1.20`:

- best overall: `met_station_core|surface_temp_ir|laser_disdrometer`
- `laser_event_lift = +0.0187`
- dynamic hand-written schedules remain worse than the best fixed laser subset

This means a fixed per-step pilot would likely demonstrate "laser is useful" rather than "scheduling is useful". To make event-conditioned scheduling identifiable, always-on laser must be feasible instantaneously but unattractive or infeasible over the episode energy account.

## Minimal model

Keep the existing instantaneous constraints:

- `startup_peak_budget`: safety cap for cold starts
- optional `per_step_budget`: maximum simultaneous load
- `max_active` and required sensors

Add a separate energy account:

- `energy_capacity`: maximum stored normalized energy
- `initial_energy`: initial SOC in normalized energy units
- `harvest_per_step`: scalar or generated time series
- `reserve_energy`: minimum SOC reserve
- `energy_cost = steady_power * dt`
- `soc_next = min(capacity, soc + harvest_per_step - energy_cost)`

The deployment interpretation is normalized scheduling energy, not measured watt-hours unless calibrated later.

## Gate behavior

For the first implementation, avoid changing the fixed-budget experiment silently.

Create a new environment/config mode with:

- SOC appended to the agent state
- `energy_remaining_ratio` and `recent_energy_deficit` in rollout info
- hard rejection or large penalty for actions that would push SOC below reserve
- metrics for `soc_min`, `energy_deficit_steps`, `energy_used`, and `harvest_used`

Recommended first policy diagnostic before PPO:

- calm mask: `met_station_core|radiometer_basic|surface_temp_ir`
- event mask: `met_station_core|surface_temp_ir|laser_disdrometer`
- static laser mask: `met_station_core|surface_temp_ir|laser_disdrometer`
- static snow-counter mask: `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`

The energy account should be tuned so:

- static laser violates reserve or has materially worse energy metrics;
- event laser schedule remains feasible over typical event duty cycles;
- static snow-counter remains a strong low-energy baseline;
- no claim is made about real battery SOC until power is calibrated.

## Suggested first parameters

Use normalized one-hour steps:

- `energy_capacity = 24.0`
- `initial_energy = 12.0`
- `reserve_energy = 2.0`
- `harvest_per_step = 0.65`
- `per_step_budget = 1.20`
- `startup_peak_budget = 1.60`

Under these numbers:

- static laser `met+surface+laser` uses about `1.16` per step and drains about `0.51` per hour;
- calm core `met+radiometer+surface` uses about `0.32` per step and charges about `0.33` per hour;
- event laser can spend stored energy during events after charging in calm periods.

## Acceptance criteria

Before PPO:

- static laser is not the best feasible energy-account policy;
- dynamic event-laser schedule beats static snow-counter on event oracle loss or overall oracle loss without SOC violations;
- dynamic event-laser schedule has lower energy deficit than static laser;
- the result is robust in a formal TCN diagnostic, not only a local smoke.

Only after this gate should a single-seed PPO pilot be launched.
