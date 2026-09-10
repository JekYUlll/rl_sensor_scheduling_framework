# Physical Energy Route Admissibility Audit (2026-09-10)

## Decision

The existing finite-energy implementation is retained as a diagnostic
component, but it is **not admissible as the next PD-PPO scene**.

## Evidence inspected

- `configs/resources/windblown_entity_heater_resource_physical_w55_v1.json`
  defines physical-watt channel loads and a 55 W development envelope. Its
  50 W Parsivel heater is a declared equipment load, while the separate
  600 W profile in the legacy energy audit is an external non-controllable
  heater.
- `scripts/127_audit_entity_energy_trajectory.py` samples only 25 hourly
  rows from the available truth. Its 24/72/168-hour scenarios therefore do
  not provide valid multi-day evidence.
- The installed envelope declares an 8640 Wh battery, while the selectable
  sensor loads are approximately 0.016--1.50 W before heater increments.

## Observed consequence

The no-external-heater audit leaves the battery near its initial state
(`8423.6 Wh` after the available short trajectory). Adding the 600 W
external profile drives the legacy audit to the energy guard and is not a
valid sensing-policy comparison because that load is neither selected by the
policy nor assigned to a purchased sensor channel.

The old `harvest_per_step` values (`0.52`, `0.65`, `0.75`, `0.92`) and
capacity `180` belong to historical normalized experiments. They are not a
verified photovoltaic/storage trace for this entity-mapped system and will
not be reused to manufacture a binding SOC regime.

## Route correction

No PPO training is launched from this audit. A future SOC experiment requires
an independently documented supply/storage trajectory, a sufficiently long
truth window, and a causal mapping from that trajectory to executable subset
feasibility. Until then, the active route remains hardware-derived
instantaneous resource geometry plus complete-subset forecast crossover and
online-transfer gates. V643 is the latest valid geometry screen.
