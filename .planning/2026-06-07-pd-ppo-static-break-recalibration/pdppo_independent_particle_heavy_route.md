# PD-PPO Independent Particle-Heavy Route Plan

## Purpose
This file is the active route note for the first-paper PD-PPO fork. Its main
purpose is to prevent future recovery sessions from mixing this fork with the
separate v1 line.

The active target remains a PD-PPO / RL sensor scheduling result in
`rl_sensor_scheduling_framework`. The v1 work is a long archived exploration
line that did not produce a stable successful result. Its records can be read as
negative evidence, failed-route memory, and design background, but it must not
be merged into the active PD-PPO implementation, main-method description, or
main numerical evidence for this paper.

## Boundary And Permitted v1 Use
- Active codebase: `rl_sensor_scheduling_framework`.
- Active planning directory:
  `.planning/2026-06-07-pd-ppo-static-break-recalibration/`.
- Active paper line: PD-PPO / RL sensor scheduling only.
- Excluded from the active implementation and main evidence chain:
  - `v1/forecast_cmdp`;
  - v1 MPC, teacher, planner, or robust-planner methods;
  - v1 protocol numbers as current PD-PPO evidence;
  - v1 manuscript claims as first-paper claims;
  - v1 planning files as instructions for this fork.
- Permitted v1 use:
  - read v1 records to avoid repeating unsuccessful routes;
  - use v1 as archived diagnostic context;
  - mention v1 only as background or future/second-paper exploration if needed;
  - never mix v1 results into the PD-PPO main result tables.

## Current Diagnosis
- The earlier `dual_flux_particle_v7` learned branch is not stable enough.
  Seeds 41--43 passed, but seeds 44--45 failed the deployable-static gate.
- The 41--45 dual branch has only `3/5` deployable-static wins and mean delta
  `-0.007197`.
- Behaviour is mostly valid, so the failure is not a deployment-duty failure.
  It is a profile / teacher / scene-value failure.
- Fixed event-pair AWBC is too narrow. It creates dynamic schedules, but the
  teacher labels do not match the best structural event/calm masks in difficult
  seeds.
- Oracle-greedy AWBC improves the seed44 dual branch from `0.411077` to
  `0.372997`, and improves event-window loss against deployable static, but
  still loses calm-window loss.
- Structural screening identifies `particle_heavy_flux_v7` as the current
  strongest profile family. It is strongest in seeds 42--44 and passes seed45,
  where `dual_flux_particle_v7` fails.

## Current Main Candidate
| Field | Value |
|-------|-------|
| Scene config | `configs/sensors/windblown_sensors_physical_event_v16_surface_boundary.yaml` |
| Budget | `1.15` |
| Startup peak | `1.55` |
| Max active sensors | `4` |
| Dwell constraint | environment/action minimum dwell `12` |
| Duty guard | low/high `0.12/0.75`, symmetric for PD-PPO and baselines |
| Energy account | harvest `0.82`, capacity `180`, reserve `20`, SOC buffer `40`, SOC penalty `0.08` |
| Eval start selection | `event_transport_rich`, stride `64` |
| Target profile | `particle_heavy_flux_v7` |
| Target weights | `0.03 0.03 0.10 0.01 0.01 0.0 16.0 22.0 22.0` |
| Target scales | `5.0 5.0 5.0 1.0 1.0 100.0 0.0001 0.2 5.0` |
| PPO controls | event-gated actor, SOC auxiliary critic, reserve-aware controls |
| Teacher | adaptive `oracle_greedy` AWBC |

## Immediate Running Experiment
Remote tmux:
`pdppo_v16_particle_heavy_seed45_h082_oraclegreedy_20260609`

Output directory:
`reports/v31_static_break_v16_particle_heavy_dwell12_ppo_seed45_h082_soc_oraclegreedy_eventeval_20260609`

Purpose:
test the current strongest profile/teacher combination on the hardest observed
extension seed.

Decision after completion:
- If seed45 passes deployable static and behaviour, launch locked seeds 41--45
  with the same settings.
- If seed45 narrowly fails but improves over the dual branch, inspect
  event/calm losses and top masks before changing simulator physics.
- If seed45 fails badly, stop PPO retries and run the v17 structural gate.

## Evidence Gates
- Primary fair static gate: beat `duty_constrained_validation_selected_static`.
- Preferred static gate: beat the best deployable static family.
- Dynamic baseline gate: beat original dynamic heuristics and duty-constrained
  dynamic baselines.
- Behaviour gate:
  - `mid_duty_sensor_count = 8` preferred;
  - `always_on_sensor_count = 0`;
  - `always_off_sensor_count = 0`;
  - `warmup_abort_count = 0`;
  - `switches_per_step` roughly `0.02--0.06`.
- Reporting rule: report all seeds. Do not drop failed seeds unless a real
  data or configuration bug is found.

## If Simulator Physics Must Change
Only change simulator/generator structure after the v16 particle-heavy
oracle-greedy learned probe is insufficient.

Candidate v17 change:
- reduce event particle microstructure correlation from `0.20` to `0.00`;
- increase microstructure amplitude:
  - sigma from `0.45` to `0.55--0.65`;
  - diameter from `0.08` to `0.12--0.16`;
  - velocity from `1.00` to `1.25--1.50`.

Constraints should remain symmetric and unchanged during the first v17 gate.
Before any PPO training on v17, run a structural gate across seeds 41--45 and
require particle-heavy headroom against deployable static without relying on
multiple always-on or always-off sensors.

## Execution Order
1. Finish seed45 particle-heavy oracle-greedy PPO.
2. Sync metrics, audit raw CSVs, and append findings to `CHANGELOG.md`.
3. If the seed45 gate passes, run locked 41--45 particle-heavy oracle-greedy
   replication.
4. If it fails, run the v17 structural gate before more PPO.
5. After a 5-seed learned positive result, update manuscript tables and figures.
   Do not import v1 results into the first-paper method or main evidence.

## Reporting Rules
- Update `CHANGELOG.md` after every new result.
- Update this active plan, `progress.md`, and `findings.md` after decisions.
- Label evidence explicitly as structural, learned, deployment-valid, or
  diagnostic.
- Keep v1 out of the active PD-PPO claim chain, while allowing it as an
  archived failed-route reference.
