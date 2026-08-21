# Flexible-subset v2 pilot, seed 402

## Configuration change from v1

- Raised the fixed thermo-hygro effective cost from 0.32 to 0.40.
- Reduced the laser channel's flux-proxy quality and strengthened FC4 flux fidelity.
- Reduced AWBC coefficient from 0.15 to 0.05 and BC pretraining from 1,500 to 500 steps.
- Reduced the steady/cold-start budgets to 1.25/1.55, leaving 24 feasible masks.

## Result

Lower values are better for both endpoints.

| Policy | Mean forecast loss | Macro normalized loss |
|---|---:|---:|
| PD-PPO | 0.165710 | 0.786622 |
| Validation-selected static | 0.197650 | 0.996109 |
| AoI | 0.196863 | 1.015170 |
| Round-robin | 0.181180 | 0.853506 |
| Random | 0.231912 | 2.057562 |

PD-PPO beats every fair reported baseline on both endpoints. It has zero power
violations, zero warm-up aborts, no always-active channel, five mid-duty channels,
and one effectively inactive channel.

## Behaviour

The executed schedule contains 11 unique masks. The most common mask is
`met_station_core + laser_disdrometer` (1,561/2,304 epochs), followed by
`surface_temp_ir + laser_disdrometer` (306 epochs) and
`radiometer_basic + shielded_thermo_hygro + surface_temp_ir` (216 epochs).
FC4 is selected for six epochs, below the mid-duty threshold. This is acceptable
for the bounded pilot's no-multiple-degeneracy gate but remains a replication
diagnostic.

## Decision

The single-seed performance and behaviour gates pass. The v2 configuration is
held fixed for development-seed replications 403 and 404 before any longer or
fresh confirmatory run.
