# Flexible-subset v1 pilot, seed 401

## Configuration

- Six physical-system channels; no required channel or coverage group.
- Effective steady budget: 1.35; cold-start budget: 1.65.
- Feasible action set: 29 masks spanning cardinalities 0--3.
- Training: 30,000 PPO steps on a development seed.

## Result

Lower values are better for both forecast-loss endpoints.

| Policy | Mean forecast loss | Macro normalized loss |
|---|---:|---:|
| PD-PPO | 0.171239 | 0.656944 |
| Validation-selected static | 0.161443 | 0.723909 |
| AoI | 0.166041 | 0.647058 |
| Round-robin | 0.183067 | 0.729620 |
| Random | 0.175288 | 0.686884 |

PD-PPO passes the macro comparison against the selected static schedule but
does not pass the mean-loss comparison and is slightly worse than AoI on both
endpoints. It has zero feasibility violations and zero warm-up aborts.

## Behaviour diagnosis

- Duty fractions in sensor order: 0.5968, 0.0078, 1.0000, 0.6046, 0.3954, 0.0000.
- The policy uses only three executed masks.
- `shielded_thermo_hygro` is always active and `fc4_flux` is never active.
- Validation-based automatic teaching selects the laser mask for both particle
  and flux conditions, showing that the first scene does not make FC4's flux
  information sufficiently distinct.

## Decision

The pilot does not pass the evidence gate and is not expanded to more seeds.
The next bounded configuration raises the effective thermo-hygro cost, improves
FC4 flux fidelity, reduces laser flux proxy quality, and weakens BC/AWBC
regularisation. These changes target the measured failure mode and do not alter
the prediction-loss reward or masked-PPO method identity.
