# V138 generic physical-channel scene gate

V138 evaluates whether arbitrary power-feasible scheduling over the six
physical-system channels retains useful temporal variation after removing the
three simulator subtype latents from sensor measurements and model state.
Subtype-weighted reward and subtype auxiliary supervision are disabled.

## Result

| Reference | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin | Mean macro margin |
|---|---:|---:|---:|---:|---:|
| Receding forecast reference | 5/5 | 5/5 | 5/5 | +0.032692 | +0.134500 |
| Online context rule | 2/5 | 2/5 | 2/5 | +0.003190 | +0.023550 |
| One-step forecast greedy | 2/5 | 2/5 | 2/5 | -0.006441 | -0.002249 |

Positive margins indicate lower loss than the validation-selected static
subset. The receding reference uses future forecast loss and is a scene-value
diagnostic, not a deployable baseline. Its consistent advantage establishes
that one fixed subset is not optimal throughout these scenes. The online
references do not pass the `5/5` joint gate, so policy training is deferred
until an online, physically available context representation exposes the
temporal value more reliably.

Seed1505 has zero particle-subtype steps in the final partition. The reported
macro for that scene averages the represented flux and thermal strata only,
matching the finite-stratum convention used by the main evaluator.

## Online-state predictability diagnostic

A leave-one-scene-out ExtraTrees diagnostic was fitted on four receding traces
and evaluated on the fifth. Inputs were restricted to the online scheduler
state; candidate forecast costs and future targets were excluded. Across the
five held-out scenes, the model reproduced 85.9% of complete executed masks and
94.9% of individual sensor decisions. At epochs where the dwell hold had
expired, exact 22-action accuracy was 53.6%; alert features alone reached 18.5%.

This diagnostic is not a policy result. It shows that the physical history,
observation masks, previous schedule, and constraint state contain transferable
information about the receding reference's choices. The weak hand-written
context rule therefore does not justify adding privileged subtype inputs. V139
may train the generic PD-PPO directly from the existing online state, with
subtype weighting and subtype auxiliary loss disabled.
