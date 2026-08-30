# V314 horizon-6 diagnostic summary (2026-08-31)

## Protocol

V314 preserves the V311 scene, evaluator, state representation, context generation, teacher dwell, and validation windows. The only intended scientific change is `greedy-lookahead-steps=6`, matching `min_dwell_steps=6`. Two development scene seeds (6811, 6812) were evaluated with zero PPO updates and 16,384 hard forecast-value pretraining steps.

## Results

Lower loss is better; a positive margin is reference loss minus custom loss.
- validation_selected_static / oracle_loss_mean: custom wins 0/2, mean margin -0.041614 (range -0.043228 to -0.040000).
- validation_selected_static / oracle_loss_macro_subtype_event_staticnorm: custom wins 0/2, mean margin -0.151948 (range -0.229551 to -0.074345).
- feasible_static_projected / oracle_loss_mean: custom wins 0/2, mean margin -0.043769 (range -0.044311 to -0.043228).
- feasible_static_projected / oracle_loss_macro_subtype_event_staticnorm: custom wins 0/2, mean margin -0.102636 (range -0.130926 to -0.074345).
- full_open_unconstrained / oracle_loss_mean: custom wins 0/2, mean margin -0.091695 (range -0.097078 to -0.086311).
- full_open_unconstrained / oracle_loss_macro_subtype_event_staticnorm: custom wins 0/2, mean margin -0.238739 (range -0.260243 to -0.217234).
- aoi / oracle_loss_mean: custom wins 0/2, mean margin -0.048006 (range -0.057116 to -0.038896).
- aoi / oracle_loss_macro_subtype_event_staticnorm: custom wins 0/2, mean margin -0.155769 (range -0.173653 to -0.137884).
- random / oracle_loss_mean: custom wins 0/2, mean margin -0.027285 (range -0.029756 to -0.024813).
- random / oracle_loss_macro_subtype_event_staticnorm: custom wins 0/2, mean margin -0.083249 (range -0.119136 to -0.047362).
- round_robin / oracle_loss_mean: custom wins 0/2, mean margin -0.028420 (range -0.039785 to -0.017055).
- round_robin / oracle_loss_macro_subtype_event_staticnorm: custom wins 0/2, mean margin -0.107919 (range -0.134938 to -0.080899).

## Behavior

- seed 6811: always-on=1, always-off=3, mid-duty=2, switches/step=0.000289, warm-up aborts=0.
- seed 6812: always-on=0, always-off=2, mid-duty=4, switches/step=0.002099, warm-up aborts=0.

## Decision

V314 is rejected as a primary improvement. Matching the teacher horizon to the six-step executable dwell did not recover closed-loop transfer: validation static wins 0/2 on ordinary loss and macro, while both seeds violate the no-constant-channel behavior gate. The next intervention must address state/return alignment, not extend the teacher horizon.
