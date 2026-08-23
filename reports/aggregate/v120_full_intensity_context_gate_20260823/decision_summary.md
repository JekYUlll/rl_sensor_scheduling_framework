# V120 full-intensity information gate

V120 changed only the continuous warning-intensity contribution from `0.75`
to `1.0`. The six-channel cost model, arbitrary power-feasible subset action
space, truth dynamics, frozen forecaster, partitions, and seeds were unchanged.

## Deployable context gate

The calibration-defined context policy beat the validation-selected static
schedule in 3/5 scenes on ordinary forecast loss and 4/5 scenes on the
static-normalized subtype macro. Mean margins were +0.005443 and +0.031276,
respectively. Seeds 1303 and 1304 each had two always-off channels, exceeding
the prespecified maximum of one.

## Structural dynamic-value gate

The privileged eight-step receding diagnostic beat the best fixed feasible
subset in 5/5 scenes. Ordinary-loss margins ranged from +0.073649 to +0.121511,
with a mean of +0.089451. Every channel had intermediate duty in every scene,
there were no always-on or always-off channels, and switching ranged from
0.048177 to 0.052373 per step.

## Decision

The scenario contains substantial state-dependent scheduling value, but the
fixed-threshold warning policy does not recover it reliably. A new PPO run is
not authorized from V120 alone. The next step is a trace-level audit of whether
the receding action can be predicted from online warning and scheduler state on
disjoint partitions.
