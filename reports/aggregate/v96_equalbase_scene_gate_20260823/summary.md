# V96 equal-base-weight scene gate

V96 changes only the nine base forecast-target weights to equal values after
the existing physical scaling. Event-subtype weights, truth-generation
parameters, six-channel sensor model, effective costs, power budget, warning
path, evaluator architecture, and temporal protocol remain fixed from V83a.

This report is a scene-readiness gate, not a trained-policy result. It separates
three evidence levels:

1. `context_alert_bandit_t0p5` is a deployable deterministic context policy
   driven by noisy online warning scores.
2. `event_label_reference_l8` is a privileged diagnostic using exact event
   subtype labels and is not deployable.
3. `dynamic:receding_oracle_l8` is an exact-lookahead upper diagnostic that
   evaluates every feasible action from restored environment snapshots and is
   not deployable.

## Gate results

| diagnostic | ordinary wins vs strongest static | mean ordinary margin | macro wins vs strongest static | mean macro margin | behavior passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deployable context replay | 4/5 | +0.027521 | 5/5 | +0.111938 | 4/5 |
| Exact-label replay | 5/5 | +0.028115 | 5/5 | +0.119585 | 4/5 |
| Exact receding upper | 5/5 | +0.064679 | not computed | not computed | 5/5 |

Positive margins mean lower loss than the strongest static subset selected for
the same scene. The exact-receding upper uses all 35 non-empty feasible actions
in every seed. All six channels have intermediate duty in every seed,
`always_on_sensor_count=0`, `always_off_sensor_count=0`, and
`warmup_abort_count=0`. Its mean switching rate is `0.060402` per step.

The only deployable-context ordinary-loss miss is seed 1105, where the margin
is `-0.001733`; its macro margin remains `+0.096932`. Seed 1101 is the only
context behavior miss, with one always-on and one always-off channel. These
failures do not invalidate scene readiness because the exact dynamic upper
shows strict dynamic headroom and valid six-channel utilization in all five
seeds. They remain targets for the learned policy rather than grounds for
further scene calibration.

## Decision

The V96 scene and equal-base-weight objective pass the pre-training readiness
gate. Further scene-weight tuning is closed. The next experiment trains the
complete nonlinear PD-PPO configuration on frozen V96 scene assets with fresh
policy seeds. Acceptance requires performance against the strongest static and
conventional dynamic families together with no always-on/off channels.
