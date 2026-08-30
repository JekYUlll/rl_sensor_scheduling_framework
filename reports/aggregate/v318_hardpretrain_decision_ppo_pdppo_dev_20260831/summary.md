# V318 hard forecast-value pretraining plus decision-only PPO

Date: 2026-08-31

## Protocol

- Scene seeds: 6811 and 6812.
- Policy seeds: 6981 and 6982.
- Intervention: feasible hard forecast-value initialization for 16,384
  pretraining steps, followed by 50,000 actual decision-only PPO timesteps.
- Preserved controls: corrected six-channel scene, B=1.75, minimum dwell 6,
  forecast reward, static-normalized subtype objective, deterministic final
  evaluation, and no trainable action prior.
- Excluded signals: bandit actions or loss, test-time event labels, residual
  actions, and counterfactual supervision.

## Results

Baseline-minus-PD-PPO margins are positive when PD-PPO has lower loss.

| Reference | Ordinary mean | Ordinary wins | Macro mean | Macro wins |
|---|---:|---:|---:|---:|
| validation-selected static | -0.015944843472277648 | 1/2 | -0.043152498773989134 | 1/2 |
| feasible static | +0.00819153349912713 | 2/2 | +0.042763993929792854 | 1/2 |
| full-open unconstrained | -0.005578722499491207 | 0/2 | -0.033463479936391816 | 0/2 |
| AoI | +0.007122285938449074 | 1/2 | -0.009207591690064298 | 1/2 |
| random | +0.02460913619111188 | 2/2 | +0.06511376370398908 | 2/2 |
| round-robin | +0.029953334076411986 | 2/2 | +0.03926863961930899 | 1/2 |

## Behavior

- Warm-up aborts: 0/2 seeds.
- Always-on channels: 0/2 seeds.
- Always-off channels: 0/2 seeds.
- Mid-duty channels: 5 and 6.
- Switching rates: 0.019250253292806482 and 0.039658416558112614 per step.

## Decision

The combination passes the permanent-channel behavior gate but does not pass
the predictive transfer gate. It is rejected as a primary improvement:
hard initialization followed by decision-only PPO does not consistently beat
the validation-selected static or macro baseline. The complete raw evidence
is stored in the two seed directories named by the launcher.
