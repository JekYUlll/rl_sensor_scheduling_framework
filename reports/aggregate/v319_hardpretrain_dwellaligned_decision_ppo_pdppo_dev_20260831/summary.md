# V319 dwell-aligned hard forecast-value pretraining plus decision-only PPO

Date: 2026-08-31

## Protocol

- Scene seeds: 6811 and 6812; policy seeds: 6991 and 6992.
- Same corrected six-channel scene, B=`1.75`, minimum dwell `6`, forecast
  reward, static-normalized subtype objective, and deterministic evaluation as
  V318.
- Hard forecast-value pretraining used `16,384` steps, followed by `50,000`
  decision-only PPO timesteps. The only intended change from V318 was aligning
  the hard teacher lookahead from `4` to `6`, equal to minimum dwell.
- No bandit-dependent signal, test-time event label, residual action, or
  trainable action prior was used.

## Results

Baseline-minus-PD-PPO margins are positive when PD-PPO has lower loss.

| Reference | Ordinary mean | Ordinary wins | Macro mean | Macro wins |
|---|---:|---:|---:|---:|
| validation-selected static | -0.002654737660626183 | 1/2 | +0.04048981816732783 | 1/2 |
| feasible static | +0.021481639310778594 | 2/2 | +0.12640631087110982 | 2/2 |
| full-open unconstrained | +0.007711383312160258 | 1/2 | +0.05017883700492515 | 2/2 |
| AoI | +0.02041239175010054 | 2/2 | +0.07443472525125266 | 2/2 |
| random | +0.037899242002763345 | 2/2 | +0.14875608064530604 | 2/2 |
| round-robin | +0.04324343988806345 | 2/2 | +0.12291095656062595 | 2/2 |

## Behavior

- Warm-up aborts: 0/2 seeds.
- Always-on channels: 0/2 seeds.
- Always-off channels: 1/2 seeds.
- Mid-duty channels: 5 and 3.
- Switching rates: 0.03285569546967723 and 0.023375307569836443 per step.

## Decision

Aligning the hard teacher horizon with the six-step dwell improved the
feasible-static macro result to `2/2` and improved all listed dynamic and
random comparisons. However, validation-selected static ordinary/macro wins
remain only `1/2`, and the behavior gate fails because seed 6811 has one
always-off channel. V319 is therefore a promising diagnostic for teacher
horizon alignment, but it is not sufficient for primary confirmation and is
not expanded to fresh final seeds.
