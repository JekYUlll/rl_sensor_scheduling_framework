# DQN frozen-evaluator backend audit

## Purpose

The matched Double-DQN run computes the frozen TCN reward on CUDA during
training for tractability, whereas the matched PPO controls compute that reward
on CPU.  Every final trajectory is rescored on CPU.  This audit quantifies the
training-reward roundoff introduced by the device change.

## Check

- Source: clean forecast-reward policy, seed 117.
- Evaluator artifact: the same checksummed `v2_tcn_oracle.pt` loaded once on
  CPU and once on CUDA.
- Inputs: all 4,096 forecast windows from the frozen final PD-PPO trajectory.
- Loss: the configured subtype-conditioned, weighted, normalized and clipped
  forecast loss used by the environment.

## Result

| Quantity | Value |
|---|---:|
| CPU mean reward loss | 0.698250313 |
| CUDA mean reward loss | 0.698312503 |
| Mean signed CUDA minus CPU difference | +0.000062191 |
| Mean absolute difference | 0.000164973 |
| Mean absolute difference relative to CPU mean | 0.0236% |
| 95th percentile absolute difference | 0.000698734 |
| 99th percentile absolute difference | 0.001379805 |
| Maximum absolute difference | 0.002130882 |

The raw physical prediction maximum is not an appropriate backend metric
because the output vector contains an unscaled snow-flux variable.  The
configured reward loss is the quantity consumed by the learner.  Its backend
difference is small relative to the mean loss, and all reported DQN, PPO and
fixed-schedule metrics are produced by the common CPU evaluator after training.
The two training backends should therefore be described as numerically
equivalent at the reward level, not bitwise identical.

