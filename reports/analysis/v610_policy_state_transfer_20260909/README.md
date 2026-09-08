# V610 complete policy-state transfer audit

## Purpose

V608 rollout files stored raw measurement observations but not the complete
state presented to the PPO policy. After adding `agent_observations` to the
rollout artifact, V610 reloaded the completed V608 checkpoint without training
and evaluated the same six starts. This audit verifies the actual policy-input
boundary and tests chronological action transfer.

## Artifact verification

The saved V610 custom-policy rollout contains:

- raw measurement observation: `(6144, 12)`;
- policy observation: `(6144, 515)`;
- dynamic-resource metadata: enabled;
- dynamic-resource tail: six values, with observed ranges including
  normalized effective channel costs up to `2.0` and previous-action cost up to
  `2.0`.

The policy therefore did receive the configured dynamic-resource state in
V608. The earlier raw-observation shape must not be used to claim otherwise.

## Chronological transfer probe

The target was the forecast-optimal feasible candidate action from the V607
candidate-alignment diagnostic. An ExtraTrees classifier was fitted on the
first `2`, `3`, or `4` rollout blocks and tested on later blocks. This is a
diagnostic of state-to-action transfer, not policy evidence or model
selection.

| training blocks | chronological action accuracy | most-frequent-action accuracy |
|---:|---:|---:|
| 2 | 11.08% | 7.59% |
| 3 | 11.00% | 5.14% |
| 4 | 7.62% | 12.40% |

As a capacity check only, a random row split reached `41.03%`; this is not a
valid deployment-transfer result because nearby rows and regime segments are
shared between train and test.

## Decision

The policy-input path is correct, but the complete state has weak chronological
transfer to the forecast-optimal candidate labels in this diagnostic. This
supports the existing physics-first conclusion: a new learner module is not
the next justified intervention. A new scene relation or observation design
must first produce a stable deployable state-to-value mapping, followed by a
fresh frozen transfer gate before PPO training.
