# V150 behavior-aware validation result

V150 selected update 15 because it had zero behavior failures across all four
training-scene validation evaluations and the best prediction score among that
valid tier. The frozen policy did not transfer the behavior or static-relative
prediction gains to held-out scene1505. Ordinary/macro margins against static
were -0.017231/-0.062708, with one always-on and three always-off channels.

This rejects behavior-aware checkpoint selection as a standalone transfer
repair. The next experiment is a five-fold leave-one-scene-out audit with the
same training and selection protocol, not further tuning on scene1505.
