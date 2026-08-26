# V147 interleaved multi-scene training

V147 used one shared policy and optimizer while rotating training episodes over
scenes1501--1504. Each scene retained its frozen forecast evaluator. Optimization
used the policy-training partitions only, and the model was frozen after 81,920
steps before evaluation on scene1505.

The held-out result improved materially over the sequential V146 curriculum.
PD-PPO ordinary loss was `0.245033` versus `0.236810` for selected static, while
its validation-normalized macro was `0.884520` versus `0.898615`. The resulting
margins were `-0.008223/+0.014095`. The best conventional dynamic loss was
`0.270239`, giving PD-PPO a positive margin of `+0.025206`.

No channel was always on or off, five channels had mid-range duty, switching was
`0.026849` per step, and warm-up aborts were zero. The dual-endpoint static gate
therefore remains unmet. The next bounded change is validation-only checkpoint
selection aggregated over all four training scenes.
