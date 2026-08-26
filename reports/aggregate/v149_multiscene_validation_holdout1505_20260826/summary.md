# V149 multi-scene validation checkpoint result

V149 selected update 40 from four training scenes' calibration/validation
partitions using the maximum of the mean ordinary and macro static ratios. The
selected score was 1.021841835.

On held-out scene1505, PD-PPO nearly matched validation-selected static on
ordinary loss (-0.000315 margin), exceeded it on the normalized macro endpoint
(+0.012189), and exceeded the best conventional dynamic policy (+0.033114).
The policy nevertheless failed the frozen behavior gate with one always-on and
two always-off channels. This diagnostic authorizes behavior-constrained
validation selection, not a change to training reward or final execution.
