# V91 frozen stochastic replay

The five frozen V90 checkpoints were replayed three times with independent,
prespecified sampling seeds. No policy fitting, checkpoint selection, scene
change, or evaluator change occurred.

- All three replicates pass flexible six-channel behavior in 5/5 scenes. Every
  channel has intermediate duty and switching is about 0.04--0.06 per step.
- All three replicates lose jointly to the strongest static family in 5/5
  scenes. Mean ordinary margins range from -0.0285 to -0.0355 and mean macro
  margins from -0.0814 to -0.0899.
- Joint wins against conventional dynamic policies are only 2/5, 3/5, and 2/5.

Unit-temperature sampling is rejected as a primary execution policy. It proves
that V90's learned distribution contains diverse feasible actions, but too much
probability remains on forecast-poor actions. A lower sampling temperature may
be considered only through calibration/validation selection before test replay;
the three final-replay replicates cannot be used to choose that temperature.
