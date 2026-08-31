# v342_decision_only_bc_diag_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.143244          -0.239433
        dynamic           2              0           0             -0.098430          -0.116329
      full_open           2              0           1             -0.086768          -0.104778

Behavior rows with zero warm-up aborts and zero constant channels: 0/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7351                           0        0.523883     1.052479               -0.103060            -0.137579                -0.053218             -0.021289                  -0.022570                0.011246                   0                       0                        1                      3           0.006803    0.583750            1.90
 6872         7352                           0        0.460076     0.768203               -0.183428            -0.341288                -0.143641             -0.211369                  -0.150967               -0.220801                   0                       1                        2                      2           0.005500    1.323437            1.78
