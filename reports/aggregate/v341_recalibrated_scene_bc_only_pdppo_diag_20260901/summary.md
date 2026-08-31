# v341_recalibrated_scene_bc_only_pdppo_diag_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.059183          -0.133777
        dynamic           2              1           1             -0.014369          -0.010673
      full_open           2              0           2             -0.002708           0.000879

Behavior rows with zero warm-up aborts and zero constant channels: 1/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7341                           0        0.504094     1.063352               -0.083271            -0.148452                -0.033429             -0.032162                  -0.002780                0.000373                   0                       0                        0                      4           0.018382    1.263056            1.92
 6872         7342                           0        0.311744     0.546018               -0.035095            -0.119103                 0.004691              0.010817                  -0.002635                0.001384                   0                       0                        3                      2           0.003618    1.171623            1.68
