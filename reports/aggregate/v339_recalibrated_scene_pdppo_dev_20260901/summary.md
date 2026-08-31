# V339 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.083182          -0.203217
        dynamic           2              0           0             -0.038368          -0.080113
      full_open           2              1           1             -0.026707          -0.068562

Behavior rows with zero warm-up aborts and zero constant channels: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7311                          49        0.472223     1.041922               -0.051400            -0.127022                -0.001558             -0.010733                   0.029091                0.021803                   0                       0                        0                      6           0.047909    1.175547            1.92
 6872         7312                          20        0.391613     0.706328               -0.114965            -0.279413                -0.075178             -0.149493                  -0.082504               -0.158926                   0                       0                        0                      4           0.033145    1.373472            2.06
