# v357_macro_checkpoint_cycling_pdppo_confirm_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           4              0           0             -0.040665          -0.105301
        dynamic           4              2           3              0.014430           0.036437
      full_open           4              2           2             -0.007364          -0.020708

Behavior rows with zero warm-up aborts and zero constant channels: 3/4.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6901         7441                          40        0.433684     0.842092               -0.040348            -0.060343                 0.033891              0.044384                   0.032327                0.054698                   0                       0                        0                      6           0.053119    1.347813            2.00
 6902         7442                          35        0.409496     1.174205               -0.059930            -0.138859                -0.013300              0.023063                  -0.042688               -0.112055                   0                       0                        0                      6           0.055001    1.447943            2.06
 6903         7443                          10        0.411990     0.959539               -0.054411            -0.135472                -0.004416             -0.019051                  -0.029653               -0.060325                   0                       0                        0                      5           0.042119    1.484592            2.06
 6904         7444                          20        0.405754     0.816040               -0.007971            -0.086530                 0.041544              0.097353                   0.010558                0.034851                   0                       0                        1                      5           0.050659    1.498533            2.06
