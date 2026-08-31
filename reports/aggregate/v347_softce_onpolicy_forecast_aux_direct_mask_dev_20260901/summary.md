# V347 soft-target on-policy forecast auxiliary diagnostic

All supplied seeds are included. Positive margins mean lower custom loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.042098          -0.107621
        dynamic           2              1           1              0.002716           0.015483
      full_open           2              2           2              0.014378           0.027034

Behavior-valid rows: 2/2. This diagnostic changes only the on-policy
forecast-value auxiliary loss from masked MSE to soft cross-entropy; the
forecast-loss reward, feasible action set, and online information boundary
remain unchanged.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7381                          15        0.485372     1.039402               -0.064548            -0.124502                -0.014707             -0.008212                   0.015942                0.024323                   0                       0                        0                      6           0.036329    1.235755            1.92
 6872         7382                          20        0.296295     0.517656               -0.019647            -0.090741                 0.020140              0.039178                   0.012814                0.029746                   0                       0                        0                      6           0.054277    1.418550            1.92
