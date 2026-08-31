# v349_mse_aux005_onpolicy_direct_mask_dev_20260901 V346 aggregate

All supplied seeds are included. Positive margins mean lower custom loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.013620          -0.059825
        dynamic           2              2           2              0.031193           0.063279
      full_open           2              2           2              0.042855           0.074830

Behavior-valid rows: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7381                           5        0.443018     0.972185               -0.022195            -0.057285                 0.027647              0.059004                   0.058295                0.091540                   0                       0                        0                      6           0.031046    1.352847            1.92
 6872         7382                           5        0.281694     0.489281               -0.005046            -0.062366                 0.034740              0.067554                   0.027415                0.058121                   0                       0                        0                      5           0.024678    1.385833            1.92
