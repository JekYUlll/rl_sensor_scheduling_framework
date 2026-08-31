# v348_mse_aux03_onpolicy_direct_mask_dev_20260901 V346 aggregate

All supplied seeds are included. Positive margins mean lower custom loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.048779          -0.120794
        dynamic           2              1           1             -0.003965           0.002310
      full_open           2              2           2              0.007697           0.013861

Behavior-valid rows: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7381                          45        0.499289     1.063513               -0.078466            -0.148613                -0.028624             -0.032323                   0.002025                0.000212                   0                       0                        0                      5           0.055869    1.332830            1.90
 6872         7382                          20        0.295740     0.519891               -0.019092            -0.092976                 0.020695              0.036943                   0.013369                0.027511                   0                       0                        0                      6           0.051238    1.482734            1.92
