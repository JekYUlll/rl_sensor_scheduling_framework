# v353_cycling_scene_pdppo_dev_20260901 V346 aggregate

All supplied seeds are included. Positive margins mean lower custom loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              1           1              0.001043           0.013663
        dynamic           2              2           2              0.021783           0.052003
      full_open           2              1           1              0.009678           0.038421

Behavior-valid rows: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6891         7411                          45        0.303884     0.679947                0.010589             0.040388                 0.023077              0.067903                   0.027346                0.077435                   0                       0                        0                      5           0.053553    1.420642            2.00
 6892         7412                          40        0.413485     1.132072               -0.008504            -0.013063                 0.020489              0.036103                  -0.007991               -0.000593                   0                       0                        0                      6           0.034593    1.349045            2.06
