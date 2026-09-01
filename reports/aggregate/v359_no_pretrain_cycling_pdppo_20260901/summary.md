# v359_no_pretrain_cycling_pdppo_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           4              0           0             -0.042464          -0.113831
        dynamic           4              3           2              0.012631           0.027907
      full_open           4              1           1             -0.009163          -0.029238

Behavior rows with zero warm-up aborts and zero constant channels: 1/4.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6901         7461                          35        0.430376     0.845445               -0.037040            -0.063696                 0.037199              0.041031                   0.035635                0.051345                   0                       0                        1                      5           0.065567    1.500417            2.06
 6902         7462                          15        0.375302     1.111927               -0.025736            -0.076581                 0.020894              0.085341                  -0.008494               -0.049778                   0                       0                        1                      4           0.039224    1.479948            1.82
 6903         7463                          20        0.418403     0.950349               -0.060824            -0.126281                -0.010829             -0.009860                  -0.036066               -0.051135                   0                       0                        1                      5           0.043711    1.529609            1.92
 6904         7464                          25        0.444040     0.918277               -0.046256            -0.188767                 0.003258             -0.004884                  -0.027727               -0.067386                   0                       0                        0                      5           0.048487    1.573655            2.06
