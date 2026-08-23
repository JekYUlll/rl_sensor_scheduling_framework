# Framework Baseline Supplement

Positive margins mean the supplementary baseline has higher loss than PD-PPO. forecast_greedy_one_step is a privileged myopic diagnostic because it chooses masks using final-test future loss; context_alert_bandit uses supplied synthetic warning-score columns; event_label_reference uses exact final-test subtype labels and is privileged.

## Summary

| policy                    | complete_seeds | mean_oracle_loss | mean_loss_margin_vs_custom_ppo | median_loss_margin_vs_custom_ppo | pdppo_step_win_count | pdppo_step_loss_count | mean_staticnorm_macro | mean_staticnorm_macro_margin_vs_custom_ppo | pdppo_staticnorm_macro_win_count | diagnostic_privilege              | mean_switches_per_step | mean_mid_duty_sensor_count | mean_warmup_abort_count |
| ------------------------- | -------------- | ---------------- | ------------------------------ | -------------------------------- | -------------------- | --------------------- | --------------------- | ------------------------------------------ | -------------------------------- | --------------------------------- | ---------------------- | -------------------------- | ----------------------- |
| context_alert_bandit_t0p5 | 5              | 0.488162         | -0.092441                      | -0.095280                        | 0                    | 5                     | 1.076240              | -0.323488                                  | 0                                | supplied_synthetic_warning_scores | 0.031698               | 3.000000                   | 0.000000                |
| event_label_reference_l8  | 5              | 0.485110         | -0.095493                      | -0.100930                        | 0                    | 5                     | 1.054635              | -0.345092                                  | 0                                | exact_final_event_subtype         | 0.011782               | 4.200000                   | 0.000000                |
