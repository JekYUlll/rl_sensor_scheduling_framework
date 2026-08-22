# Framework Baseline Supplement

Positive margins mean the supplementary baseline has higher loss than PD-PPO. forecast_greedy_one_step is a privileged myopic diagnostic because it chooses masks using final-test future loss; context_alert_bandit uses supplied synthetic warning-score columns; event_label_reference uses exact final-test subtype labels and is privileged.

## Summary

| policy                    | complete_seeds | mean_oracle_loss | mean_loss_margin_vs_custom_ppo | median_loss_margin_vs_custom_ppo | pdppo_step_win_count | pdppo_step_loss_count | mean_staticnorm_macro | mean_staticnorm_macro_margin_vs_custom_ppo | pdppo_staticnorm_macro_win_count | diagnostic_privilege              | mean_switches_per_step | mean_mid_duty_sensor_count | mean_warmup_abort_count |
| ------------------------- | -------------- | ---------------- | ------------------------------ | -------------------------------- | -------------------- | --------------------- | --------------------- | ------------------------------------------ | -------------------------------- | --------------------------------- | ---------------------- | -------------------------- | ----------------------- |
| context_alert_bandit_t0p5 | 5              | 0.422401         | -0.086652                      | -0.083517                        | 0                    | 5                     | 1.223011              | -0.359723                                  | 0                                | supplied_synthetic_warning_scores | 0.015386               | 4.400000                   | 0.000000                |
| event_label_reference_l8  | 5              | 0.422908         | -0.086145                      | -0.087607                        | 0                    | 5                     | 1.232568              | -0.350166                                  | 0                                | exact_final_event_subtype         | 0.015328               | 4.400000                   | 0.000000                |
