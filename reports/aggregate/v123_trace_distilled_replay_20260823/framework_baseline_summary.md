# Framework Baseline Supplement

Positive margins mean the supplementary baseline has higher loss than PD-PPO. forecast_greedy_one_step is a privileged myopic diagnostic because it chooses masks using final-test future loss; context_alert_bandit uses supplied synthetic warning-score columns; event_label_reference uses exact final-test subtype labels and is privileged.

## Summary

| policy                         | complete_seeds | mean_oracle_loss | mean_loss_margin_vs_custom_ppo | median_loss_margin_vs_custom_ppo | pdppo_step_win_count | pdppo_step_loss_count | mean_staticnorm_macro | mean_staticnorm_macro_margin_vs_custom_ppo | pdppo_staticnorm_macro_win_count | diagnostic_privilege                     | mean_switches_per_step | mean_mid_duty_sensor_count | mean_warmup_abort_count |
| ------------------------------ | -------------- | ---------------- | ------------------------------ | -------------------------------- | -------------------- | --------------------- | --------------------- | ------------------------------------------ | -------------------------------- | ---------------------------------------- | ---------------------- | -------------------------- | ----------------------- |
| trace_distilled_forecast_value | 5              | 0.584727         | -0.032931                      | -0.080619                        | 1                    | 4                     | 1.317955              | -0.083292                                  | 1                                | policy_training_future_loss_distillation | 0.007845               | 3.400000                   | 0.000000                |
