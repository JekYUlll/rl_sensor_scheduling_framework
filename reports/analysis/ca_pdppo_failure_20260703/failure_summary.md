# CA-PD-PPO Failure Analysis

This analysis uses completed development seeds 201--224. Margins are `bandit - PD-PPO`; positive values mean CA-PD-PPO is better.

## Losing Seeds

- Losing macro seeds: 11/24.
- Worst macro seed: `214` with margin `-0.009557`.

| seed | macro_margin_vs_bandit | step_margin_vs_bandit | event_particle_margin | event_flux_margin | event_thermal_margin | dominant_losing_event_type | switches_per_step | warmup_abort_count | min_on_blocked_action_rate |
| ---- | ---------------------- | --------------------- | --------------------- | ----------------- | -------------------- | -------------------------- | ----------------- | ------------------ | -------------------------- |
| 214  | -0.009557              | -0.017740             | -0.006789             | -0.121963         | 0.000231             | flux                       | 0.018559          | 0.000000           |                            |
| 215  | -0.007961              | -0.006760             | 0.003403              | 0.003745          | -0.024485            | thermal                    | 0.020024          | 0.000000           |                            |
| 208  | -0.007819              | -0.012628             | -0.011035             | -0.059629         | -0.003792            | flux                       | 0.016606          | 0.000000           |                            |
| 212  | -0.005930              | -0.004137             | -0.005885             | -0.023175         | -0.007884            | flux                       | 0.018559          | 0.000000           |                            |
| 224  | -0.004127              | -0.015842             | -0.000017             | -0.010700         | -0.004428            | flux                       | 0.013187          | 0.000000           |                            |
| 218  | -0.002574              | -0.007864             | 0.003595              | -0.033581         | -0.004653            | flux                       | 0.016606          | 0.000000           |                            |
| 209  | -0.002090              | -0.005921             | -0.006721             | -0.019846         | 0.000941             | flux                       | 0.024420          | 0.000000           |                            |
| 207  | -0.001967              | 0.002070              | 0.005194              | 0.006119          | -0.004618            | thermal                    | 0.018071          | 0.000000           |                            |
| 203  | -0.001689              | -0.004197             | 0.002411              | -0.045467         | 0.000345             | flux                       | 0.019048          | 0.000000           |                            |
| 219  | -0.000942              | -0.008874             | -0.002373             | -0.003255         | 0.000403             | flux                       | 0.017094          | 0.000000           |                            |
| 217  | -0.000789              | -0.009309             | -0.004271             | -0.006379         | 0.001776             | flux                       | 0.025397          | 0.000000           |                            |

Dominant event type among losing seeds:

| dominant_losing_event_type | count |
| -------------------------- | ----- |
| flux                       | 9     |
| thermal                    | 2     |

## Context Confidence Bins

| bin         | num_windows | pdppo_loss | bandit_loss | margin    | pdppo_specialist_entropy | pdppo_action_confidence | bandit_action_entropy |
| ----------- | ----------- | ---------- | ----------- | --------- | ------------------------ | ----------------------- | --------------------- |
| [0.00,0.40) | 18420       | 1.112777   | 1.107992    | -0.004785 | 0.124335                 |                         | 0.000237              |
| [0.40,0.55) | 1619        | 0.885148   | 0.881953    | -0.003195 | 1.064811                 |                         | 0.574122              |
| [0.55,0.70) | 3558        | 0.948801   | 0.948065    | -0.000736 | 1.049366                 |                         | 1.030458              |
| [0.70,0.85) | 2692        | 0.971749   | 0.974214    | 0.002465  | 1.042967                 |                         | 1.132565              |
| [0.85,1.00] | 72015       | 1.606405   | 1.612813    | 0.006407  | 1.034257                 |                         | 0.987188              |

Interpretation: negative margins in high-confidence bins point to weak context-to-action mapping; negative margins in low-confidence bins point to uncertain-context handling.

## Alert Onset / Offset Lag

Alert threshold: `0.5`. Lag bins use ±`12` simulation steps because the current benchmark does not expose wall-clock timestamps in these rollouts.

| bin           | num_windows | pdppo_loss | bandit_loss | margin    | pdppo_specialist_entropy | pdppo_action_confidence | bandit_action_entropy |
| ------------- | ----------- | ---------- | ----------- | --------- | ------------------------ | ----------------------- | --------------------- |
| pre_onset     | 8875        | 1.546427   | 1.543635    | -0.002792 | 1.141271                 |                         | 1.102975              |
| early_event   | 9630        | 1.246921   | 1.249341    | 0.002420  | 1.038600                 |                         | 1.132552              |
| mid_event     | 59505       | 1.577350   | 1.585027    | 0.007676  | 1.031415                 |                         | 0.984503              |
| late_event    | 2393        | 1.649984   | 1.647125    | -0.002859 | 1.017525                 |                         | 0.974150              |
| post_offset   | 521         | 0.701402   | 0.687422    | -0.013981 | 0.331519                 |                         | 0.007217              |
| outside_alert | 17380       | 1.137296   | 1.133178    | -0.004118 | 0.030183                 |                         | 0.000000              |

## PPO Stability Proxies

Exact masked action probabilities and critic prediction errors are not stored in the rollout artifacts. The table reports available proxies from the training log and rollout scores.
Stored rollout scores are saturated execution scores in this run, so score-gap confidence is intentionally left blank.

| seed | macro_margin_vs_bandit | value_prediction_error_proxy_tail_value_loss | advantage_mean_tail | advantage_std_tail | policy_entropy_tail | bandit_preferred_score_gap_proxy | top_two_specialist_score_gap | exact_mask_agreement_with_bandit |
| ---- | ---------------------- | -------------------------------------------- | ------------------- | ------------------ | ------------------- | -------------------------------- | ---------------------------- | -------------------------------- |
| 214  | -0.009557              | 162.011374                                   | -1.813069           | 20.560017          | 0.191418            | 1.667114                         |                              | 0.778809                         |
| 215  | -0.007961              | 195.292617                                   | -2.786258           | 21.967098          | 0.224637            | 1.632690                         |                              | 0.754150                         |
| 208  | -0.007819              | 180.301261                                   | -1.294174           | 21.478629          | 0.222530            | 1.421436                         |                              | 0.400879                         |
| 212  | -0.005930              | 271.969401                                   | -6.545694           | 25.937912          | 0.259377            | 1.902832                         |                              | 0.757080                         |
| 224  | -0.004127              | 191.521480                                   | -2.166751           | 22.226735          | 0.276571            | 1.512573                         |                              | 0.674805                         |
| 218  | -0.002574              | 230.460957                                   | -2.882294           | 21.846232          | 0.247359            | 1.851367                         |                              | 0.628418                         |
| 209  | -0.002090              | 185.679839                                   | -4.039091           | 21.841288          | 0.262049            | 1.948926                         |                              | 0.875977                         |
| 207  | -0.001967              | 250.570085                                   | -2.319491           | 24.110716          | 0.235298            | 1.936279                         |                              | 0.957764                         |
| 203  | -0.001689              | 186.146243                                   | -2.823253           | 22.249844          | 0.181163            | 1.883691                         |                              | 0.709229                         |
| 219  | -0.000942              | 308.063244                                   | -5.291751           | 25.167180          | 0.255338            | 1.447021                         |                              | 0.633789                         |
| 217  | -0.000789              | 212.137227                                   | -3.466993           | 23.101702          | 0.240359            | 1.816162                         |                              | 0.878418                         |
| 201  | 0.000058               | 166.887160                                   | -3.262065           | 21.052934          | 0.226270            | 1.953125                         |                              | 0.968750                         |
| 223  | 0.000735               | 178.462367                                   | -0.748987           | 20.861077          | 0.239762            | 1.923462                         |                              | 0.949463                         |
| 213  | 0.001257               | 199.391648                                   | -2.085343           | 22.343717          | 0.219222            | 1.950195                         |                              | 0.972168                         |
| 202  | 0.002103               | 320.604663                                   | -3.341155           | 28.271431          | 0.303279            | 1.889038                         |                              | 0.921143                         |
| 222  | 0.003177               | 186.914877                                   | -2.138222           | 21.138131          | 0.242584            | 1.864136                         |                              | 0.909912                         |
| 221  | 0.004597               | 146.909991                                   | -0.996704           | 19.553886          | 0.259475            | 1.711060                         |                              | 0.806885                         |
| 216  | 0.005224               | 278.624609                                   | -3.264774           | 26.887902          | 0.258097            | 1.853223                         |                              | 0.747070                         |
| 220  | 0.006539               | 256.430742                                   | -3.774367           | 25.644494          | 0.323909            | 1.732666                         |                              | 0.821777                         |
| 210  | 0.008833               | 257.650694                                   | -1.708167           | 25.054604          | 0.281228            | 1.783936                         |                              | 0.855957                         |
| 205  | 0.016172               | 244.532263                                   | -3.524377           | 23.775236          | 0.256458            | 1.781836                         |                              | 0.630615                         |
| 211  | 0.029975               | 218.223479                                   | -3.953629           | 22.956720          | 0.241991            | 1.219238                         |                              | 0.481201                         |
| 204  | 0.055278               | 210.261854                                   | -4.050447           | 22.551420          | 0.284987            | 1.306763                         |                              | 0.387451                         |
| 206  | 0.109675               | 207.407391                                   | -6.318785           | 21.697283          | 0.252316            | 1.314819                         |                              | 0.543213                         |

## Artifact Limits

- `min_on_blocked_action_rate` is not directly recorded and is intentionally left as NaN rather than inferred from ambiguous score/projection mismatches.
- `masked_action_probability_for_bandit_preferred_action` is not recorded; `bandit_preferred_score_gap_proxy` is provided as a non-equivalent diagnostic proxy.
- Stored rollout scores are saturated execution scores rather than actor logits; score-gap confidence is therefore not interpreted.

## Decision

Do not launch fresh final seeds from this analysis alone. The next clean step is a bounded development wave only if the failure structure supports one of the method-consistent variants: stronger context encoder, gated fusion, entropy decay, longer rollout, or lower learning rate.
