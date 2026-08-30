# V264 decision-sampled forecast reward

Seeds 5401--5402; positive delta means baseline loss minus PD-PPO loss.

| baseline | ordinary wins | mean ordinary delta | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.073276 | 1/2 | -0.067630 |
| feasible_static_projected | 1/2 | -0.052971 | 1/2 | -0.039336 |
| aoi | 0/2 | -0.061980 | 0/2 | -0.065780 |
| round_robin | 1/2 | -0.051938 | 1/2 | -0.057165 |
| random | 0/2 | -0.032680 | 0/2 | -0.026951 |
| full_open_unconstrained | 0/2 | -0.076104 | 0/2 | -0.080185 |
| best_static | 0/2 | -0.073276 | 1/2 | -0.067630 |
| best_original_dynamic | 0/2 | -0.066841 | 0/2 | -0.075141 |
