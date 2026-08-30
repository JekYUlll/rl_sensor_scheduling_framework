# V266 decision-sampled forecast reward

Seeds 5601--5602; positive delta means baseline loss minus PD-PPO loss.

| baseline | ordinary wins | mean ordinary delta | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 1/2 | -0.014113 | 1/2 | -0.016508 |
| feasible_static_projected | 1/2 | -0.017261 | 1/2 | -0.018588 |
| aoi | 1/2 | -0.004164 | 1/2 | -0.003824 |
| round_robin | 1/2 | -0.001571 | 0/2 | -0.005079 |
| random | 2/2 | 0.012520 | 2/2 | 0.012248 |
| full_open_unconstrained | 0/2 | -0.006030 | 0/2 | -0.004969 |
| best_static | 1/2 | -0.019259 | 1/2 | -0.020492 |
| best_original_dynamic | 0/2 | -0.006518 | 0/2 | -0.010840 |
