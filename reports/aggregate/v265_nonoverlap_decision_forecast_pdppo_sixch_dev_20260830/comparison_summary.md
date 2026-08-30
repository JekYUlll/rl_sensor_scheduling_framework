# V265 decision-sampled forecast reward

Seeds 5501--5502; positive delta means baseline loss minus PD-PPO loss.

| baseline | ordinary wins | mean ordinary delta | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.046216 | 0/2 | -0.021243 |
| feasible_static_projected | 0/2 | -0.038677 | 1/2 | -0.000385 |
| aoi | 1/2 | 0.021618 | 2/2 | 0.056843 |
| round_robin | 1/2 | 0.002669 | 2/2 | 0.016507 |
| random | 2/2 | 0.010209 | 2/2 | 0.036655 |
| full_open_unconstrained | 1/2 | -0.023701 | 1/2 | -0.000126 |
| best_static | 0/2 | -0.046216 | 0/2 | -0.021243 |
| best_original_dynamic | 1/2 | -0.005246 | 2/2 | 0.016036 |
