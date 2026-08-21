# H75 Operational Evidence Summary

## Result Source

- Experiment directory:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced`
- Final table:
  `env_dwell12_h75_operational_summary_5seed.csv`
- Comparison table:
  `env_dwell12_h75_operational_summary_5seed_comparisons.csv`
- Seeds:
  `41`, `42`, `43`, `44`, `45`
- Scenario:
  no-warmup balanced scene, B=`1.70`, environment dwell=`12`,
  symmetric duty guard high=`0.75`.

## Per-Seed Summary

| Seed | PD-PPO | Compact Static | Deployable Static | Best Original Dynamic | Best Duty Non-PD-PPO | Behaviour |
|---:|---:|---:|---:|---:|---:|---|
| 41 | 0.132783 | 0.138556 | 0.133001 | 0.141961 | 0.134914 | valid |
| 42 | 0.148363 | 0.137324 | 0.150508 | 0.157569 | 0.158304 | valid |
| 43 | 0.145440 | 0.142992 | 0.149899 | 0.151947 | 0.154271 | valid |
| 44 | 0.141450 | 0.157129 | 0.146482 | 0.156499 | 0.156419 | valid |
| 45 | 0.148030 | 0.159664 | 0.141213 | 0.150318 | 0.145130 | valid |

## Win Counts

| Comparator family | PD-PPO wins | Mean comparator-minus-PD-PPO delta |
|---|---:|---:|
| Full-open reference | 0/5 | -0.021959 |
| Original compact static | 3/5 | +0.003920 |
| Deployable selected static | 4/5 | +0.001007 |
| Best original dynamic heuristic | 5/5 | +0.008445 |
| Best duty-constrained non-PD-PPO | 4/5 | +0.006594 |

## Behaviour Gate

- `mid_duty_sensor_count=8` in all seeds.
- `always_on_sensor_count=0` in all seeds.
- `always_off_sensor_count=0` in all seeds.
- `warmup_abort_count=0` in all seeds.
- Switch rate range: `0.030400`--`0.031988`.
- Duty max range: `0.729818`--`0.745931`.

## Claims Supported

- PD-PPO learns nondegenerate schedules under symmetric duty/dwell deployment
  constraints.
- PD-PPO consistently outperforms original dynamic heuristics in the h75
  operational setting.
- PD-PPO outperforms deployable selected static and duty-constrained
  non-PD-PPO baselines in most seeds.
- Original compact static remains a useful diagnostic shortcut, but it violates
  the deployment behaviour target with always-on/off sensors.

## Claims Not Supported

- Do not claim full-open superiority.
- Do not claim universal dominance over original compact static.
- Do not claim all duty-constrained non-PD-PPO baselines are beaten in every
  seed; seed45 is a boundary case.
