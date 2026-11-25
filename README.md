# RL Sensor Scheduling Framework

A self-contained research framework for **power-constrained online scheduling of a
multi-sensor observation system**. The scheduling problem is treated as a sequential
decision problem rather than a post-hoc imputation exercise: at every step only a subset
of sensors can be active under a power budget, and the policy decides which subset that is.

The repository contains the simulator, the estimators, the reinforcement-learning
agents, the forecasting baselines, the evaluation utilities, and the aggregated result
tables used in our experiments.

## Idea

1. A high-frequency, fully observed truth series defines the assumed environment.
2. A scheduler sees the same truth series and decides which sensors to activate at each step.
3. Only activated sensors return observations and charge their power cost.
4. An estimator reconstructs the state from the partial observations and feeds uncertainty back.
5. Each policy produces its own dataset, which is passed to the same downstream forecasting
   models, so different scheduling strategies are compared on identical predictive machinery.

The objective is to minimise long-run estimation and downstream forecast loss under an
average/instantaneous power constraint, with optional switching and warm-up penalties.

## Repository layout

```
src/                     Framework package
  core/                  Config loading, shared types, math utilities
  envs/                  Scheduling environment (constraints, power accounting, episodes)
  sensors/               Sensor models, refresh/warm-up behaviour, power profiles
  estimators/            Kalman-style state estimators
  scheduling/            Policies: RL agents (PPO/DQN variants) and heuristic baselines
  reward/                Reward assembly, forecast-loss reward, oracle utilities
  forecasting/           Forecast model wrappers and training loops
  evaluation/            Metrics, rank analysis, protocol audits
  pipelines/             End-to-end training / evaluation / dataset pipelines
  data_sources/          Truth generators and public-weather synthesis
  business_cases/        Concrete cases (including the wind-blown snow case)
  visualization/         Plotting helpers
  v2/                    Second-generation trainer/evaluator used for the reported runs

scripts/                 Entry points (data generation, training, evaluation, aggregation)
configs/                 YAML configuration for env / sensors / estimator / reward / scheduler
tests/                   Unit tests
results/                 Aggregated result tables for the final experiment grid
reproducibility/         Seed-level aggregates and protocol metadata for the reported runs
```

## Install

Python 3.10+.

```bash
uv sync                     # or: pip install -r requirements.txt
```

## Quick start

```bash
# 1. Generate a synthetic truth series (wind-blown snow case)
python scripts/00_generate_business_data.py --steps 200000

# 2. Train a scheduler
python scripts/01_train_rl_scheduler.py \
    --scheduler_cfg configs/scheduler/ppo.yaml \
    --run_id demo_ppo

# 3. Evaluate the trained scheduler against the baselines
python scripts/02_evaluate_scheduler.py \
    --scheduler_cfg configs/scheduler/ppo.yaml \
    --run_id demo_ppo \
    --checkpoint <run_dir>/checkpoints/last.pt
```

Heuristic baselines (`full_open`, `periodic`, `round_robin`, `random`, `info_priority`,
`warmup_round_robin`) are configured in `configs/scheduler/` and share the same
evaluation path, so they can be compared without retraining.

The scripts under `scripts/` are numbered in the order they were used during the project;
`00`–`11` form the base loop, later numbers are experiment-specific collectors and audits.
`src/v2/custom_ppo.py` holds the PD-PPO agent used in the reported experiments.

## Results and reproducibility

- `results/final_grid_20260830/` — comparison and seed-level metrics for the final grid.
- `reproducibility/pdppo_evidence_20260718/` — 24-seed aggregates for the main evidence
  blocks (validation-frozen macro comparison, matched-reward control, matched DQN control,
  secondary forecaster, mechanism/duty diagnostics), plus the protocol metadata needed to
  interpret them.
- `reproducibility/pdppo_evidence_20260710/` — earlier 24-seed and 6-seed robustness blocks.

All reported numbers come from simulated data. The physical-truth generator scripts can
consume an external automatic-weather-station archive (`--antaws-root`), but no raw field
data is redistributed here.

## Tests

```bash
pytest -q
```

## Citation

If you use this code, please cite the repository (see `CITATION.cff`).

## License

MIT — see `LICENSE`.

## 中文简介

本仓库是“有限功耗约束下多传感器在线调度”的研究框架：在每个时刻只开启部分传感器，
在功耗、切换与预热约束下最小化长期状态估计误差和下游预测损失。包含环境与功耗模型、
卡尔曼类估计器、PPO/DQN 及其 PD-PPO 变体、启发式基线、预测模型封装、评估协议与
聚合结果表。所有实验数据均由仿真生成，不随仓库分发原始观测数据。
