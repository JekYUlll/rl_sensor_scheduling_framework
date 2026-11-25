#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.env import WarmupEnvConfig  # noqa: E402
from v2.forecast_eval import forecast_metric_tables, load_oracle_from_metadata  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.sensor_spec import SensorSpecV2, load_sensor_specs  # noqa: E402


REFERENCE_POLICIES = (
    "full_open_unconstrained",
    "feasible_static_projected",
    "custom_ppo",
    "random",
)


@dataclass
class StaticMaskPolicy:
    mask: np.ndarray
    name: str

    def reset(self) -> None:
        pass

    def act_mask(self, env: object) -> np.ndarray:
        del env
        return np.asarray(self.mask, dtype=bool)


def load_train_helpers() -> Any:
    path = ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("_v2_train_ppo_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(path: str | Path, *, run_dir: Path) -> Path:
    source = Path(path)
    if source.exists():
        return source
    for base in (Path.cwd(), ROOT, run_dir, *run_dir.parents):
        candidate = base / source
        if candidate.exists():
            return candidate
    return source


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def parse_budget_seed(run_dir: Path) -> tuple[float, int]:
    budget = float("nan")
    seed = -1
    for part in run_dir.name.split("_"):
        if part.startswith("budget"):
            try:
                budget = float(part.replace("budget", "").replace("p", "."))
            except ValueError:
                pass
        if part.startswith("seed"):
            try:
                seed = int(part.replace("seed", ""))
            except ValueError:
                pass
    return budget, seed


def constraints_from_metadata(metadata: dict[str, Any]) -> PowerConstraintsV2:
    raw = dict(metadata.get("constraints", {}))
    coverage_groups = []
    for item in raw.get("coverage_groups", []):
        coverage_groups.append((str(item.get("name", "")), tuple(str(x) for x in item.get("sensor_ids", []))))
    return PowerConstraintsV2(
        max_active=raw.get("max_active"),
        per_step_budget=raw.get("per_step_budget"),
        startup_peak_budget=raw.get("startup_peak_budget"),
        required_sensor_ids=tuple(str(x) for x in raw.get("required_sensor_ids", [])),
        coverage_groups=tuple(coverage_groups),
    )


def candidate_masks(
    *,
    n_sensors: int,
    subset_sizes: tuple[int, ...],
    max_subsets_per_k: int,
    seed: int,
) -> list[tuple[int, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    out: list[tuple[int, np.ndarray]] = []
    for size in subset_sizes:
        combos = list(combinations(range(int(n_sensors)), int(size)))
        if max_subsets_per_k > 0 and len(combos) > int(max_subsets_per_k):
            selected = rng.choice(len(combos), size=int(max_subsets_per_k), replace=False)
            combos = [combos[int(idx)] for idx in sorted(selected)]
        for combo in combos:
            mask = np.zeros(int(n_sensors), dtype=bool)
            mask[list(combo)] = True
            out.append((int(size), mask))
    return out


def mask_power_summary(
    mask: np.ndarray,
    sensors: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
) -> dict[str, float | bool | int]:
    indices = [int(idx) for idx in np.flatnonzero(np.asarray(mask, dtype=bool))]
    steady = float(sum(float(sensors[idx].power_cost) for idx in indices))
    peak = float(sum(float(sensors[idx].startup_peak_power) for idx in indices))
    max_active_ok = constraints.max_active is None or len(indices) <= int(constraints.max_active)
    steady_ok = constraints.per_step_budget is None or steady <= float(constraints.per_step_budget) + 1e-12
    peak_ok = constraints.startup_peak_budget is None or peak <= float(constraints.startup_peak_budget) + 1e-12
    sensor_ids = tuple(s.sensor_id for s in sensors)
    selected_ids = {sensor_ids[idx] for idx in indices}
    coverage_ok = True
    for _, group_ids in constraints.coverage_groups:
        if not any(str(sensor_id) in selected_ids for sensor_id in group_ids):
            coverage_ok = False
            break
    return {
        "steady_power_cold": steady,
        "startup_peak_power_cold": peak,
        "max_active_ok": bool(max_active_ok),
        "steady_budget_ok": bool(steady_ok),
        "startup_peak_ok": bool(peak_ok),
        "coverage_ok": bool(coverage_ok),
        "budget_feasible_cold": bool(max_active_ok and steady_ok and peak_ok),
        "fully_feasible_cold": bool(max_active_ok and steady_ok and peak_ok and coverage_ok),
    }


def eval_one_mask(
    *,
    helpers: Any,
    truth: pd.DataFrame,
    sensors: list[SensorSpecV2],
    oracle: Any,
    metadata: dict[str, Any],
    start_indices: tuple[int, ...],
    steps: int,
    mask: np.ndarray,
    policy_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_cfg = WarmupEnvConfig(
        state_columns=tuple(metadata.get("reward_state_columns", metadata.get("state_columns", helpers.STATE_COLUMNS))),
        reward_target_columns=tuple(metadata.get("reward_target_columns", helpers.REWARD_TARGET_COLUMNS)),
        lookback=int(metadata.get("lookback", getattr(oracle.cfg, "lookback", 20))),
        episode_len=int(steps),
        seed=int(metadata.get("seed", 0)) + 120_000,
        base_freq_s=int(metadata.get("freq_s", 10800)),
        lambda_warmup_abort=float(metadata.get("reward_shaping", {}).get("lambda_warmup_abort", 0.08)),
        lambda_switch=float(metadata.get("reward_shaping", {}).get("lambda_switch", 0.002)),
    )
    policy = StaticMaskPolicy(mask=np.asarray(mask, dtype=bool), name=str(policy_name))
    result, rollout_stats = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=PowerConstraintsV2(),
        cfg=eval_cfg,
        oracle=oracle,
        policy=policy,
        steps=int(steps),
        start_indices=start_indices,
    )
    # Forecast evaluator accepts loaded rollout objects with a ``policy``
    # attribute. RolloutResult is intentionally lightweight, so attach it here.
    result.policy = str(result.policy_name)
    result.state_columns = tuple(eval_cfg.state_columns)
    result.sensor_ids = tuple(spec.sensor_id for spec in sensors)
    eval_metadata = dict(metadata)
    eval_metadata["eval_start_indices"] = [int(x) for x in start_indices]
    eval_metadata["eval_steps"] = int(steps)
    overall, by_variable, _ = forecast_metric_tables(
        result,
        truth_df=truth,
        oracle=oracle,
        metadata=eval_metadata,
        target_columns=tuple(metadata.get("reward_target_columns", helpers.REWARD_TARGET_COLUMNS)),
        target_weights=tuple(float(x) for x in metadata.get("target_weights", helpers.DEFAULT_TARGET_WEIGHTS)),
        target_scales=tuple(float(x) for x in metadata.get("target_scales", helpers.DEFAULT_TARGET_SCALES)),
    )
    overall.update({f"rollout_{key}": value for key, value in rollout_stats.items() if key != "policy"})
    return overall, by_variable


def read_reference_rows(run_dir: Path, *, budget: float, seed: int) -> pd.DataFrame:
    path = run_dir / "evaluation" / "v2_eval_overall.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    table = table[table["policy"].isin(REFERENCE_POLICIES)].copy()
    if table.empty:
        return table
    table["budget"] = float(budget)
    table["seed"] = int(seed)
    table["reference_group"] = table["policy"]
    return table


def summarise_numeric(table: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    return (
        table.groupby(group_cols, dropna=False)[value_col]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen oracle robustness under controlled partial observation.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="reports/v2_supplement_experiments/E2_oracle_robustness")
    parser.add_argument("--asset-dir", default="reports/v2_supplement_assets")
    parser.add_argument("--subset-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-subsets-per-k", type=int, default=30)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--rollouts", type=int, default=6)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--only-budget-feasible", action="store_true")
    args = parser.parse_args()

    helpers = load_train_helpers()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    subset_rows: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []
    reference_rows: list[pd.DataFrame] = []

    for run_dir_raw in args.run_dirs:
        run_dir = Path(run_dir_raw)
        metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
        budget, seed = parse_budget_seed(run_dir)
        truth = pd.read_csv(resolve_path(metadata["truth_csv"], run_dir=run_dir))
        sensors = load_sensor_specs(resolve_path(metadata.get("sensor_cfg", "configs/sensors/windblown_sensors_balanced.yaml"), run_dir=run_dir))
        constraints = constraints_from_metadata(metadata)
        oracle = load_oracle_from_metadata(metadata, run_dir=run_dir, device=str(args.oracle_device))
        horizon = int(metadata.get("horizon", getattr(oracle.cfg, "horizon", 8)))
        start_indices = tuple(int(x) for x in metadata.get("eval_start_indices", []))
        if not start_indices:
            start_indices = helpers.select_eval_start_indices(
                truth,
                steps=int(args.steps),
                horizon=int(horizon),
                n_rollouts=int(args.rollouts),
                event_fraction=float(metadata.get("eval_event_fraction", 0.67)),
                seed=int(seed) + 70_000,
            )
        else:
            start_indices = start_indices[: int(args.rollouts)]
        reference_rows.append(read_reference_rows(run_dir, budget=budget, seed=seed))

        masks = candidate_masks(
            n_sensors=len(sensors),
            subset_sizes=tuple(int(x) for x in args.subset_sizes),
            max_subsets_per_k=int(args.max_subsets_per_k),
            seed=int(seed) + 77_000,
        )
        full_mask = np.ones(len(sensors), dtype=bool)
        masks.append((len(sensors), full_mask))
        for subset_size, mask in masks:
            power_summary = mask_power_summary(mask, sensors, constraints)
            if bool(args.only_budget_feasible) and not bool(power_summary["budget_feasible_cold"]):
                continue
            sensor_ids = tuple(s.sensor_id for s in sensors)
            selected_ids = tuple(sensor_ids[idx] for idx in np.flatnonzero(mask))
            label_ids = "-".join(selected_ids) if selected_ids else "none"
            policy_name = f"oracle_subset_k{subset_size}_{label_ids}"
            overall, by_variable = eval_one_mask(
                helpers=helpers,
                truth=truth,
                sensors=sensors,
                oracle=oracle,
                metadata=metadata,
                start_indices=start_indices,
                steps=int(args.steps),
                mask=mask,
                policy_name=policy_name,
            )
            row = {
                "run_dir": str(run_dir),
                "budget": float(budget),
                "seed": int(seed),
                "subset_size": int(subset_size),
                "subset_sensor_ids": ";".join(selected_ids),
                "is_full_observation": bool(np.all(mask)),
                **power_summary,
                **overall,
            }
            subset_rows.append(row)
            for item in by_variable:
                variable_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "budget": float(budget),
                        "seed": int(seed),
                        "subset_size": int(subset_size),
                        "subset_sensor_ids": ";".join(selected_ids),
                        "is_full_observation": bool(np.all(mask)),
                        **item,
                    }
                )
            print(
                f"[e2] budget={budget_tag(budget)} seed={seed} k={subset_size} "
                f"fw_mae={float(overall.get('forecast_weighted_mae_overall', math.nan)):.4f} subset={label_ids}",
                flush=True,
            )

    subset_table = pd.DataFrame(subset_rows)
    variable_table = pd.DataFrame(variable_rows)
    references = pd.concat([item for item in reference_rows if not item.empty], ignore_index=True) if reference_rows else pd.DataFrame()
    subset_table.to_csv(out_dir / "e2_oracle_subset_rows.csv", index=False)
    variable_table.to_csv(out_dir / "e2_oracle_subset_by_variable.csv", index=False)
    references.to_csv(out_dir / "e2_reference_policy_rows.csv", index=False)

    subset_stats = summarise_numeric(
        subset_table,
        ["subset_size", "is_full_observation"],
        "forecast_weighted_mae_overall",
    )
    variable_stats = summarise_numeric(
        variable_table,
        ["subset_size", "variable"],
        "forecast_mae",
    )
    reference_stats = summarise_numeric(
        references,
        ["reference_group"],
        "forecast_weighted_mae_overall",
    )
    subset_stats.to_csv(asset_dir / "exp_e2_oracle_robustness_stats.csv", index=False)
    variable_stats.to_csv(asset_dir / "exp_e2_oracle_robustness_by_variable.csv", index=False)
    reference_stats.to_csv(asset_dir / "exp_e2_oracle_reference_stats.csv", index=False)

    manifest = {
        "run_dirs": [str(Path(x)) for x in args.run_dirs],
        "subset_sizes": [int(x) for x in args.subset_sizes],
        "max_subsets_per_k": int(args.max_subsets_per_k),
        "steps": int(args.steps),
        "rollouts": int(args.rollouts),
        "only_budget_feasible": bool(args.only_budget_feasible),
        "outputs": {
            "subset_rows": str(out_dir / "e2_oracle_subset_rows.csv"),
            "subset_stats": str(asset_dir / "exp_e2_oracle_robustness_stats.csv"),
            "reference_stats": str(asset_dir / "exp_e2_oracle_reference_stats.csv"),
        },
    }
    (out_dir / "e2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(asset_dir / "exp_e2_oracle_robustness_stats.csv")
    if not subset_stats.empty:
        print(subset_stats.to_string(index=False))


if __name__ == "__main__":
    main()
