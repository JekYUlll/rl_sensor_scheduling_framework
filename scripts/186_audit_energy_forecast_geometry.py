"""Audit frozen forecast geometry under a documented cumulative-energy state."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.power_projector import PowerConstraintsV2
from v2.rollout import concat_rollout_results, run_policy_rollout
from v2.sensor_spec import load_sensor_specs
from v2.entity_resource_envelope import FIXED_AUXILIARY_POWER_W


def _load_109():
    path = Path(__file__).with_name("109_v32_audit_subset_forecast_geometry.py")
    spec = importlib.util.spec_from_file_location("energy_geometry_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedMaskPolicy:
    def __init__(self, target_mask: np.ndarray, candidates: np.ndarray, name: str) -> None:
        self.target_mask = np.asarray(target_mask, dtype=bool).reshape(-1)
        self.candidates = np.asarray(candidates, dtype=bool)
        self.name = str(name)

    def reset(self) -> None:
        pass

    def act_mask(self, env: object) -> np.ndarray:
        return self.target_mask.copy()


def _resource_starts(truth: pd.DataFrame, count: int, steps: int) -> list[int]:
    """Select resource-diverse starts without inspecting forecast losses.

    A low-SOC-only rule can place every rollout in the same zero-generation
    condition. The audit therefore matches deterministic SOC/generation-rank
    targets using only the resource trace, while keeping starts separated so
    that the windows are not duplicates. This is a geometry sampling rule,
    not a candidate-performance selection rule.
    """
    soc = truth["baseline_soc_wh"].to_numpy(dtype=float)
    harvest = truth["energy_harvest_wh"].to_numpy(dtype=float)
    n = min(len(truth), len(soc), len(harvest))
    valid_n = n - int(steps) + 1
    if valid_n <= 0:
        raise ValueError("truth is shorter than the requested rollout")
    soc = soc[:valid_n]
    harvest = harvest[:n]
    forward_generation = np.convolve(
        harvest, np.ones(int(steps), dtype=float) / float(steps), mode="valid"
    )

    def percentile_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks / max(1.0, float(len(values) - 1))

    soc_rank = percentile_rank(soc)
    generation_rank = percentile_rank(forward_generation)
    target_count = max(1, int(count))
    base_targets = [(0.10, 0.10), (0.50, 0.50), (0.90, 0.90), (0.50, 0.10)]
    targets = [base_targets[i % len(base_targets)] for i in range(target_count)]
    available = np.arange(valid_n, dtype=int)
    starts: list[int] = []
    minimum_separation = max(1, int(steps) // 2)
    for target_soc, target_generation in targets:
        candidates = available[
            np.array(
                [all(abs(int(index) - start) >= minimum_separation for start in starts) for index in available],
                dtype=bool,
            )
        ]
        if len(candidates) == 0:
            candidates = available
        score = np.abs(soc_rank[candidates] - target_soc) + np.abs(
            generation_rank[candidates] - target_generation
        )
        starts.append(int(candidates[int(np.argmin(score))]))
    return starts


def audit_run(
    run_dir: Path,
    out_dir: Path,
    steps: int,
    start_count: int,
    reserve_wh: float,
    trace_root: Path | None = None,
    entity_dynamic_loads: bool = False,
) -> dict:
    base = _load_109()
    meta = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(meta["truth_csv"])
    truth = base.merge_dynamic_resource_trace(truth, meta)
    resource_meta = dict(meta.get("dynamic_resource", {}))
    trace_path = (
        trace_root / f"resource_seed{int(meta.get('seed', -1))}.csv"
        if trace_root is not None
        else Path(resource_meta["trace_csv"])
    )
    trace = pd.read_csv(trace_path)
    harvest_cols = ["time_idx", "energy_harvest_wh", "baseline_soc_wh"]
    trace_columns = harvest_cols + (
        [column for column in trace.columns if column.startswith("resource_power_w_")]
        if entity_dynamic_loads
        else []
    )
    if "time_idx" in truth.columns and "time_idx" in trace.columns:
        truth = truth.drop(columns=[c for c in trace_columns[1:] if c in truth.columns], errors="ignore")
        truth = truth.merge(trace[trace_columns], on="time_idx", how="left", validate="one_to_one", sort=False)
    else:
        for column in trace_columns[1:]:
            truth[column] = trace[column].to_numpy(dtype=float)
    if truth[trace_columns[1:]].isna().any().any():
        raise ValueError("energy trace does not cover every truth row")

    sensors = load_sensor_specs(meta["sensor_cfg"])
    constraints = PowerConstraintsV2(
        max_active=len(sensors),
        per_step_budget=1000.0,
        startup_peak_budget=1000.0,
        required_sensor_ids=("cr1000xe_backbone",),
    )
    diag = base.load_diagnostic_module()
    masks = diag.build_candidate_masks(sensors, constraints, max_candidate_warmup=None)
    if len(masks) != 32:
        raise ValueError(f"expected 32 candidate masks, got {len(masks)}")
    steady_cost_by_candidate = {
        f"candidate_{candidate_idx:03d}": float(
            sum(spec.power_cost for spec, selected in zip(sensors, mask, strict=True) if bool(selected))
        )
        for candidate_idx, mask in enumerate(masks)
    }
    oracle_path = run_dir / str(meta.get("oracle_path", "v2_tcn_oracle.pt"))
    oracle = base.load_oracle(oracle_path, str(meta.get("oracle_type", "tcn")))
    starts = _resource_starts(truth, start_count, steps)
    dynamic_mapping = tuple(
        (spec.sensor_id, f"resource_power_w_{spec.sensor_id}")
        for spec in sensors
        if f"resource_power_w_{spec.sensor_id}" in truth.columns
    )
    if entity_dynamic_loads and len(dynamic_mapping) != len(sensors):
        raise ValueError("entity dynamic trace must contain a power column for every sensor")
    records: list[dict[str, object]] = []
    candidate_support: dict[str, int] = {}

    for candidate_idx, mask in enumerate(masks):
        policy = FixedMaskPolicy(mask, masks, f"candidate_{candidate_idx:03d}")
        for offset, start_idx in enumerate(starts):
            cfg = base.env_config_from_metadata(
                meta,
                truth,
                seed=int(meta.get("seed", 0)) + 1000 + offset,
                episode_len=int(steps),
            )
            cfg = replace(
                cfg,
                energy_account_enabled=True,
                energy_capacity=8640.0,
                initial_energy=float(truth["baseline_soc_wh"].iloc[start_idx]),
                harvest_per_step=0.0,
                energy_harvest_column="energy_harvest_wh",
                reserve_energy=float(reserve_wh),
                energy_step_hours=1.0,
                dynamic_resource_budget_w=None,
                dynamic_resource_power_columns=(),
                include_dynamic_resource_state=False,
                dynamic_resource_fixed_power_w=0.0,
                energy_use_dynamic_resource_cost=False,
            )
            if entity_dynamic_loads:
                cfg = replace(
                    cfg,
                    dynamic_resource_budget_w=55.0,
                    dynamic_resource_power_columns=dynamic_mapping,
                    dynamic_resource_fixed_power_w=float(FIXED_AUXILIARY_POWER_W),
                    energy_use_dynamic_resource_cost=True,
                )
            env = diag.WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
            result = run_policy_rollout(env, policy, steps=int(steps), start_idx=int(start_idx))
            requested = np.all(result.selected_masks == np.asarray(mask, dtype=int), axis=1)
            candidate_support[policy.name] = candidate_support.get(policy.name, 0) + int(np.sum(requested))
            for row_idx, loss, soc, requested_ok in zip(
                result.step_indices,
                result.oracle_losses,
                result.soc,
                requested,
                strict=True,
            ):
                if not bool(requested_ok):
                    continue
                harvest = float(truth["energy_harvest_wh"].iloc[int(row_idx)])
                soc_regime = "low" if soc < 0.25 * 8640.0 else ("mid" if soc < 0.75 * 8640.0 else "high")
                generation_regime = "zero" if harvest <= 1e-9 else ("low" if harvest < 50.0 else "high")
                records.append(
                    {
                        "seed": int(meta.get("seed", -1)),
                        "time_idx": int(row_idx),
                        "candidate": policy.name,
                        "selected_sensor_ids": ";".join(
                            spec.sensor_id for spec, selected in zip(sensors, mask, strict=True) if bool(selected)
                        ),
                        "steady_cost": steady_cost_by_candidate[policy.name],
                        "oracle_loss": float(loss),
                        "soc_wh": float(soc),
                        "energy_harvest_wh": harvest,
                        "resource_condition": f"{generation_regime}_{soc_regime}",
                    }
                )

    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / f"energy_subset_losses_seed{int(meta.get('seed', -1))}.csv", index=False)
    geometry = base.summarize_condition_geometry(
        frame.rename(columns={"resource_condition": "condition"}),
        condition_column="condition",
        epsilons=[0.01, 0.05],
    )
    overall = frame.groupby(["candidate", "selected_sensor_ids"], as_index=False).oracle_loss.mean()
    best = overall.sort_values("oracle_loss").iloc[0] if not overall.empty else None
    summary = {
        "seed": int(meta.get("seed", -1)),
        "candidate_count": int(len(masks)),
        "start_indices": starts,
        "steps_per_start": int(steps),
        "reserve_wh": float(reserve_wh),
        "support_rows": candidate_support,
        "record_rows": int(len(frame)),
        "best_overall_candidate": None if best is None else str(best.candidate),
        "best_overall_sensors": None if best is None else str(best.selected_sensor_ids),
        "best_overall_loss": None if best is None else float(best.oracle_loss),
        "resource_geometry": geometry,
    }
    (out_dir / f"energy_subset_geometry_seed{int(meta.get('seed', -1))}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--start-count", type=int, default=4)
    parser.add_argument("--reserve-wh", type=float, default=1728.0)
    parser.add_argument("--trace-root", type=Path, default=None)
    parser.add_argument("--entity-dynamic-loads", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        audit_run(
            path,
            args.out_dir,
            args.steps,
            args.start_count,
            args.reserve_wh,
            trace_root=args.trace_root,
            entity_dynamic_loads=bool(args.entity_dynamic_loads),
        )
        for path in args.run_dir
    ]
    (args.out_dir / "energy_subset_geometry_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
