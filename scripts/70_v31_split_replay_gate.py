#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.env import WarmupEnvConfig  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402

SUBTYPE_LABELS = {
    1: "particle",
    2: "flux",
    3: "thermal",
}
SUBTYPE_LOSS_COLUMNS = tuple(f"oracle_loss_subtype_{label}" for label in SUBTYPE_LABELS.values())
MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event"
STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event_staticnorm"
MACRO_SUBTYPE_COUNT_COLUMN = "macro_subtype_event_count"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(path_value: str | Path, *, run_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path, run_dir / path.name, ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve {path_value!r} from {run_dir}")


def sensor_ids_for_mask(sensors: list[Any], mask: np.ndarray) -> str:
    return "|".join(str(sensors[idx].sensor_id) for idx in np.flatnonzero(mask))


def parse_sensor_id_tokens(values: list[str] | None) -> tuple[str, ...]:
    tokens: list[str] = []
    for value in values or []:
        tokens.extend(part for part in str(value).replace(",", " ").split() if part)
    return tuple(tokens)


def mask_from_sensor_ids(
    sensors: list[Any],
    sensor_ids: list[str] | None,
    *,
    label: str,
) -> np.ndarray:
    tokens = parse_sensor_id_tokens(sensor_ids)
    if not tokens:
        raise ValueError(f"{label} requires at least one sensor id")
    id_to_idx = {str(spec.sensor_id): idx for idx, spec in enumerate(sensors)}
    unknown = [sensor_id for sensor_id in tokens if sensor_id not in id_to_idx]
    if unknown:
        known = ", ".join(sorted(id_to_idx))
        raise ValueError(f"Unknown {label} sensor ids {unknown}; known ids: {known}")
    mask = np.zeros(len(sensors), dtype=bool)
    for sensor_id in tokens:
        mask[int(id_to_idx[sensor_id])] = True
    return mask


def candidate_index_for_mask(candidate_masks: np.ndarray, mask: np.ndarray) -> int:
    target = np.asarray(mask, dtype=bool).reshape(-1)
    for idx, candidate in enumerate(np.asarray(candidate_masks, dtype=bool)):
        if np.array_equal(np.asarray(candidate, dtype=bool).reshape(-1), target):
            return int(idx)
    return -1


def loss_split(result: Any) -> tuple[float, float]:
    losses = np.asarray(result.oracle_losses, dtype=float)
    events = np.asarray(result.event_flags, dtype=bool)
    event_losses = losses[events & np.isfinite(losses)]
    non_event_losses = losses[(~events) & np.isfinite(losses)]
    event = float(np.mean(event_losses)) if event_losses.size else float("nan")
    non_event = float(np.mean(non_event_losses)) if non_event_losses.size else float("nan")
    return event, non_event


def append_loss_split(row: dict[str, Any], result: Any) -> dict[str, Any]:
    event, non_event = loss_split(result)
    row["oracle_loss_event"] = event
    row["oracle_loss_non_event"] = non_event
    return row


def append_subtype_loss_split(row: dict[str, Any], result: Any, truth: pd.DataFrame) -> dict[str, Any]:
    losses = np.asarray(result.oracle_losses, dtype=float).reshape(-1)
    step_indices = np.asarray(getattr(result, "step_indices", np.asarray([], dtype=int)), dtype=int).reshape(-1)
    if "event_subtype_id" not in truth.columns or step_indices.size != losses.size:
        for label in SUBTYPE_LABELS.values():
            row[f"oracle_loss_subtype_{label}"] = float("nan")
            row[f"steps_subtype_{label}"] = 0
        row[MACRO_SUBTYPE_LOSS_COLUMN] = float("nan")
        row[MACRO_SUBTYPE_COUNT_COLUMN] = 0
        return row

    valid = (step_indices >= 0) & (step_indices < len(truth))
    subtype_values = np.zeros_like(step_indices, dtype=int)
    subtype_values[valid] = truth["event_subtype_id"].to_numpy(dtype=int)[step_indices[valid]]
    finite = np.isfinite(losses)
    subtype_losses: list[float] = []
    for subtype_id, label in SUBTYPE_LABELS.items():
        mask = (subtype_values == int(subtype_id)) & finite
        subtype_loss = float(np.mean(losses[mask])) if np.any(mask) else float("nan")
        row[f"oracle_loss_subtype_{label}"] = subtype_loss
        row[f"steps_subtype_{label}"] = int(np.sum(subtype_values == int(subtype_id)))
        if np.isfinite(subtype_loss):
            subtype_losses.append(subtype_loss)
    row[MACRO_SUBTYPE_LOSS_COLUMN] = float(np.mean(subtype_losses)) if subtype_losses else float("nan")
    row[MACRO_SUBTYPE_COUNT_COLUMN] = int(len(subtype_losses))
    return row


def best_finite_row(table: pd.DataFrame, score_col: str) -> dict[str, Any]:
    if score_col not in table.columns:
        return {}
    values = pd.to_numeric(table[score_col], errors="coerce")
    candidates = table[np.isfinite(values)].copy()
    if candidates.empty:
        return {}
    candidates[score_col] = pd.to_numeric(candidates[score_col], errors="coerce")
    return candidates.sort_values(score_col).iloc[0].to_dict()


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def finite_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def subtype_static_normalizers(table: pd.DataFrame | None) -> dict[str, float]:
    normalizers: dict[str, float] = {}
    if table is None or table.empty:
        return normalizers
    for col in SUBTYPE_LOSS_COLUMNS:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            normalizers[col] = float(np.median(values))
    return normalizers


def add_staticnorm_macro(table: pd.DataFrame, normalizers: dict[str, float]) -> pd.DataFrame:
    if table.empty or not normalizers:
        return table
    result = table.copy()
    norm_cols: list[str] = []
    for col in SUBTYPE_LOSS_COLUMNS:
        denom = float(normalizers.get(col, float("nan")))
        if col not in result.columns or not np.isfinite(denom) or denom <= 0.0:
            continue
        norm_col = f"{col}_staticnorm"
        result[norm_col] = pd.to_numeric(result[col], errors="coerce") / denom
        norm_cols.append(norm_col)
    if norm_cols:
        result[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN] = result[norm_cols].apply(
            lambda row: finite_mean([float(value) for value in row.to_list()]),
            axis=1,
        )
    return result


class CyclicEventMaskPolicy:
    def __init__(
        self,
        *,
        name: str,
        calm_masks: tuple[np.ndarray, ...],
        event_masks: tuple[np.ndarray, ...],
        lookahead_steps: int,
        dwell_steps: int,
    ) -> None:
        self.name = str(name)
        self.calm_masks = tuple(np.asarray(mask, dtype=bool).reshape(-1) for mask in calm_masks)
        self.event_masks = tuple(np.asarray(mask, dtype=bool).reshape(-1) for mask in event_masks)
        self.lookahead_steps = int(max(0, int(lookahead_steps)))
        self.dwell_steps = int(max(1, int(dwell_steps)))

    def reset(self) -> None:
        return None

    def act_mask(self, env: object) -> np.ndarray:
        event_flags = np.asarray(getattr(env, "event_flags"), dtype=bool)
        current_idx = int(getattr(env, "current_idx"))
        start_idx = int(getattr(env, "episode_start_idx", current_idx))
        end_idx = min(len(event_flags), current_idx + self.lookahead_steps + 1)
        trigger = bool(np.any(event_flags[current_idx:end_idx]))
        masks = self.event_masks if trigger else self.calm_masks
        phase = max(0, current_idx - start_idx) // self.dwell_steps
        return masks[int(phase) % len(masks)].copy()

    def act_scores(self, env: object) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


class SubtypeMaskPolicy:
    def __init__(
        self,
        *,
        name: str,
        subtype_ids: np.ndarray,
        calm_mask: np.ndarray,
        subtype_masks: dict[int, np.ndarray],
        lookahead_steps: int,
    ) -> None:
        self.name = str(name)
        self.subtype_ids = np.asarray(subtype_ids, dtype=int).reshape(-1)
        self.calm_mask = np.asarray(calm_mask, dtype=bool).reshape(-1)
        self.subtype_masks = {
            int(subtype_id): np.asarray(mask, dtype=bool).reshape(-1)
            for subtype_id, mask in subtype_masks.items()
        }
        self.lookahead_steps = int(max(0, int(lookahead_steps)))

    def reset(self) -> None:
        return None

    def act_mask(self, env: object) -> np.ndarray:
        current_idx = int(getattr(env, "current_idx"))
        end_idx = min(len(self.subtype_ids), current_idx + self.lookahead_steps + 1)
        window = self.subtype_ids[current_idx:end_idx]
        active = window[window > 0]
        subtype_id = int(active[0]) if active.size else 0
        return self.subtype_masks.get(subtype_id, self.calm_mask).copy()

    def act_scores(self, env: object) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


def build_eval_cfg(
    *,
    helpers: Any,
    ops: Any,
    metadata: dict[str, Any],
    truth: pd.DataFrame,
    run_dir: Path,
    env_min_dwell_steps: int | None,
    oracle_device: str,
) -> tuple[Any, WarmupEnvConfig, WarmupEnvConfig, int, tuple[int, ...]]:
    oracle = ops.load_oracle(metadata, run_dir=run_dir, device=str(oracle_device))
    norm_mean, norm_std = ops.normalization_stats(
        truth,
        state_columns=helpers.STATE_COLUMNS,
        metadata=metadata,
    )
    eval_steps = ops.infer_eval_steps(metadata, run_dir=run_dir)
    eval_starts = tuple(int(x) for x in metadata["eval_start_indices"])
    env_kwargs = ops.env_kwargs_from_metadata(metadata)
    if env_min_dwell_steps is not None:
        env_kwargs["min_dwell_steps"] = int(max(1, int(env_min_dwell_steps)))
    eval_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=int(metadata["lookback"]),
        episode_len=int(eval_steps),
        seed=int(metadata["seed"]) + 15_000,
        base_freq_s=int(metadata["freq_s"]),
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        **env_kwargs,
    )
    static_cfg = (
        eval_cfg
        if bool(metadata.get("reward_shaping", {}).get("primary_eval_duty_guard", False))
        else replace(eval_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
    )
    return oracle, eval_cfg, static_cfg, int(eval_steps), eval_starts


def evaluate_static_candidates(
    *,
    helpers: Any,
    truth: pd.DataFrame,
    sensors: list[Any],
    constraints: Any,
    cfg: WarmupEnvConfig,
    oracle: Any,
    candidate_masks: np.ndarray,
    eval_steps: int,
    eval_starts: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for action_idx, mask in enumerate(np.asarray(candidate_masks, dtype=bool)):
        policy = StaticMaskPolicy(mask=tuple(bool(x) for x in mask), name=f"static_action{int(action_idx)}")
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            policy=policy,
            steps=int(eval_steps),
            start_indices=eval_starts,
        )
        row = append_loss_split(dict(metrics), result)
        row = append_subtype_loss_split(row, result, truth)
        row["action_idx"] = int(action_idx)
        row["sensor_ids"] = sensor_ids_for_mask(sensors, mask)
        for idx, spec in enumerate(sensors):
            row[f"mask__{spec.sensor_id}"] = int(bool(mask[idx]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("oracle_loss_mean").reset_index(drop=True)


def top_static_masks(
    *,
    static_table: pd.DataFrame,
    candidate_masks: np.ndarray,
    score_col: str,
    top_size: int,
) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], tuple[int, ...]]:
    if score_col not in static_table.columns:
        return tuple(), tuple(), tuple()
    ranked = static_table[np.isfinite(static_table[score_col].to_numpy(dtype=float))].sort_values(score_col)
    masks: list[np.ndarray] = []
    names: list[str] = []
    action_indices: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for _, row in ranked.iterrows():
        action_idx = int(row["action_idx"])
        mask = np.asarray(candidate_masks[action_idx], dtype=bool).reshape(-1)
        key = tuple(int(x) for x in mask)
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
        names.append(str(row["sensor_ids"]))
        action_indices.append(action_idx)
        if len(masks) >= int(top_size):
            break
    return tuple(masks), tuple(names), tuple(action_indices)


def reference_row(source_metrics: pd.DataFrame, pattern: str) -> dict[str, Any]:
    policy_col = "policy" if "policy" in source_metrics.columns else "policy_name"
    regex = re.compile(str(pattern))
    mask = source_metrics[policy_col].astype(str).map(lambda value: bool(regex.search(value)))
    candidates = source_metrics[mask].copy()
    if candidates.empty:
        raise ValueError(f"No reference policies matched pattern: {pattern}")
    row = candidates.sort_values("oracle_loss_mean").iloc[0].to_dict()
    row["policy"] = str(row.get(policy_col, ""))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-split cyclic replay gate before PPO training.")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--env-min-dwell-steps", type=int, default=None)
    parser.add_argument("--top-sizes", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--lead-steps", nargs="+", type=int, default=[0, 3, 6])
    parser.add_argument("--dwell-steps", nargs="+", type=int, default=[6, 12, 24])
    parser.add_argument(
        "--replay-family",
        choices=["event_cyclic", "subtype_auto", "subtype_explicit"],
        default="event_cyclic",
        help="Replay schedule family. event_cyclic preserves the historical event/calm cyclic gate.",
    )
    parser.add_argument(
        "--explicit-calm-sensors",
        nargs="*",
        default=None,
        help="Sensor ids for subtype_explicit calm/non-event steps. Comma or space separated.",
    )
    parser.add_argument(
        "--explicit-particle-sensors",
        nargs="*",
        default=None,
        help="Sensor ids for subtype_explicit particle events. Comma or space separated.",
    )
    parser.add_argument(
        "--explicit-flux-sensors",
        nargs="*",
        default=None,
        help="Sensor ids for subtype_explicit flux events. Comma or space separated.",
    )
    parser.add_argument(
        "--explicit-thermal-sensors",
        nargs="*",
        default=None,
        help=(
            "Sensor ids for subtype_explicit thermal events. If omitted, the calm mask is used. "
            "Comma or space separated."
        ),
    )
    parser.add_argument(
        "--explicit-policy-name",
        default="split_subtype_explicit",
        help="Policy name prefix for subtype_explicit replay rows.",
    )
    parser.add_argument(
        "--subtype-top-size-cap",
        type=int,
        default=2,
        help="Maximum per-subtype pool size for subtype_auto to avoid a large Cartesian product.",
    )
    parser.add_argument("--min-margin-abs", type=float, default=0.005)
    parser.add_argument("--min-margin-rel", type=float, default=0.01)
    parser.add_argument(
        "--macro-min-margin-abs",
        type=float,
        default=None,
        help="Absolute margin required for the event-subtype macro replay gate. Defaults to --min-margin-abs.",
    )
    parser.add_argument(
        "--macro-min-margin-rel",
        type=float,
        default=None,
        help="Relative margin required for the event-subtype macro replay gate. Defaults to --min-margin-rel.",
    )
    parser.add_argument(
        "--macro-score-column",
        choices=[MACRO_SUBTYPE_LOSS_COLUMN, STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN],
        default=MACRO_SUBTYPE_LOSS_COLUMN,
        help="Macro score column used for best replay/static reference and macro gate.",
    )
    parser.add_argument(
        "--reference-policy-regex",
        default=(
            "^(duty_constrained_.*|feasible_static_projected|"
            "validation_selected_static|round_robin|aoi)$"
        ),
    )
    parser.add_argument(
        "--enforce-static-candidate-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the best replay policy to beat the replay-local best "
            "static candidate as well as the source metrics reference."
        ),
    )
    parser.add_argument(
        "--static-reference-duty-guard",
        choices=["metadata", "on", "off"],
        default="metadata",
        help=(
            "Duty-guard contract used when recomputing replay-local static candidates. "
            "'metadata' preserves historical behavior; 'off' evaluates true fixed static "
            "subsets without hard duty forcing."
        ),
    )
    parser.add_argument("--max-candidate-warmup", type=int, default=-1)
    args = parser.parse_args()

    helpers = load_module(ROOT / "scripts" / "23_v2_train_ppo.py", "_v31_split_replay_gate_helpers")
    ops = load_module(
        ROOT / "scripts" / "64_v31_eval_saved_run_operational_baselines.py",
        "_v31_split_replay_gate_ops",
    )

    source_run_dir = Path(args.source_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source_run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth = helpers.ensure_state_columns(pd.read_csv(resolve(str(metadata["truth_csv"]), run_dir=source_run_dir)))
    sensors = load_sensor_specs(resolve(str(metadata["sensor_cfg"]), run_dir=source_run_dir))
    constraints = ops.constraints_from_metadata(metadata)
    oracle, eval_cfg, static_cfg, eval_steps, eval_starts = build_eval_cfg(
        helpers=helpers,
        ops=ops,
        metadata=metadata,
        truth=truth,
        run_dir=source_run_dir,
        env_min_dwell_steps=args.env_min_dwell_steps,
        oracle_device=str(args.oracle_device),
    )
    if str(args.static_reference_duty_guard) == "on":
        static_cfg = eval_cfg
    elif str(args.static_reference_duty_guard) == "off":
        static_cfg = replace(eval_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
    candidate_masks = helpers.build_projected_candidate_masks(
        sensors,
        constraints,
        max_candidate_warmup=None if int(args.max_candidate_warmup) < 0 else int(args.max_candidate_warmup),
    )
    static_table = evaluate_static_candidates(
        helpers=helpers,
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=static_cfg,
        oracle=oracle,
        candidate_masks=candidate_masks,
        eval_steps=int(eval_steps),
        eval_starts=eval_starts,
    )
    static_normalizers = subtype_static_normalizers(static_table)
    static_table = add_staticnorm_macro(static_table, static_normalizers)
    static_table.to_csv(out_dir / "split_static_candidate_event_table.csv", index=False)

    replay_rows: list[dict[str, Any]] = []
    if str(args.replay_family) == "event_cyclic":
        event_ranked = static_table[np.isfinite(static_table["oracle_loss_event"].to_numpy(dtype=float))].sort_values(
            "oracle_loss_event"
        )
        calm_ranked = static_table[
            np.isfinite(static_table["oracle_loss_non_event"].to_numpy(dtype=float))
        ].sort_values("oracle_loss_non_event")
        if event_ranked.empty or calm_ranked.empty:
            raise ValueError("Cannot build replay pools: missing finite event/non-event static candidates")

        for top_size in sorted({max(1, int(x)) for x in args.top_sizes}):
            calm_rows = calm_ranked.head(top_size)
            event_rows = event_ranked.head(top_size)
            calm_masks = tuple(
                np.asarray(candidate_masks[int(row["action_idx"])], dtype=bool) for _, row in calm_rows.iterrows()
            )
            event_masks = tuple(
                np.asarray(candidate_masks[int(row["action_idx"])], dtype=bool) for _, row in event_rows.iterrows()
            )
            calm_names = ";".join(str(row["sensor_ids"]) for _, row in calm_rows.iterrows())
            event_names = ";".join(str(row["sensor_ids"]) for _, row in event_rows.iterrows())
            for lead in sorted({max(0, int(x)) for x in args.lead_steps}):
                for dwell in sorted({max(1, int(x)) for x in args.dwell_steps}):
                    name = f"split_top{int(top_size)}_l{int(lead)}_dwell{int(dwell)}"
                    policy = CyclicEventMaskPolicy(
                        name=name,
                        calm_masks=calm_masks,
                        event_masks=event_masks,
                        lookahead_steps=int(lead),
                        dwell_steps=int(dwell),
                    )
                    result, metrics = helpers.evaluate_score_policy_over_starts(
                        truth=truth,
                        sensors=sensors,
                        constraints=constraints,
                        cfg=eval_cfg,
                        oracle=oracle,
                        policy=policy,
                        steps=int(eval_steps),
                        start_indices=eval_starts,
                    )
                    row = append_loss_split(dict(metrics), result)
                    row = append_subtype_loss_split(row, result, truth)
                    row["replay_family"] = str(args.replay_family)
                    row["top_size"] = int(top_size)
                    row["lead_steps"] = int(lead)
                    row["dwell_steps"] = int(dwell)
                    row["calm_pool_sensor_ids"] = calm_names
                    row["event_pool_sensor_ids"] = event_names
                    replay_rows.append(row)
                    save_rollout_npz(
                        out_dir / f"rollout_{name}.npz",
                        result,
                        sensor_ids=[str(spec.sensor_id) for spec in sensors],
                        state_columns=helpers.STATE_COLUMNS,
                    )
    elif str(args.replay_family) == "subtype_auto":
        if "event_subtype_id" not in truth.columns:
            raise ValueError("subtype_auto replay requires truth column event_subtype_id")
        subtype_ids = truth["event_subtype_id"].to_numpy(dtype=int)
        subtype_score_cols = {
            1: "oracle_loss_subtype_particle",
            2: "oracle_loss_subtype_flux",
            3: "oracle_loss_subtype_thermal",
        }
        seen_specs: set[tuple[int, ...]] = set()
        for requested_top_size in sorted({max(1, int(x)) for x in args.top_sizes}):
            top_size = min(int(requested_top_size), max(1, int(args.subtype_top_size_cap)))
            calm_masks, calm_names, calm_action_indices = top_static_masks(
                static_table=static_table,
                candidate_masks=candidate_masks,
                score_col="oracle_loss_non_event",
                top_size=top_size,
            )
            if not calm_masks:
                raise ValueError("Cannot build subtype_auto replay: no finite calm static candidates")
            subtype_pools: dict[int, tuple[tuple[np.ndarray, ...], tuple[str, ...], tuple[int, ...]]] = {}
            for subtype_id, score_col in subtype_score_cols.items():
                masks, names, action_indices = top_static_masks(
                    static_table=static_table,
                    candidate_masks=candidate_masks,
                    score_col=score_col,
                    top_size=top_size,
                )
                if not masks:
                    masks, names, action_indices = calm_masks[:1], calm_names[:1], calm_action_indices[:1]
                subtype_pools[int(subtype_id)] = (masks, names, action_indices)

            particle_masks, particle_names, particle_action_indices = subtype_pools[1]
            flux_masks, flux_names, flux_action_indices = subtype_pools[2]
            thermal_masks, thermal_names, thermal_action_indices = subtype_pools[3]
            for lead in sorted({max(0, int(x)) for x in args.lead_steps}):
                for calm_idx, calm_mask in enumerate(calm_masks):
                    for particle_idx, particle_mask in enumerate(particle_masks):
                        for flux_idx, flux_mask in enumerate(flux_masks):
                            for thermal_idx, thermal_mask in enumerate(thermal_masks):
                                spec_key = tuple(
                                    int(x)
                                    for mask in (calm_mask, particle_mask, flux_mask, thermal_mask)
                                    for x in np.asarray(mask, dtype=bool).astype(int).tolist()
                                ) + (int(lead),)
                                if spec_key in seen_specs:
                                    continue
                                seen_specs.add(spec_key)
                                name = (
                                    "split_subtype_auto_"
                                    f"top{int(top_size)}_c{int(calm_idx)}_p{int(particle_idx)}_"
                                    f"f{int(flux_idx)}_t{int(thermal_idx)}_l{int(lead)}"
                                )
                                policy = SubtypeMaskPolicy(
                                    name=name,
                                    subtype_ids=subtype_ids,
                                    calm_mask=calm_mask,
                                    subtype_masks={
                                        1: particle_mask,
                                        2: flux_mask,
                                        3: thermal_mask,
                                    },
                                    lookahead_steps=int(lead),
                                )
                                result, metrics = helpers.evaluate_score_policy_over_starts(
                                    truth=truth,
                                    sensors=sensors,
                                    constraints=constraints,
                                    cfg=eval_cfg,
                                    oracle=oracle,
                                    policy=policy,
                                    steps=int(eval_steps),
                                    start_indices=eval_starts,
                                )
                                row = append_loss_split(dict(metrics), result)
                                row = append_subtype_loss_split(row, result, truth)
                                row["replay_family"] = str(args.replay_family)
                                row["requested_top_size"] = int(requested_top_size)
                                row["top_size"] = int(top_size)
                                row["lead_steps"] = int(lead)
                                row["dwell_steps"] = 0
                                row["calm_action_idx"] = int(calm_action_indices[calm_idx])
                                row["particle_action_idx"] = int(particle_action_indices[particle_idx])
                                row["flux_action_idx"] = int(flux_action_indices[flux_idx])
                                row["thermal_action_idx"] = int(thermal_action_indices[thermal_idx])
                                row["calm_sensor_ids"] = calm_names[calm_idx]
                                row["particle_sensor_ids"] = particle_names[particle_idx]
                                row["flux_sensor_ids"] = flux_names[flux_idx]
                                row["thermal_sensor_ids"] = thermal_names[thermal_idx]
                                row["calm_pool_sensor_ids"] = ";".join(calm_names)
                                row["particle_pool_sensor_ids"] = ";".join(particle_names)
                                row["flux_pool_sensor_ids"] = ";".join(flux_names)
                                row["thermal_pool_sensor_ids"] = ";".join(thermal_names)
                                replay_rows.append(row)
                                save_rollout_npz(
                                    out_dir / f"rollout_{name}.npz",
                                    result,
                                    sensor_ids=[str(spec.sensor_id) for spec in sensors],
                                    state_columns=helpers.STATE_COLUMNS,
                                )
    elif str(args.replay_family) == "subtype_explicit":
        if "event_subtype_id" not in truth.columns:
            raise ValueError("subtype_explicit replay requires truth column event_subtype_id")
        subtype_ids = truth["event_subtype_id"].to_numpy(dtype=int)
        calm_mask = mask_from_sensor_ids(
            sensors,
            args.explicit_calm_sensors,
            label="--explicit-calm-sensors",
        )
        particle_mask = mask_from_sensor_ids(
            sensors,
            args.explicit_particle_sensors,
            label="--explicit-particle-sensors",
        )
        flux_mask = mask_from_sensor_ids(
            sensors,
            args.explicit_flux_sensors,
            label="--explicit-flux-sensors",
        )
        thermal_mask = (
            mask_from_sensor_ids(
                sensors,
                args.explicit_thermal_sensors,
                label="--explicit-thermal-sensors",
            )
            if parse_sensor_id_tokens(args.explicit_thermal_sensors)
            else calm_mask.copy()
        )
        mask_specs = {
            "calm": calm_mask,
            "particle": particle_mask,
            "flux": flux_mask,
            "thermal": thermal_mask,
        }
        action_indices = {
            label: candidate_index_for_mask(candidate_masks, mask)
            for label, mask in mask_specs.items()
        }
        for lead in sorted({max(0, int(x)) for x in args.lead_steps}):
            name = f"{str(args.explicit_policy_name)}_l{int(lead)}"
            policy = SubtypeMaskPolicy(
                name=name,
                subtype_ids=subtype_ids,
                calm_mask=calm_mask,
                subtype_masks={
                    1: particle_mask,
                    2: flux_mask,
                    3: thermal_mask,
                },
                lookahead_steps=int(lead),
            )
            result, metrics = helpers.evaluate_score_policy_over_starts(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=eval_cfg,
                oracle=oracle,
                policy=policy,
                steps=int(eval_steps),
                start_indices=eval_starts,
            )
            row = append_loss_split(dict(metrics), result)
            row = append_subtype_loss_split(row, result, truth)
            row["replay_family"] = str(args.replay_family)
            row["requested_top_size"] = 0
            row["top_size"] = 0
            row["lead_steps"] = int(lead)
            row["dwell_steps"] = 0
            row["calm_action_idx"] = int(action_indices["calm"])
            row["particle_action_idx"] = int(action_indices["particle"])
            row["flux_action_idx"] = int(action_indices["flux"])
            row["thermal_action_idx"] = int(action_indices["thermal"])
            row["calm_sensor_ids"] = sensor_ids_for_mask(sensors, calm_mask)
            row["particle_sensor_ids"] = sensor_ids_for_mask(sensors, particle_mask)
            row["flux_sensor_ids"] = sensor_ids_for_mask(sensors, flux_mask)
            row["thermal_sensor_ids"] = sensor_ids_for_mask(sensors, thermal_mask)
            row["calm_pool_sensor_ids"] = row["calm_sensor_ids"]
            row["particle_pool_sensor_ids"] = row["particle_sensor_ids"]
            row["flux_pool_sensor_ids"] = row["flux_sensor_ids"]
            row["thermal_pool_sensor_ids"] = row["thermal_sensor_ids"]
            replay_rows.append(row)
            save_rollout_npz(
                out_dir / f"rollout_{name}.npz",
                result,
                sensor_ids=[str(spec.sensor_id) for spec in sensors],
                state_columns=helpers.STATE_COLUMNS,
            )

    replay_table = pd.DataFrame(replay_rows)
    replay_table = add_staticnorm_macro(replay_table, static_normalizers)
    replay_table = replay_table.sort_values("oracle_loss_mean").reset_index(drop=True)
    source_metrics_path = source_run_dir / "v2_custom_ppo_metrics.csv"
    if not source_metrics_path.exists():
        raise FileNotFoundError(source_metrics_path)
    source_metrics = pd.read_csv(source_metrics_path)
    ref = reference_row(source_metrics, str(args.reference_policy_regex))
    ref_loss = float(ref["oracle_loss_mean"])
    static_ref = static_table.iloc[0].to_dict()
    static_ref_policy = str(static_ref.get("policy", ""))
    static_ref_loss = float(static_ref["oracle_loss_mean"])
    best_replay = replay_table.iloc[0].to_dict()
    best_loss = float(best_replay["oracle_loss_mean"])
    macro_score_col = str(args.macro_score_column)
    static_macro_ref = best_finite_row(static_table, macro_score_col)
    replay_macro_ref = best_finite_row(replay_table, macro_score_col)
    static_macro_ref_policy = str(static_macro_ref.get("policy", ""))
    static_macro_ref_loss = finite_float(static_macro_ref.get(macro_score_col))
    replay_macro_ref_policy = str(replay_macro_ref.get("policy", ""))
    replay_macro_ref_loss = finite_float(replay_macro_ref.get(macro_score_col))
    margin_abs = ref_loss - best_loss
    margin_rel = margin_abs / ref_loss if ref_loss > 0 else float("nan")
    static_margin_abs = static_ref_loss - best_loss
    static_margin_rel = static_margin_abs / static_ref_loss if static_ref_loss > 0 else float("nan")
    required_margin = max(float(args.min_margin_abs), float(args.min_margin_rel) * ref_loss)
    static_required_margin = max(float(args.min_margin_abs), float(args.min_margin_rel) * static_ref_loss)
    macro_min_margin_abs = float(args.macro_min_margin_abs if args.macro_min_margin_abs is not None else args.min_margin_abs)
    macro_min_margin_rel = float(args.macro_min_margin_rel if args.macro_min_margin_rel is not None else args.min_margin_rel)
    static_macro_margin_abs = static_macro_ref_loss - replay_macro_ref_loss
    static_macro_margin_rel = (
        static_macro_margin_abs / static_macro_ref_loss
        if np.isfinite(static_macro_ref_loss) and static_macro_ref_loss > 0
        else float("nan")
    )
    static_macro_required_margin = (
        max(float(macro_min_margin_abs), float(macro_min_margin_rel) * static_macro_ref_loss)
        if np.isfinite(static_macro_ref_loss)
        else float("nan")
    )
    source_gate_pass = bool(np.isfinite(margin_abs) and margin_abs >= required_margin)
    static_gate_pass = bool(np.isfinite(static_margin_abs) and static_margin_abs >= static_required_margin)
    static_macro_positive_pass = bool(np.isfinite(static_macro_margin_abs) and static_macro_margin_abs > 0.0)
    static_macro_gate_pass = bool(
        np.isfinite(static_macro_margin_abs)
        and np.isfinite(static_macro_required_margin)
        and static_macro_margin_abs >= static_macro_required_margin
    )
    gate_pass = bool(source_gate_pass and (static_gate_pass or not bool(args.enforce_static_candidate_reference)))
    replay_table["reference_policy"] = str(ref["policy"])
    replay_table["reference_oracle_loss_mean"] = ref_loss
    replay_table["margin_abs_vs_reference"] = ref_loss - replay_table["oracle_loss_mean"].astype(float)
    replay_table["margin_rel_vs_reference"] = replay_table["margin_abs_vs_reference"] / ref_loss
    replay_table["source_reference_gate_pass"] = replay_table["margin_abs_vs_reference"] >= required_margin
    replay_table["static_reference_policy"] = static_ref_policy
    replay_table["static_reference_oracle_loss_mean"] = static_ref_loss
    replay_table["margin_abs_vs_static_reference"] = static_ref_loss - replay_table["oracle_loss_mean"].astype(float)
    replay_table["margin_rel_vs_static_reference"] = replay_table["margin_abs_vs_static_reference"] / static_ref_loss
    replay_table["static_reference_gate_pass"] = replay_table["margin_abs_vs_static_reference"] >= static_required_margin
    replay_table["static_macro_reference_policy"] = static_macro_ref_policy
    replay_table["static_macro_reference_oracle_loss_macro_subtype_event"] = static_macro_ref_loss
    replay_table["static_macro_reference_score_column"] = macro_score_col
    replay_table["margin_abs_vs_static_macro_reference"] = (
        static_macro_ref_loss - replay_table[macro_score_col].astype(float)
        if macro_score_col in replay_table.columns
        else float("nan")
    )
    replay_table["margin_rel_vs_static_macro_reference"] = (
        replay_table["margin_abs_vs_static_macro_reference"] / static_macro_ref_loss
        if np.isfinite(static_macro_ref_loss) and static_macro_ref_loss > 0
        else float("nan")
    )
    replay_table["static_macro_positive_pass"] = replay_table["margin_abs_vs_static_macro_reference"] > 0.0
    replay_table["static_macro_reference_gate_pass"] = (
        replay_table["margin_abs_vs_static_macro_reference"] >= static_macro_required_margin
        if np.isfinite(static_macro_required_margin)
        else False
    )
    if bool(args.enforce_static_candidate_reference):
        replay_table["gate_pass"] = replay_table["source_reference_gate_pass"] & replay_table["static_reference_gate_pass"]
    else:
        replay_table["gate_pass"] = replay_table["source_reference_gate_pass"]
    replay_table.to_csv(out_dir / "split_replay_gate_metrics.csv", index=False)
    source_metrics.to_csv(out_dir / "source_run_metrics.csv", index=False)

    summary = {
        "source_run_dir": str(source_run_dir),
        "replay_family": str(args.replay_family),
        "eval_steps": int(eval_steps),
        "eval_start_indices": [int(x) for x in eval_starts],
        "reference_policy": str(ref["policy"]),
        "reference_oracle_loss_mean": ref_loss,
        "required_margin_abs": required_margin,
        "source_reference_gate_pass": source_gate_pass,
        "static_reference_policy": static_ref_policy,
        "static_reference_oracle_loss_mean": static_ref_loss,
        "static_required_margin_abs": static_required_margin,
        "best_static_reference_event_loss": float(static_ref.get("oracle_loss_event", float("nan"))),
        "best_static_reference_non_event_loss": float(static_ref.get("oracle_loss_non_event", float("nan"))),
        "static_macro_reference_policy": static_macro_ref_policy,
        "static_macro_reference_oracle_loss_macro_subtype_event": static_macro_ref_loss,
        "macro_score_column": macro_score_col,
        "static_macro_reference_score": static_macro_ref_loss,
        "static_macro_required_margin_abs": static_macro_required_margin,
        "macro_min_margin_abs": macro_min_margin_abs,
        "macro_min_margin_rel": macro_min_margin_rel,
        "enforce_static_candidate_reference": bool(args.enforce_static_candidate_reference),
        "static_reference_duty_guard": str(args.static_reference_duty_guard),
        "best_replay_policy": str(best_replay["policy"]),
        "best_replay_oracle_loss_mean": best_loss,
        "best_replay_event_loss": float(best_replay.get("oracle_loss_event", float("nan"))),
        "best_replay_non_event_loss": float(best_replay.get("oracle_loss_non_event", float("nan"))),
        "best_replay_macro_subtype_policy": replay_macro_ref_policy,
        "best_replay_oracle_loss_macro_subtype_event": replay_macro_ref_loss,
        "best_replay_macro_score": replay_macro_ref_loss,
        "margin_abs_vs_reference": float(margin_abs),
        "margin_rel_vs_reference": float(margin_rel),
        "margin_abs_vs_static_reference": float(static_margin_abs),
        "margin_rel_vs_static_reference": float(static_margin_rel),
        "margin_abs_vs_static_macro_reference": float(static_macro_margin_abs),
        "margin_rel_vs_static_macro_reference": float(static_macro_margin_rel),
        "static_reference_gate_pass": static_gate_pass,
        "static_macro_positive_pass": static_macro_positive_pass,
        "static_macro_reference_gate_pass": static_macro_gate_pass,
        "gate_pass": gate_pass,
        "top_sizes": [int(x) for x in args.top_sizes],
        "subtype_top_size_cap": int(args.subtype_top_size_cap),
        "lead_steps": [int(x) for x in args.lead_steps],
        "dwell_steps": [int(x) for x in args.dwell_steps],
    }
    (out_dir / "split_replay_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(replay_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
