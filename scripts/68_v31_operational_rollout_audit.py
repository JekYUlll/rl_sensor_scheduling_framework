#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit event-conditioned loss and sensor duty from operational rollouts."
    )
    parser.add_argument(
        "--base-dir",
        default="reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced",
    )
    parser.add_argument("--budget-label", default="budget1p70")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(41, 51)))
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["custom_ppo", "duty_constrained_validation_selected_static"],
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out-prefix", default="h75_pdppo_vs_deployable_static_10seed")
    return parser.parse_args()


def resolve(path: str) -> Path:
    result = Path(path)
    if not result.is_absolute():
        result = ROOT / result
    return result


def switch_rate(mask: np.ndarray) -> float:
    if mask.shape[0] <= 1:
        return 0.0
    return float(np.abs(np.diff(mask.astype(int), axis=0)).sum() / ((mask.shape[0] - 1) * mask.shape[1]))


def load_rollout(base: Path, budget_label: str, seed: int, policy: str) -> tuple[Path, np.lib.npyio.NpzFile]:
    path = base / "raw" / f"{budget_label}_seed{seed}" / f"rollout_{policy}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, np.load(path, allow_pickle=True)


def sensor_names(data: np.lib.npyio.NpzFile, width: int) -> list[str]:
    if "sensor_ids" not in data:
        return [f"sensor_{idx}" for idx in range(width)]
    return [str(item) for item in data["sensor_ids"].tolist()]


def mask_name(mask: np.ndarray, sensors: list[str]) -> str:
    active = [sensors[idx] for idx, value in enumerate(mask) if int(value) == 1]
    return "|".join(active) if active else "none"


def audit_seed(
    base: Path,
    budget_label: str,
    seed: int,
    policy: str,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    _, data = load_rollout(base, budget_label, seed, policy)
    losses = np.asarray(data["oracle_losses"], dtype=float)
    events = np.asarray(data["event_flags"], dtype=bool)
    masks = np.asarray(data["selected_masks"], dtype=int)
    powers = np.asarray(data["powers"], dtype=float) if "powers" in data else np.full(len(losses), np.nan)
    sensors = sensor_names(data, masks.shape[1])

    if len(losses) != len(events) or len(losses) != len(masks):
        raise ValueError(f"Length mismatch for seed={seed} policy={policy}")

    event_losses = losses[events]
    non_event_losses = losses[~events]
    row = {
        "seed": int(seed),
        "policy": policy,
        "oracle_loss_mean": float(losses.mean()),
        "event_loss_mean": float(event_losses.mean()) if event_losses.size else float("nan"),
        "non_event_loss_mean": float(non_event_losses.mean()) if non_event_losses.size else float("nan"),
        "event_rate": float(events.mean()),
        "power_mean": float(powers.mean()),
        "switches_per_step": switch_rate(masks),
        "unique_masks": int(len({tuple(item) for item in masks})),
    }

    sensor_rows: list[dict[str, object]] = []
    for idx, sensor in enumerate(sensors):
        sensor_mask = masks[:, idx].astype(float)
        sensor_rows.append(
            {
                "seed": int(seed),
                "policy": policy,
                "sensor": sensor,
                "duty": float(sensor_mask.mean()),
                "event_duty": float(sensor_mask[events].mean()) if events.any() else float("nan"),
                "non_event_duty": float(sensor_mask[~events].mean()) if (~events).any() else float("nan"),
                "switch_rate": switch_rate(masks[:, idx : idx + 1]),
            }
        )

    counts = Counter(tuple(item) for item in masks)
    top_rows: list[dict[str, object]] = []
    for rank, (mask, count) in enumerate(counts.most_common(), start=1):
        if rank > int(top_k):
            break
        top_rows.append(
            {
                "seed": int(seed),
                "policy": policy,
                "rank": int(rank),
                "fraction": float(count / len(masks)),
                "count": int(count),
                "sensor_set": mask_name(np.asarray(mask), sensors),
            }
        )
    return row, sensor_rows, top_rows


def main() -> None:
    args = parse_args()
    base = resolve(args.base_dir)

    loss_rows: list[dict[str, object]] = []
    sensor_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for policy in args.policies:
            loss, sensors, top = audit_seed(
                base,
                str(args.budget_label),
                int(seed),
                str(policy),
                int(args.top_k),
            )
            loss_rows.append(loss)
            sensor_rows.extend(sensors)
            top_rows.extend(top)

    loss_df = pd.DataFrame(loss_rows)
    sensor_df = pd.DataFrame(sensor_rows)
    top_df = pd.DataFrame(top_rows)

    loss_path = base / f"{args.out_prefix}_loss_audit.csv"
    sensor_path = base / f"{args.out_prefix}_sensor_audit.csv"
    top_path = base / f"{args.out_prefix}_top_masks.csv"
    loss_df.to_csv(loss_path, index=False)
    sensor_df.to_csv(sensor_path, index=False)
    top_df.to_csv(top_path, index=False)

    print(loss_path.relative_to(ROOT) if loss_path.is_relative_to(ROOT) else loss_path)
    print(sensor_path.relative_to(ROOT) if sensor_path.is_relative_to(ROOT) else sensor_path)
    print(top_path.relative_to(ROOT) if top_path.is_relative_to(ROOT) else top_path)
    print(loss_df.groupby("policy")[["oracle_loss_mean", "event_loss_mean", "non_event_loss_mean", "switches_per_step", "unique_masks"]].mean().to_string())


if __name__ == "__main__":
    main()
