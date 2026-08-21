#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


POLICY_ORDER = [
    "full_open_unconstrained",
    "feasible_static_projected",
    "custom_ppo",
    "round_robin",
    "aoi",
    "random",
]


def parse_run_name(name: str) -> tuple[float, int]:
    # Expected: budget1p70_seed41
    budget_part, seed_part = name.split("_seed", maxsplit=1)
    budget = float(budget_part.replace("budget", "").replace("p", "."))
    seed = int(seed_part)
    return budget, seed


def event_stratum(event_fraction: float, calm_max: float, heavy_min: float) -> str:
    if event_fraction < calm_max:
        return "calm"
    if event_fraction > heavy_min:
        return "event_heavy"
    return "mixed"


def iter_rollout_rows(
    *,
    out_dir: Path,
    window: int,
    calm_max: float,
    heavy_min: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_dir = out_dir / "raw"
    for run_dir in sorted(raw_dir.glob("budget*_seed*")):
        if not run_dir.is_dir():
            continue
        budget, seed = parse_run_name(run_dir.name)
        for policy in POLICY_ORDER:
            rollout_path = run_dir / f"rollout_{policy}.npz"
            if not rollout_path.exists():
                continue
            data = np.load(rollout_path, allow_pickle=True)
            events = np.asarray(data["event_flags"], dtype=float).reshape(-1)
            losses = np.asarray(data["oracle_losses"], dtype=float).reshape(-1)
            n = min(events.size, losses.size)
            usable = (n // window) * window
            if usable <= 0:
                continue
            for start in range(0, usable, window):
                end = start + window
                event_fraction = float(np.nanmean(events[start:end]))
                loss = float(np.nanmean(losses[start:end]))
                rows.append(
                    {
                        "run_tag": run_dir.name,
                        "budget": budget,
                        "seed": seed,
                        "policy": policy,
                        "window_start": start,
                        "window_end": end,
                        "window_size": window,
                        "event_fraction": event_fraction,
                        "stratum": event_stratum(event_fraction, calm_max, heavy_min),
                        "forecast_weighted_mae": loss,
                    }
                )
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return math.nan
    mu = mean(values)
    return float(math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1)))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str, str], list[float]] = {}
    window_counts: dict[tuple[float, str], set[tuple[int, int, int]]] = {}
    for row in rows:
        budget = float(row["budget"])
        stratum = str(row["stratum"])
        policy = str(row["policy"])
        grouped.setdefault((budget, stratum, policy), []).append(
            float(row["forecast_weighted_mae"])
        )
        window_counts.setdefault((budget, stratum), set()).add(
            (
                int(row["seed"]),
                int(row["window_start"]),
                int(row["window_end"]),
            )
        )
    out: list[dict[str, object]] = []
    for key in sorted(grouped):
        budget, stratum, policy = key
        values = grouped[key]
        out.append(
            {
                "budget": budget,
                "stratum": stratum,
                "policy": policy,
                "forecast_weighted_mae_mean": mean(values),
                "forecast_weighted_mae_std": std(values),
                "samples": len(values),
                "unique_windows": len(window_counts.get((budget, stratum), set())),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def budget_check(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stats = {
        (float(row["budget"]), str(row["stratum"]), str(row["policy"])): float(
            row["forecast_weighted_mae_mean"]
        )
        for row in rows
    }
    out: list[dict[str, object]] = []
    for budget in sorted({key[0] for key in stats}):
        for stratum in ["calm", "mixed", "event_heavy"]:
            key = (budget, stratum)
            pdppo = stats.get((*key, "custom_ppo"), math.nan)
            static = stats.get((*key, "feasible_static_projected"), math.nan)
            round_robin = stats.get((*key, "round_robin"), math.nan)
            aoi = stats.get((*key, "aoi"), math.nan)
            random = stats.get((*key, "random"), math.nan)
            full = stats.get((*key, "full_open_unconstrained"), math.nan)
            out.append(
                {
                    "budget": budget,
                    "stratum": stratum,
                    "full_open_best": full
                    == min(x for x in [full, static, pdppo, round_robin, aoi, random] if not math.isnan(x)),
                    "pdppo_mean": pdppo,
                    "static_mean": static,
                    "pdppo_static_gap": (pdppo - static) / static if static and not math.isnan(static) else math.nan,
                    "round_robin_mean": round_robin,
                    "pdppo_beats_round_robin": pdppo < round_robin if not math.isnan(round_robin) else False,
                    "aoi_mean": aoi,
                    "pdppo_beats_aoi": pdppo < aoi if not math.isnan(aoi) else False,
                    "random_mean": random,
                    "pdppo_beats_random": pdppo < random if not math.isnan(random) else False,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect V3.1 S2 event-fraction window strata from completed rollouts."
    )
    parser.add_argument("--out-dir", default="reports/v31_s2_main")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--calm-max", type=float, default=0.25)
    parser.add_argument("--heavy-min", type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = iter_rollout_rows(
        out_dir=out_dir,
        window=int(args.window),
        calm_max=float(args.calm_max),
        heavy_min=float(args.heavy_min),
    )
    stats = aggregate(rows)
    checks = budget_check(stats)

    long_path = out_dir / "v31_s2_event_fraction_long.csv"
    stats_path = out_dir / "v31_s2_event_fraction_stats.csv"
    check_path = out_dir / "v31_s2_event_fraction_check.csv"
    write_csv(long_path, rows)
    write_csv(stats_path, stats)
    write_csv(check_path, checks)

    print(long_path)
    print(stats_path)
    print(check_path)
    for row in checks:
        print(row)


if __name__ == "__main__":
    main()
