#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for headless collectors.
    plt = None


A1_VARIANTS = (
    "full",
    "no_action_emb",
    "no_event_critic",
    "no_awbc",
    "no_oracle",
    "no_awbc_oracle",
    "no_action_mask",
    "masked_only",
)

A1_LABELS = {
    "full": "Full PD-PPO",
    "no_action_emb": "No ActionEmbedding",
    "no_event_critic": "No EventAwareCritic",
    "no_awbc": "No AWBC",
    "no_oracle": "No oracle prior",
    "no_awbc_oracle": "No AWBC/prior",
    "no_action_mask": "No action mask",
    "masked_only": "MaskedActor only",
}

A2_STAGES = (
    "D1_masked_actor_action_embedding",
    "D2_plus_event_critic",
    "D3_plus_awbc",
    "D4_plus_oracle_prior_full",
)

A2_LABELS = {
    "D1_masked_actor_action_embedding": "D1 MaskedActor + ActionEmbedding",
    "D2_plus_event_critic": "D2 + EventAwareCritic",
    "D3_plus_awbc": "D3 + AWBC",
    "D4_plus_oracle_prior_full": "D4 + oracle prior (full)",
}

BONFERRONI_A1_ALPHA = 0.05 / 7.0
BASELINE_AWBC = 0.1
BASELINE_KL = 1.0


def budget_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def value_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def metric_col(df: pd.DataFrame) -> str:
    for candidate in (
        "forecast_weighted_mae_overall",
        "weighted_normalized_mae",
        "obs_reconstruction_mae",
        "mae",
        "oracle_loss_mean",
    ):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No supported score column found in columns={list(df.columns)}")


def read_policy_score(path: Path, *, policy: str = "custom_ppo") -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "policy" in df.columns:
        df = df[df["policy"].astype(str) == str(policy)]
    if df.empty:
        return None
    col = metric_col(df)
    value = float(df.iloc[0][col])
    return value if np.isfinite(value) else None


def read_run_score(run_dir: Path, *, policy: str = "custom_ppo") -> float | None:
    for rel_path in ("evaluation/v2_eval_overall.csv", "v2_custom_ppo_metrics.csv"):
        score = read_policy_score(run_dir / rel_path, policy=policy)
        if score is not None:
            return score
    return None


def run_dir(
    run_root: Path,
    *,
    experiment: str,
    variant: str,
    budget: float,
    seed: int,
) -> Path:
    return (
        run_root
        / "raw"
        / experiment
        / variant
        / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
    )


def paired_wilcoxon(reference: pd.Series, variant: pd.Series) -> tuple[float, str]:
    paired = pd.concat([reference.rename("ref"), variant.rename("variant")], axis=1).dropna()
    if len(paired) < 2:
        return float("nan"), "insufficient_n"
    diffs = paired["variant"].to_numpy(dtype=float) - paired["ref"].to_numpy(dtype=float)
    if np.all(np.abs(diffs) <= 1e-12):
        return 1.0, "all_zero_diff"
    try:
        from scipy.stats import wilcoxon

        _, p_value = wilcoxon(paired["variant"], paired["ref"], alternative="two-sided", zero_method="wilcox")
        return float(p_value), "scipy_wilcoxon"
    except Exception:
        nonzero = diffs[np.abs(diffs) > 1e-12]
        if nonzero.size == 0:
            return 1.0, "all_zero_diff"
        k = int(min(np.sum(nonzero > 0), np.sum(nonzero < 0)))
        n = int(nonzero.size)
        p_value = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
        return float(min(1.0, p_value)), "fallback_sign_test"


def bootstrap_ci(values: np.ndarray, *, seed: int, n_bootstrap: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def load_a2_full_reference(run_root: Path, s2_main_root: Path, *, budget: float, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        source = "aligned_a2_d4"
        score = read_run_score(
            run_dir(
                run_root,
                experiment="A2_staged_v31_aligned",
                variant="D4_plus_oracle_prior_full",
                budget=budget,
                seed=int(seed),
            )
        )
        path_label = "A2_staged_v31_aligned/D4_plus_oracle_prior_full"
        if score is None:
            source = "s2_main_fallback"
            fallback_dir = s2_main_root / "raw" / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
            score = read_run_score(fallback_dir)
            path_label = str(fallback_dir)
        if score is None:
            continue
        rows.append(
            {
                "variant": "full",
                "stage": "D4_plus_oracle_prior_full",
                "seed": int(seed),
                "budget": float(budget),
                "fw_mae": float(score),
                "source": source,
                "source_path": path_label,
            }
        )
    return pd.DataFrame(rows)


def collect_variant_rows(
    run_root: Path,
    *,
    experiment: str,
    variant: str,
    budget: float,
    seeds: list[int],
    extra: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = run_dir(run_root, experiment=experiment, variant=variant, budget=budget, seed=int(seed))
        score = read_run_score(path)
        if score is None:
            continue
        row = {
            "variant": variant,
            "seed": int(seed),
            "budget": float(budget),
            "fw_mae": float(score),
            "source": "aligned_run",
            "source_path": str(path),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_variants(
    raw: pd.DataFrame,
    *,
    variant_order: tuple[str, ...],
    label_map: dict[str, str],
    reference_variant: str,
    bonferroni_alpha: float | None,
    n_bootstrap: int,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    reference = raw[raw["variant"] == reference_variant].set_index("seed")["fw_mae"]
    ref_mean = float(reference.mean()) if len(reference) else float("nan")
    rows = []
    for idx, variant in enumerate(variant_order):
        group = raw[raw["variant"] == variant]
        if group.empty:
            continue
        values = group["fw_mae"].to_numpy(dtype=float)
        mean = float(np.mean(values))
        ci_lower, ci_upper = bootstrap_ci(values, seed=31_000 + idx, n_bootstrap=n_bootstrap)
        if variant == reference_variant:
            p_value = float("nan")
            test_method = "reference"
        else:
            p_value, test_method = paired_wilcoxon(reference, group.set_index("seed")["fw_mae"])
        row = {
            "variant": variant,
            "label": label_map.get(variant, variant),
            "budget": float(group["budget"].iloc[0]),
            "mean": mean,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n": int(len(values)),
            "delta_vs_reference": float(mean - ref_mean) if np.isfinite(ref_mean) else float("nan"),
            "delta_pct_vs_reference": float((mean - ref_mean) / ref_mean * 100.0) if ref_mean else float("nan"),
            "p_vs_reference": float(p_value),
            "test_method": test_method,
        }
        if bonferroni_alpha is not None:
            row["bonferroni_alpha"] = float(bonferroni_alpha)
            row["significant"] = bool(np.isfinite(p_value) and p_value < float(bonferroni_alpha))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_a1(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = Path(args.run_root)
    s2_main_root = Path(args.s2_main_root)
    seeds = [int(seed) for seed in args.a1_seeds]
    frames = [load_a2_full_reference(run_root, s2_main_root, budget=float(args.focus_budget), seeds=seeds)]
    for variant in A1_VARIANTS:
        if variant == "full":
            continue
        frames.append(
            collect_variant_rows(
                run_root,
                experiment="A1_remove_one_v31_aligned",
                variant=variant,
                budget=float(args.focus_budget),
                seeds=seeds,
            )
        )
    raw = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if raw.empty:
        return raw, pd.DataFrame()
    raw = raw.drop_duplicates(["variant", "seed", "budget"], keep="last").sort_values(["variant", "seed"])
    stats = summarize_variants(
        raw,
        variant_order=A1_VARIANTS,
        label_map=A1_LABELS,
        reference_variant="full",
        bonferroni_alpha=BONFERRONI_A1_ALPHA,
        n_bootstrap=int(args.n_bootstrap),
    )
    return raw.reset_index(drop=True), stats


def collect_a2(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = Path(args.run_root)
    seeds = [int(seed) for seed in args.a2_seeds]
    frames = []
    for stage in A2_STAGES:
        frames.append(
            collect_variant_rows(
                run_root,
                experiment="A2_staged_v31_aligned",
                variant=stage,
                budget=float(args.focus_budget),
                seeds=seeds,
                extra={"stage": stage},
            )
        )
    raw = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if raw.empty:
        return raw, pd.DataFrame()
    raw = raw.drop_duplicates(["variant", "seed", "budget"], keep="last").sort_values(["variant", "seed"])
    stats = summarize_variants(
        raw,
        variant_order=A2_STAGES,
        label_map=A2_LABELS,
        reference_variant="D4_plus_oracle_prior_full",
        bonferroni_alpha=None,
        n_bootstrap=int(args.n_bootstrap),
    )
    return raw.reset_index(drop=True), stats


def parse_h1_variant(variant: str) -> tuple[float, float] | None:
    match = re.fullmatch(r"awbc([0-9]+p[0-9]+)_kl([0-9]+p[0-9]+)", str(variant))
    if not match:
        return None
    return float(match.group(1).replace("p", ".")), float(match.group(2).replace("p", "."))


def collect_h1(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = Path(args.run_root)
    s2_main_root = Path(args.s2_main_root)
    seeds = [int(seed) for seed in args.h1_seeds]
    frames = []
    baseline = load_a2_full_reference(run_root, s2_main_root, budget=float(args.focus_budget), seeds=seeds)
    if not baseline.empty:
        baseline = baseline.assign(
            variant=f"awbc{value_tag(BASELINE_AWBC)}_kl{value_tag(BASELINE_KL)}",
            awbc=BASELINE_AWBC,
            prior_kl=BASELINE_KL,
        )
        frames.append(baseline)
    for awbc in (0.05, 0.1, 0.2):
        for kl in (0.5, 1.0, 2.0):
            if abs(awbc - BASELINE_AWBC) < 1e-12 and abs(kl - BASELINE_KL) < 1e-12:
                continue
            variant = f"awbc{value_tag(awbc)}_kl{value_tag(kl)}"
            frames.append(
                collect_variant_rows(
                    run_root,
                    experiment="H1_hyperparam_v31_aligned",
                    variant=variant,
                    budget=float(args.focus_budget),
                    seeds=seeds,
                    extra={"awbc": float(awbc), "prior_kl": float(kl)},
                )
            )
    raw = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if raw.empty:
        return raw, pd.DataFrame()
    raw = raw.drop_duplicates(["variant", "seed", "budget"], keep="last").sort_values(["awbc", "prior_kl", "seed"])
    rows = []
    for (awbc, kl), group in raw.groupby(["awbc", "prior_kl"], sort=True):
        values = group["fw_mae"].to_numpy(dtype=float)
        rows.append(
            {
                "awbc": float(awbc),
                "prior_kl": float(kl),
                "variant": f"awbc{value_tag(float(awbc))}_kl{value_tag(float(kl))}",
                "budget": float(group["budget"].iloc[0]),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
                "is_default": bool(abs(float(awbc) - BASELINE_AWBC) < 1e-12 and abs(float(kl) - BASELINE_KL) < 1e-12),
            }
        )
    stats = pd.DataFrame(rows).sort_values(["awbc", "prior_kl"]).reset_index(drop=True)
    default_rows = stats[stats["is_default"]]
    if not default_rows.empty:
        default_mean = float(default_rows.iloc[0]["mean"])
        stats["delta_vs_default"] = stats["mean"] - default_mean
        stats["delta_pct_vs_default"] = (stats["mean"] - default_mean) / default_mean * 100.0
    return raw.reset_index(drop=True), stats


def completion_check(args: argparse.Namespace, a1_raw: pd.DataFrame, a2_raw: pd.DataFrame, h1_raw: pd.DataFrame) -> pd.DataFrame:
    expected = [
        {
            "experiment": "A1",
            "expected_runs": (len(A1_VARIANTS) - 1 + 1) * len(args.a1_seeds),
            "completed_runs": int(len(a1_raw)),
            "note": "Includes full reference from A2 D4.",
        },
        {
            "experiment": "A2",
            "expected_runs": len(A2_STAGES) * len(args.a2_seeds),
            "completed_runs": int(len(a2_raw)),
            "note": "Four staged configurations, including full D4.",
        },
        {
            "experiment": "H1",
            "expected_runs": 9 * len(args.h1_seeds),
            "completed_runs": int(len(h1_raw)),
            "note": "Includes default cell from A2 D4.",
        },
    ]
    return pd.DataFrame(expected)


def save_bar(stats: pd.DataFrame, path: Path, *, x_col: str = "label", y_col: str = "mean") -> None:
    if stats.empty or plt is None:
        return
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    labels = stats[x_col].astype(str).tolist()
    values = stats[y_col].to_numpy(dtype=float)
    errors = stats["std"].to_numpy(dtype=float) if "std" in stats else None
    ax.bar(np.arange(len(values)), values, yerr=errors, color="#4c78a8", alpha=0.88, capsize=3)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("FW-MAE (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_h1_heatmap(stats: pd.DataFrame, path: Path) -> None:
    if stats.empty or plt is None:
        return
    pivot = stats.pivot(index="awbc", columns="prior_kl", values="mean")
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    ax.set_xlabel("oracle prior KL coefficient")
    ax.set_ylabel("AWBC coefficient")
    for row_idx, awbc in enumerate(pivot.index):
        for col_idx, kl in enumerate(pivot.columns):
            value = pivot.loc[awbc, kl]
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="FW-MAE")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V3.1-aligned A1/A2/H1 ablation outputs.")
    parser.add_argument("--run-root", default="reports/v31_ablation_aligned")
    parser.add_argument("--s2-main-root", default="reports/v31_s2_main")
    parser.add_argument("--focus-budget", type=float, default=1.70)
    parser.add_argument("--a1-seeds", nargs="+", type=int, default=list(range(41, 51)))
    parser.add_argument("--a2-seeds", nargs="+", type=int, default=list(range(41, 51)))
    parser.add_argument("--h1-seeds", nargs="+", type=int, default=list(range(41, 46)))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    out_dir = Path(parsed.run_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    a1_raw, a1_stats = collect_a1(parsed)
    a2_raw, a2_stats = collect_a2(parsed)
    h1_raw, h1_stats = collect_h1(parsed)
    check = completion_check(parsed, a1_raw, a2_raw, h1_raw)

    a1_raw.to_csv(out_dir / "v31_aligned_a1_raw.csv", index=False)
    a1_stats.to_csv(out_dir / "v31_aligned_a1_stats.csv", index=False)
    a2_raw.to_csv(out_dir / "v31_aligned_a2_raw.csv", index=False)
    a2_stats.to_csv(out_dir / "v31_aligned_a2_stats.csv", index=False)
    h1_raw.to_csv(out_dir / "v31_aligned_h1_raw.csv", index=False)
    h1_stats.to_csv(out_dir / "v31_aligned_h1_stats.csv", index=False)
    check.to_csv(out_dir / "v31_aligned_completion_check.csv", index=False)

    save_bar(a1_stats, out_dir / "figures" / "v31_aligned_a1_ablation.png")
    save_bar(a2_stats, out_dir / "figures" / "v31_aligned_a2_staged.png")
    save_h1_heatmap(h1_stats, out_dir / "figures" / "v31_aligned_h1_heatmap.png")

    print("[v31-ablation-collect] completion")
    print(check.to_string(index=False))
