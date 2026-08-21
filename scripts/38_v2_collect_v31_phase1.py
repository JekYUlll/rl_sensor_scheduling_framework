#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_A1_VARIANTS = (
    "full",
    "no_action_emb",
    "no_event_critic",
    "no_awbc",
    "no_oracle",
    "no_awbc_oracle",
    "no_action_mask",
    "masked_only",
)
DEFAULT_A1_LABELS = {
    "full": "Full PD-PPO",
    "no_action_emb": "No ActionEmbedding",
    "no_event_critic": "No EventAwareCritic",
    "no_awbc": "No AWBC",
    "no_oracle": "No oracle prior",
    "no_awbc_oracle": "No AWBC/prior",
    "no_action_mask": "No action mask",
    "masked_only": "MaskedActor only",
}
BONFERRONI_A1_ALPHA = 0.05 / 7.0
BASELINE_AWBC = 0.1
BASELINE_KL = 1.0


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


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
    raise ValueError(f"No supported FW-MAE column found in columns={list(df.columns)}")


def bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


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


def read_policy_score(path: Path, *, policy: str = "custom_ppo") -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "policy" in df.columns:
        df = df[df["policy"].astype(str) == str(policy)]
    if df.empty:
        return None
    col = metric_col(df)
    return float(df.iloc[0][col])


def read_run_score(run_dir: Path, *, policy: str = "custom_ppo") -> float | None:
    for rel_path in ("evaluation/v2_eval_overall.csv", "v2_custom_ppo_metrics.csv"):
        score = read_policy_score(run_dir / rel_path, policy=policy)
        if score is not None and np.isfinite(score):
            return score
    return None


def parse_awbc_kl(variant: str) -> tuple[float, float] | None:
    match = re.fullmatch(r"awbc([0-9]+p[0-9]+)_kl([0-9]+p[0-9]+)", str(variant))
    if not match:
        return None
    return float(match.group(1).replace("p", ".")), float(match.group(2).replace("p", "."))


def load_main_baseline(
    path: Path,
    grid_dir: Path,
    *,
    budget: float,
    seeds: list[int],
    policy: str = "custom_ppo",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    grid_rows = []
    for seed in seeds:
        run_dir = grid_dir / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
        score = read_run_score(run_dir, policy=policy)
        if score is None:
            continue
        grid_rows.append(
            {
                "variant": "full",
                "seed": int(seed),
                "budget": float(budget),
                "fw_mae": float(score),
                "source_path": str(run_dir),
            }
        )
    if grid_rows:
        frames.append(pd.DataFrame(grid_rows))
    if path.exists():
        df = pd.read_csv(path)
        if "budget" in df.columns:
            df = df[np.isclose(df["budget"].astype(float), float(budget))]
        if "seed" in df.columns and seeds:
            df = df[df["seed"].astype(int).isin([int(seed) for seed in seeds])]
        if "policy" in df.columns:
            df = df[df["policy"].astype(str) == str(policy)]
        if not df.empty:
            col = metric_col(df)
            frames.append(
                pd.DataFrame(
                    {
                        "variant": "full",
                        "seed": df["seed"].astype(int).to_numpy(),
                        "budget": float(budget),
                        "fw_mae": df[col].astype(float).to_numpy(),
                        "source_path": str(path),
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=["variant", "seed", "budget", "fw_mae", "source_path"])
    # Prefer raw run directories over the compact overall_long table because
    # the latter may be an older 3-seed snapshot while the raw grid has n=10.
    out = pd.concat(frames, ignore_index=True)
    out["source_rank"] = np.where(out["source_path"].astype(str).str.contains("/budget"), 1, 0)
    out = out.sort_values(["seed", "source_rank"]).drop_duplicates(["seed"], keep="last")
    return out.drop(columns=["source_rank"]).sort_values("seed").reset_index(drop=True)


def collect_a1_raw(args: argparse.Namespace) -> pd.DataFrame:
    root = Path(args.run_root) / "A1_ablation_v31"
    frames = [
        load_main_baseline(
            Path(args.main_overall),
            Path(args.main_grid_dir),
            budget=float(args.focus_budget),
            seeds=[int(seed) for seed in args.a1_seeds],
        )
    ]
    for variant in DEFAULT_A1_VARIANTS:
        if variant == "full":
            continue
        rows = []
        for seed in args.a1_seeds:
            run_dir = root / variant / f"budget{budget_tag(float(args.focus_budget))}_seed{int(seed)}"
            score = read_run_score(run_dir)
            if score is None:
                continue
            rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "budget": float(args.focus_budget),
                    "fw_mae": float(score),
                    "source_path": str(run_dir),
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))
    raw = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw
    raw = raw.drop_duplicates(["variant", "seed", "budget"], keep="last")
    raw = raw.sort_values(["variant", "seed"]).reset_index(drop=True)
    return raw


def summarize_a1(raw: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    full = raw[raw["variant"] == "full"].set_index("seed")["fw_mae"]
    full_mean = float(full.mean()) if len(full) else float("nan")
    rows = []
    for variant in DEFAULT_A1_VARIANTS:
        group = raw[raw["variant"] == variant]
        if group.empty:
            continue
        values = group["fw_mae"].to_numpy(dtype=float)
        ci_lower, ci_upper = bootstrap_ci(
            values,
            n_bootstrap=int(n_bootstrap),
            seed=17_000 + DEFAULT_A1_VARIANTS.index(variant),
        )
        if variant == "full":
            p_value = float("nan")
            test_method = "reference"
        else:
            p_value, test_method = paired_wilcoxon(full, group.set_index("seed")["fw_mae"])
        mean = float(np.mean(values))
        rows.append(
            {
                "variant": variant,
                "label": DEFAULT_A1_LABELS.get(variant, variant),
                "budget": float(group["budget"].iloc[0]),
                "mean": mean,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "n": int(len(values)),
                "delta_vs_full": float(mean - full_mean) if np.isfinite(full_mean) else float("nan"),
                "delta_pct_vs_full": float((mean - full_mean) / full_mean * 100.0) if full_mean else float("nan"),
                "p_vs_full": float(p_value),
                "significant": bool(np.isfinite(p_value) and p_value < BONFERRONI_A1_ALPHA),
                "test_method": test_method,
                "bonferroni_alpha": BONFERRONI_A1_ALPHA,
            }
        )
    return pd.DataFrame(rows)


def collect_h1_raw(args: argparse.Namespace) -> pd.DataFrame:
    root = Path(args.run_root) / "H1_hyperparam_v31"
    frames = []
    baseline = load_main_baseline(
        Path(args.main_overall),
        Path(args.main_grid_dir),
        budget=float(args.focus_budget),
        seeds=[int(seed) for seed in args.h1_seeds],
    )
    if not baseline.empty:
        baseline = baseline.assign(
            variant=f"awbc{str(BASELINE_AWBC).replace('.', 'p')}_kl{str(BASELINE_KL).replace('.', 'p')}",
            awbc_coef=BASELINE_AWBC,
            prior_kl_coef=BASELINE_KL,
            is_baseline=True,
        )
        frames.append(baseline)
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []:
        parsed = parse_awbc_kl(variant_dir.name)
        if parsed is None:
            continue
        awbc, kl = parsed
        rows = []
        for seed in args.h1_seeds:
            run_dir = variant_dir / f"budget{budget_tag(float(args.focus_budget))}_seed{int(seed)}"
            score = read_run_score(run_dir)
            if score is None:
                continue
            rows.append(
                {
                    "variant": variant_dir.name,
                    "seed": int(seed),
                    "budget": float(args.focus_budget),
                    "fw_mae": float(score),
                    "source_path": str(run_dir),
                    "awbc_coef": float(awbc),
                    "prior_kl_coef": float(kl),
                    "is_baseline": bool(abs(awbc - BASELINE_AWBC) < 1e-12 and abs(kl - BASELINE_KL) < 1e-12),
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw
    raw = raw.drop_duplicates(["awbc_coef", "prior_kl_coef", "seed"], keep="last")
    raw = raw.sort_values(["awbc_coef", "prior_kl_coef", "seed"]).reset_index(drop=True)
    return raw


def summarize_h1(raw: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    baseline = raw[
        np.isclose(raw["awbc_coef"].astype(float), BASELINE_AWBC)
        & np.isclose(raw["prior_kl_coef"].astype(float), BASELINE_KL)
    ]["fw_mae"]
    baseline_mean = float(baseline.mean()) if len(baseline) else float("nan")
    rows = []
    for (awbc, kl), group in raw.groupby(["awbc_coef", "prior_kl_coef"]):
        values = group["fw_mae"].to_numpy(dtype=float)
        ci_lower, ci_upper = bootstrap_ci(
            values,
            n_bootstrap=int(n_bootstrap),
            seed=23_000 + int(round(float(awbc) * 1000)) + int(round(float(kl) * 100)),
        )
        mean = float(np.mean(values))
        rows.append(
            {
                "awbc_coef": float(awbc),
                "prior_kl_coef": float(kl),
                "mean": mean,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "n": int(len(values)),
                "delta_pct": float((mean - baseline_mean) / baseline_mean * 100.0) if baseline_mean else float("nan"),
                "is_baseline": bool(abs(float(awbc) - BASELINE_AWBC) < 1e-12 and abs(float(kl) - BASELINE_KL) < 1e-12),
                "within_5pct": bool(np.isfinite(baseline_mean) and abs((mean - baseline_mean) / baseline_mean) <= 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values(["awbc_coef", "prior_kl_coef"]).reset_index(drop=True)


def plot_a1(stats: pd.DataFrame, out_dir: Path) -> None:
    if stats.empty:
        return
    ordered = stats.set_index("variant").reindex([v for v in DEFAULT_A1_VARIANTS if v in set(stats["variant"])])
    x = np.arange(len(ordered), dtype=float)
    colors = ["#1f4e8c" if idx == "full" else "#d98b35" for idx in ordered.index]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(x, ordered["mean"], yerr=ordered["std"], color=colors, capsize=3)
    ax.axhline(float(ordered.loc["full", "mean"]), color="#1f4e8c", ls="--", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["label"], rotation=30, ha="right")
    ax.set_ylabel("FW-MAE (lower is better)")
    ax.set_title("A1 full component ablation at B=1.70")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_a1_ablation_v31.png", dpi=300)
    fig.savefig(out_dir / "figure_a1_ablation_v31.pdf")
    plt.close(fig)


def plot_h1(stats: pd.DataFrame, out_dir: Path) -> None:
    if stats.empty:
        return
    awbc_values = sorted(float(x) for x in stats["awbc_coef"].dropna().unique())
    kl_values = sorted(float(x) for x in stats["prior_kl_coef"].dropna().unique())
    pivot = stats.pivot_table(index="awbc_coef", columns="prior_kl_coef", values="mean", aggfunc="first").reindex(
        index=awbc_values, columns=kl_values
    )
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(kl_values)))
    ax.set_xticklabels([f"{value:g}" for value in kl_values])
    ax.set_yticks(np.arange(len(awbc_values)))
    ax.set_yticklabels([f"{value:g}" for value in awbc_values])
    ax.set_xlabel("prior_kl_coef")
    ax.set_ylabel("awbc_coef")
    ax.set_title("H1 hyperparameter sensitivity")
    for i, awbc in enumerate(awbc_values):
        for j, kl in enumerate(kl_values):
            value = pivot.loc[awbc, kl] if kl in pivot.columns else np.nan
            if np.isfinite(value):
                marker = "*" if abs(awbc - BASELINE_AWBC) < 1e-12 and abs(kl - BASELINE_KL) < 1e-12 else ""
                ax.text(j, i, f"{value:.3f}{marker}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="FW-MAE")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_h1_heatmap.png", dpi=300)
    fig.savefig(out_dir / "figure_h1_heatmap.pdf")
    plt.close(fig)


def write_manifest(out_dir: Path, a1_raw: pd.DataFrame, a1_stats: pd.DataFrame, h1_raw: pd.DataFrame, h1_stats: pd.DataFrame) -> None:
    lines = [
        "# V3.1 Phase-1 Collection Manifest",
        "",
        f"- A1 raw rows: {len(a1_raw)}",
        f"- A1 completed variants: {', '.join(a1_stats['variant'].astype(str).tolist()) if not a1_stats.empty else 'none'}",
        f"- H1 raw rows: {len(h1_raw)}",
        f"- H1 completed cells: {len(h1_stats)}",
        "",
        "Acceptance notes:",
        "- A1 should contain full plus seven ablation variants with n=10 each before paper integration.",
        "- H1 should contain baseline plus eight grid cells with n=5 each before robustness claims.",
    ]
    (out_dir / "v31_phase1_collection_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V3.1 Phase-1 A1/H1 supplement results.")
    parser.add_argument("--run-root", default="reports/v3_supplement_assets")
    parser.add_argument("--out-dir", default="reports/v3_supplement_assets")
    parser.add_argument("--main-overall", default="reports/v2_paper_tables_prior_kl1/overall_long.csv")
    parser.add_argument("--main-grid-dir", default="reports/v2_forecast_eval_grid_prior_kl1")
    parser.add_argument("--focus-budget", type=float, default=1.70)
    parser.add_argument("--a1-seeds", nargs="+", type=int, default=list(range(41, 51)))
    parser.add_argument("--h1-seeds", nargs="+", type=int, default=list(range(41, 46)))
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a1_raw = collect_a1_raw(args)
    a1_stats = summarize_a1(a1_raw, n_bootstrap=int(args.bootstrap))
    h1_raw = collect_h1_raw(args)
    h1_stats = summarize_h1(h1_raw, n_bootstrap=int(args.bootstrap))

    a1_raw.to_csv(out_dir / "exp_a1_ablation_raw.csv", index=False)
    a1_stats.to_csv(out_dir / "exp_a1_ablation_stats.csv", index=False)
    h1_raw.to_csv(out_dir / "exp_h1_hyperparam_raw.csv", index=False)
    h1_stats.to_csv(out_dir / "exp_h1_hyperparam_stats.csv", index=False)
    plot_a1(a1_stats, out_dir)
    plot_h1(h1_stats, out_dir)
    write_manifest(out_dir, a1_raw, a1_stats, h1_raw, h1_stats)
    print(out_dir)
    if not a1_stats.empty:
        print(a1_stats.to_string(index=False))
    if not h1_stats.empty:
        print(h1_stats.to_string(index=False))


if __name__ == "__main__":
    main()
