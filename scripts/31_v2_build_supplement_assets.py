#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BUDGETS = (1.65, 1.70, 1.75)
DEFAULT_SEEDS = tuple(range(41, 51))
BONFERRONI_ALPHA = 0.05 / 6.0

POLICY_LABELS = {
    "full_open_unconstrained": "Full observation",
    "feasible_static_projected": "Static projection",
    "custom_ppo": "PD-PPO",
    "dqn": "DQN",
    "cmdp_dqn": "CMDP-DQN",
    "round_robin": "Round-robin",
    "aoi": "AoI",
    "random": "Random",
}

POLICY_ORDER = (
    "full_open_unconstrained",
    "feasible_static_projected",
    "custom_ppo",
    "dqn",
    "cmdp_dqn",
    "round_robin",
    "aoi",
    "random",
)

ABLATION_ORDER = (
    "full_pd_ppo",
    "minus_oracle_prior",
    "minus_masked_actor",
    "minus_action_embedding",
    "minus_event_aware_critic",
    "minus_awbc",
)

ABLATION_LABELS = {
    "full_pd_ppo": "Full PD-PPO",
    "minus_oracle_prior": "- prior",
    "minus_masked_actor": "- mask",
    "minus_action_embedding": "- action emb.",
    "minus_event_aware_critic": "- event critic",
    "minus_awbc": "- AWBC",
}

D_STAGE_LABELS = {
    "D1_masked_actor": "D1\nMask",
    "D2_event_critic": "D2\n+Event",
    "D3_event_critic_awbc": "D3\n+AWBC",
    "D4_oracle_prior": "D4\n+Prior",
}


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def policy_label(policy: str) -> str:
    return POLICY_LABELS.get(str(policy), str(policy))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_grid_overall(
    grid_dirs: list[Path],
    *,
    budgets: list[float],
    seeds: list[int],
    filename: str = "v2_eval_overall.csv",
) -> pd.DataFrame:
    frames = []
    for grid_dir, budget, seed in product(grid_dirs, budgets, seeds):
        path = grid_dir / f"budget{budget_tag(budget)}_seed{int(seed)}" / "evaluation" / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(seed))
        frame.insert(0, "budget", float(budget))
        frame.insert(0, "source_dir", str(grid_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    key_cols = [col for col in ("budget", "seed", "policy") if col in df.columns]
    if key_cols:
        df = df.drop_duplicates(key_cols, keep="last")
    return df


def load_main_overall(args: argparse.Namespace) -> pd.DataFrame:
    table_dir = Path(args.table_dir)
    table_path = table_dir / "overall_long.csv"
    grid_df = load_grid_overall([Path(path) for path in args.grid_dirs], budgets=args.budgets, seeds=args.seeds)
    if table_path.exists():
        df = pd.read_csv(table_path)
        if args.seeds:
            df = df[df["seed"].isin([int(x) for x in args.seeds])]
        if args.budgets:
            df = df[df["budget"].isin([float(x) for x in args.budgets])]
        key_cols = [col for col in ("budget", "seed", "policy") if col in df.columns]
        table_keys = df[key_cols].drop_duplicates().shape[0] if key_cols else len(df)
        grid_keys = grid_df[key_cols].drop_duplicates().shape[0] if (not grid_df.empty and key_cols) else len(grid_df)
        if not grid_df.empty and grid_keys > table_keys:
            print(
                f"[warn] {table_path} contains fewer filtered rows ({table_keys}) than raw grid ({grid_keys}); "
                "using raw grid results instead.",
                flush=True,
            )
            return grid_df
        return df
    return grid_df


def metric_col(df: pd.DataFrame) -> str:
    for candidate in ("forecast_weighted_mae_overall", "weighted_normalized_mae", "obs_reconstruction_mae", "mae"):
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


def paired_wilcoxon(reference: pd.Series, baseline: pd.Series) -> tuple[float, str]:
    paired = pd.concat([reference.rename("ref"), baseline.rename("base")], axis=1).dropna()
    if len(paired) < 2:
        return float("nan"), "insufficient_n"
    try:
        from scipy.stats import wilcoxon

        _, p_value = wilcoxon(paired["ref"], paired["base"], alternative="two-sided", zero_method="wilcox")
        return float(p_value), "scipy_wilcoxon"
    except Exception:
        diffs = paired["ref"].to_numpy(dtype=float) - paired["base"].to_numpy(dtype=float)
        nonzero = diffs[np.abs(diffs) > 1e-12]
        if nonzero.size == 0:
            return 1.0, "all_zero_diff"
        # Conservative sign-test fallback. The CSV records the method so it is
        # not mistaken for the requested Wilcoxon test when SciPy is absent.
        k = int(min(np.sum(nonzero > 0), np.sum(nonzero < 0)))
        n = int(nonzero.size)
        p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
        return float(min(1.0, p)), "fallback_sign_test"


def build_s1_stats(overall: pd.DataFrame, out_dir: Path, *, n_bootstrap: int) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    score_col = metric_col(overall)
    rows = []
    for budget in sorted(float(x) for x in overall["budget"].dropna().unique()):
        budget_df = overall[overall["budget"] == budget]
        ref = budget_df[budget_df["policy"] == "custom_ppo"].set_index("seed")[score_col]
        for policy, group in budget_df.groupby("policy"):
            values = group[score_col].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=int(n_bootstrap), seed=int(round(budget * 1000)) + len(policy))
            if policy == "custom_ppo":
                p_value = float("nan")
                test_method = "reference"
            else:
                p_value, test_method = paired_wilcoxon(ref, group.set_index("seed")[score_col])
            rows.append(
                {
                    "method": str(policy),
                    "method_label": policy_label(str(policy)),
                    "budget": float(budget),
                    "mean_fwmae": float(np.mean(values)),
                    "std_fwmae": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "n": int(values.size),
                    "p_vs_pdppo": float(p_value),
                    "significant": bool(np.isfinite(p_value) and p_value < BONFERRONI_ALPHA),
                    "test_method": test_method,
                    "bonferroni_alpha": BONFERRONI_ALPHA,
                }
            )
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats.to_csv(out_dir / "exp_s1_main_stats.csv", index=False)
    return stats


def plot_f1(stats: pd.DataFrame, out_dir: Path) -> None:
    if stats.empty:
        return
    policies = [policy for policy in POLICY_ORDER if policy in set(stats["method"])]
    budgets = sorted(float(x) for x in stats["budget"].dropna().unique())
    x = np.arange(len(budgets), dtype=float)
    width = min(0.11, 0.75 / max(len(policies), 1))
    colors = {
        "custom_ppo": "#1f4e8c",
        "round_robin": "#f28e2b",
        "aoi": "#59a14f",
        "random": "#8c8c8c",
        "full_open_unconstrained": "#4e79a7",
        "feasible_static_projected": "#76b7b2",
        "dqn": "#b07aa1",
        "cmdp_dqn": "#9c755f",
    }
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    for idx, policy in enumerate(policies):
        sub = stats[stats["method"] == policy].set_index("budget")
        means = np.asarray([sub.loc[budget, "mean_fwmae"] if budget in sub.index else np.nan for budget in budgets], dtype=float)
        lows = np.asarray([sub.loc[budget, "ci_lower"] if budget in sub.index else np.nan for budget in budgets], dtype=float)
        highs = np.asarray([sub.loc[budget, "ci_upper"] if budget in sub.index else np.nan for budget in budgets], dtype=float)
        offset = (idx - (len(policies) - 1) / 2.0) * width
        yerr = np.vstack([means - lows, highs - means])
        ax.bar(x + offset, means, width=width, label=policy_label(policy), color=colors.get(policy, "#cccccc"))
        ax.errorbar(x + offset, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{budget:.2f}" for budget in budgets])
    ax.set_xlabel("Power budget B")
    ax.set_ylabel("FW-MAE (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_s1_main_results_ci.png", dpi=300)
    fig.savefig(out_dir / "figure_s1_main_results_ci.pdf")
    plt.close(fig)


def scan_experiment_root(root: Path, *, score_policy: str = "custom_ppo") -> pd.DataFrame:
    frames = []
    if not root.exists():
        return pd.DataFrame()
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in variant_dir.glob("budget*_seed*") if path.is_dir()):
            path = run_dir / "evaluation" / "v2_eval_overall.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "policy" in df.columns:
                df = df[df["policy"] == score_policy]
            if df.empty:
                continue
            budget = _parse_budget(run_dir.name)
            seed = _parse_seed(run_dir.name)
            df.insert(0, "seed", seed)
            df.insert(0, "budget", budget)
            df.insert(0, "variant", variant_dir.name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_variants(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    score_col = metric_col(df)
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[score_col].dropna().to_numpy(dtype=float)
        row = {col: key for col, key in zip(group_cols, keys, strict=True)}
        row.update(
            {
                "mean_fwmae": float(np.mean(values)) if values.size else float("nan"),
                "std_fwmae": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "n": int(values.size),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_assets(supp_root: Path, out_dir: Path) -> pd.DataFrame:
    df = scan_experiment_root(supp_root / "A1_ablation")
    stats = summarize_variants(df, group_cols=["variant", "budget"])
    if stats.empty:
        return stats
    full = stats[stats["variant"] == "full_pd_ppo"].set_index("budget")["mean_fwmae"]
    stats["delta_vs_full"] = [
        float(row["mean_fwmae"] - full.get(float(row["budget"]), np.nan))
        for _, row in stats.iterrows()
    ]
    stats.to_csv(out_dir / "exp_a1_ablation_stats.csv", index=False)
    plot_f2_ablation(stats, out_dir)
    return stats


def plot_f2_ablation(stats: pd.DataFrame, out_dir: Path) -> None:
    variants = [name for name in ABLATION_ORDER if name in set(stats["variant"])]
    variants.extend(name for name in stats["variant"].unique() if name not in variants)
    budgets = sorted(float(x) for x in stats["budget"].dropna().unique())
    pivot = stats.pivot_table(index="variant", columns="budget", values="mean_fwmae", aggfunc="first").reindex(variants)
    stds = stats.pivot_table(index="variant", columns="budget", values="std_fwmae", aggfunc="first").reindex(variants)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(4.9, 3.2))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels([f"{budget:.2f}" for budget in budgets])
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([ABLATION_LABELS.get(v, v) for v in variants])
    for i, variant in enumerate(variants):
        for j, budget in enumerate(budgets):
            mean = pivot.loc[variant, budget] if budget in pivot.columns else np.nan
            std = stds.loc[variant, budget] if budget in stds.columns else np.nan
            if np.isfinite(mean):
                ax.text(j, i, f"{mean:.3f}\n±{std:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Power budget B")
    ax.set_title("Ablation robustness across budgets")
    fig.colorbar(im, ax=ax, label="FW-MAE")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_a1_ablation_heatmap.png", dpi=300)
    fig.savefig(out_dir / "figure_a1_ablation_heatmap.pdf")
    plt.close(fig)


def build_hyperparam_assets(supp_root: Path, out_dir: Path) -> pd.DataFrame:
    frames = []
    for root in sorted(supp_root.glob("H1_*")):
        df = scan_experiment_root(root)
        if df.empty:
            continue
        df.insert(0, "hyperparam_name", root.name.removeprefix("H1_"))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    stats = summarize_variants(raw, group_cols=["hyperparam_name", "variant"])
    stats["hyperparam_value"] = [
        str(row["variant"]).removeprefix(f"{row['hyperparam_name']}_")
        for _, row in stats.iterrows()
    ]
    stats.to_csv(out_dir / "exp_h1_hyperparam_stats.csv", index=False)
    plot_f3_hyperparams(stats, out_dir)
    return stats


def _value_to_float(value: str) -> float:
    text = str(value).replace("p", ".").replace("x", "")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def plot_f3_hyperparams(stats: pd.DataFrame, out_dir: Path) -> None:
    params = ["lambda_awbc", "lambda_kl", "embed_dim", "lambda_warm"]
    titles = [r"$\lambda_{\mathrm{AWBC}}$", r"$\lambda_{\mathrm{KL}}$", r"$d_{\mathrm{emb}}$", r"$\lambda_{\mathrm{warm}}$ multiplier"]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8))
    for ax, param, title in zip(axes.reshape(-1), params, titles, strict=True):
        sub = stats[stats["hyperparam_name"] == param].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub["x"] = sub["hyperparam_value"].map(_value_to_float)
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["mean_fwmae"], marker="o", color="#1f4e8c")
        ax.fill_between(
            sub["x"].to_numpy(dtype=float),
            (sub["mean_fwmae"] - sub["std_fwmae"]).to_numpy(dtype=float),
            (sub["mean_fwmae"] + sub["std_fwmae"]).to_numpy(dtype=float),
            color="#1f4e8c",
            alpha=0.18,
        )
        if param in {"lambda_awbc", "lambda_kl", "lambda_warm"}:
            ax.set_xscale("log")
        default_x = {"lambda_awbc": 0.1, "lambda_kl": 1.0, "embed_dim": 32.0, "lambda_warm": 1.0}.get(param)
        if default_x:
            ax.axvline(default_x, ls="--", color="black", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("FW-MAE")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_h1_hyperparam_sensitivity.png", dpi=300)
    fig.savefig(out_dir / "figure_h1_hyperparam_sensitivity.pdf")
    plt.close(fig)


def build_condition_assets(
    supp_root: Path,
    out_dir: Path,
    *,
    e1_subdir: str,
    budgets: list[float],
    seeds: list[int],
) -> pd.DataFrame:
    frames = []
    root = supp_root / str(e1_subdir)
    if not root.exists():
        return pd.DataFrame()
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        df = load_grid_overall([condition_dir], budgets=budgets, seeds=seeds)
        if df.empty:
            continue
        df.insert(0, "episode_type", condition_dir.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    stats = summarize_variants(raw.rename(columns={"episode_type": "variant"}), group_cols=["variant", "policy"])
    stats = stats.rename(columns={"variant": "episode_type", "policy": "method"})
    suffix = "" if str(e1_subdir) == "E1_condition_eval" else f"_{str(e1_subdir).removeprefix('E1_condition_eval_')}"
    stats.to_csv(out_dir / f"exp_e1_condition_stats{suffix}.csv", index=False)
    plot_f4_conditions(stats, out_dir, suffix=suffix)
    return stats


def plot_f4_conditions(stats: pd.DataFrame, out_dir: Path, *, suffix: str = "") -> None:
    methods = [m for m in ("custom_ppo", "round_robin", "aoi", "random") if m in set(stats["method"])]
    episode_types = [e for e in ("calm", "mixed", "event") if e in set(stats["episode_type"])]
    if not methods or not episode_types:
        return
    x = np.arange(len(episode_types), dtype=float)
    width = 0.18
    colors = {"custom_ppo": "#1f4e8c", "round_robin": "#f28e2b", "aoi": "#59a14f", "random": "#8c8c8c"}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for idx, method in enumerate(methods):
        sub = stats[stats["method"] == method].set_index("episode_type")
        means = np.asarray([sub.loc[e, "mean_fwmae"] if e in sub.index else np.nan for e in episode_types], dtype=float)
        stds = np.asarray([sub.loc[e, "std_fwmae"] if e in sub.index else np.nan for e in episode_types], dtype=float)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, means, yerr=stds, width=width, capsize=2, color=colors.get(method, "#cccccc"), label=policy_label(method))
    ax.set_xticks(x)
    ax.set_xticklabels(["Calm", "Mixed", "Event"])
    ax.set_ylabel("FW-MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"figure_e1_condition_eval{suffix}.png", dpi=300)
    fig.savefig(out_dir / f"figure_e1_condition_eval{suffix}.pdf")
    plt.close(fig)


def build_diagnostic_assets(supp_root: Path, out_dir: Path) -> pd.DataFrame:
    df = scan_experiment_root(supp_root / "A2_diagnostic")
    stats = summarize_variants(df, group_cols=["variant"])
    if stats.empty:
        return stats
    stats.to_csv(out_dir / "exp_a2_diagnostic_stats.csv", index=False)
    plot_f5_diagnostic(stats, out_dir)
    return stats


def plot_f5_diagnostic(stats: pd.DataFrame, out_dir: Path) -> None:
    stages = [stage for stage in D_STAGE_LABELS if stage in set(stats["variant"])]
    if not stages:
        return
    sub = stats.set_index("variant").loc[stages]
    x = np.arange(len(stages), dtype=float)
    means = sub["mean_fwmae"].to_numpy(dtype=float)
    stds = sub["std_fwmae"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.errorbar(x, means, yerr=stds, marker="o", color="#1f4e8c", capsize=3)
    ax.step(x, means, where="mid", color="#1f4e8c", alpha=0.35)
    for idx in range(1, len(stages)):
        delta = means[idx] - means[idx - 1]
        ax.annotate(f"{delta:+.3f}", xy=(idx - 0.5, (means[idx] + means[idx - 1]) / 2), fontsize=8, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([D_STAGE_LABELS[s] for s in stages])
    ax.set_ylabel("FW-MAE")
    ax.set_title("DQN/PD-PPO diagnostic sequence")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_a2_diagnostic_sequence.png", dpi=300)
    fig.savefig(out_dir / "figure_a2_diagnostic_sequence.pdf")
    plt.close(fig)


def write_manifest(out_dir: Path, sections: dict[str, pd.DataFrame]) -> None:
    lines = ["# Supplement Asset Manifest", ""]
    for name, frame in sections.items():
        lines.append(f"- {name}: {'available' if not frame.empty else 'missing'} ({len(frame)} rows)")
    (out_dir / "supplement_asset_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_budget(text: str) -> float:
    for part in str(text).split("_"):
        if part.startswith("budget"):
            return float(part.replace("budget", "").replace("p", "."))
    return float("nan")


def _parse_seed(text: str) -> int:
    for part in str(text).split("_"):
        if part.startswith("seed"):
            return int(part.replace("seed", ""))
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supplementary statistics and figures for v2 PD-PPO experiments.")
    parser.add_argument("--grid-dirs", nargs="+", default=["reports/v2_forecast_eval_grid_prior_kl1"])
    parser.add_argument("--table-dir", default="reports/v2_paper_tables_prior_kl1")
    parser.add_argument("--supp-root", default="reports/v2_supplement_experiments")
    parser.add_argument("--out-dir", default="reports/v2_supplement_assets")
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--e1-subdir", default="E1_condition_eval")
    args = parser.parse_args()

    args.budgets = [float(x) for x in args.budgets]
    args.seeds = [int(x) for x in args.seeds]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    supp_root = Path(args.supp_root)

    main_overall = load_main_overall(args)
    s1_stats = build_s1_stats(main_overall, out_dir, n_bootstrap=int(args.bootstrap))
    plot_f1(s1_stats, out_dir)
    a1_stats = build_ablation_assets(supp_root, out_dir)
    h1_stats = build_hyperparam_assets(supp_root, out_dir)
    e1_stats = build_condition_assets(
        supp_root,
        out_dir,
        e1_subdir=str(args.e1_subdir),
        budgets=args.budgets,
        seeds=args.seeds,
    )
    a2_stats = build_diagnostic_assets(supp_root, out_dir)
    write_manifest(
        out_dir,
        {
            "S1 main statistics": s1_stats,
            "A1 ablation": a1_stats,
            "H1 hyperparameter sensitivity": h1_stats,
            "E1 condition evaluation": e1_stats,
            "A2 diagnostic sequence": a2_stats,
        },
    )
    print(out_dir)


if __name__ == "__main__":
    main()
