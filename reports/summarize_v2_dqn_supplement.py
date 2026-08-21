from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    variants = {
        "D1_warm2_500k_nstep3": Path("reports/v2_forecast_eval_grid_dqn_d1_500k_warm2"),
        "D2_full_500k_nstep8_reference": Path("reports/v2_forecast_eval_grid_dqn_d2_500k_nstep8_full"),
        "D4_full_500k_nstep8_prefill3k_lh8": Path(
            "reports/v2_forecast_eval_grid_dqn_d4_500k_prefill3k_lh8_full_b170"
        ),
    }
    rows = []
    for variant, base in variants.items():
        for seed in (41, 42, 43):
            path = base / f"budget1p70_seed{seed}/evaluation/v2_eval_overall.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["variant"] = variant
            df["seed"] = seed
            df["path"] = str(path)
            rows.append(df)
    if not rows:
        raise SystemExit("no rows found")

    detail = pd.concat(rows, ignore_index=True)
    outdir = Path("reports/v2_dqn_supplement_20260506")
    outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(outdir / "budget1p70_detail.csv", index=False)

    summary = (
        detail.groupby(["variant", "policy"])["forecast_weighted_mae_overall"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["variant", "mean"])
    )
    summary.to_csv(outdir / "budget1p70_policy_summary.csv", index=False)

    keep = {
        "full_open_unconstrained",
        "feasible_static_projected",
        "round_robin",
        "dqn",
        "random",
        "aoi",
    }
    key = summary[summary["policy"].isin(keep)]
    lines = [
        "# V2 DQN supplementary diagnosis (2026-05-06)",
        "",
        "Metric: `forecast_weighted_mae_overall` at `budget=1.70` over seeds 41/42/43. Lower is better.",
        "",
        "D2 is kept as a diagnostic reference because a duplicate writer was detected during the first full-candidate run; D1 and D4 are clean for the acceptance decision.",
        "",
    ]
    for variant in variants:
        block = key[key["variant"] == variant].sort_values("mean")
        if block.empty:
            continue
        lines.append(f"## {variant}")
        for _, row in block.iterrows():
            lines.append(
                f"- {row['policy']}: mean={row['mean']:.6f}, "
                f"std={row['std']:.6f}, n={int(row['count'])}"
            )
        dqn_rows = block.loc[block["policy"] == "dqn", "mean"]
        rr_rows = block.loc[block["policy"] == "round_robin", "mean"]
        if not dqn_rows.empty and not rr_rows.empty:
            dqn = float(dqn_rows.iloc[0])
            rr = float(rr_rows.iloc[0])
            verdict = "passes" if dqn < rr else "does not pass"
            lines.append(
                f"- verdict: DQN {verdict} round_robin comparison "
                f"(dqn={dqn:.6f}, round_robin={rr:.6f})."
            )
        lines.append("")
    lines.append(
        "Conclusion: among clean variants, extended warm2 DQN is the closest but remains slightly above "
        "round_robin; oracle-prefill full-candidate DQN does not rescue the gap. This supports keeping "
        "PPO as the main algorithm and discussing discrete DQN as structurally limited in the "
        "warmup-aware CMDP."
    )
    (outdir / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        plot_policies = ["full_open_unconstrained", "feasible_static_projected", "round_robin", "dqn", "random"]
        plot_df = key[key["policy"].isin(plot_policies)].copy()
        pivot = plot_df.pivot(index="variant", columns="policy", values="mean").loc[list(variants)]
        ax = pivot[plot_policies].plot(kind="bar", figsize=(11, 4.8), width=0.82)
        ax.set_ylabel("forecast_weighted_mae_overall")
        ax.set_xlabel("")
        ax.set_title("DQN supplementary variants at budget=1.70")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "budget1p70_variant_policy_mae.png", dpi=180)
        plt.savefig(outdir / "budget1p70_variant_policy_mae.svg")
        plt.close()
    except Exception as exc:  # pragma: no cover - report generation should not block tables.
        (outdir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
    print((outdir / "budget1p70_policy_summary.csv").as_posix())
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
