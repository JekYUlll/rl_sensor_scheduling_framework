#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

UNIT_BY_VARIABLE = {
    "air_temperature_c": "degC",
    "wind_speed_ms": "m s^-1",
    "snow_mass_flux_kg_m2_s": "kg m^-2 s^-1",
}


DISPLAY_POLICY = {
    "custom_ppo": "PD-PPO",
    "feasible_static_projected": "Static projection",
    "validation_selected_static": "Validation-selected static",
    "full_open_unconstrained": "Full observation",
    "round_robin": "Round-robin",
    "aoi": "AoI",
    "random": "Random",
}

POLICY_ORDER = [
    ("full_open_unconstrained", "Full obs."),
    ("validation_selected_static", "Val.-selected static"),
    ("custom_ppo", "PD-PPO"),
    ("round_robin", "Round-robin"),
    ("aoi", "AoI"),
    ("random", "Random"),
]

VARIABLE_LABELS = {
    "air_temperature_c": ("Air temperature", r"$^{\circ}$C"),
    "wind_speed_ms": ("Wind speed", r"m\,s$^{-1}$"),
    "snow_mass_flux_kg_m2_s": ("Snow mass flux", r"kg\,m$^{-2}$\,s$^{-1}$"),
}


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def framework_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def collect_by_variable(grid_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for grid_dir in grid_dirs:
        for path in sorted(grid_dir.rglob("budget*_seed*/evaluation/v2_eval_by_variable.csv")):
            budget, seed = parse_budget_seed(path.parents[1])
            table = pd.read_csv(path)
            table["budget"] = float(budget)
            table["seed"] = int(seed)
            table["run_dir"] = str(path.parents[1])
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise(table: pd.DataFrame, *, variables: list[str], policies: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    metric_col = "forecast_mae" if "forecast_mae" in table.columns else "mae"
    filtered = table[table["variable"].isin(variables) & table["policy"].isin(policies)].copy()
    filtered = filtered[pd.notna(filtered[metric_col])]
    if filtered.empty:
        return filtered
    summary = (
        filtered.groupby(["budget", "policy", "variable"], dropna=False)[metric_col]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "mae_mean", "std": "mae_std", "min": "mae_min", "max": "mae_max"})
    )
    summary["unit"] = summary["variable"].map(UNIT_BY_VARIABLE).fillna("")
    summary["policy_label"] = summary["policy"].map(DISPLAY_POLICY).fillna(summary["policy"])
    summary = summary[
        [
            "budget",
            "policy",
            "policy_label",
            "variable",
            "unit",
            "mae_mean",
            "mae_std",
            "mae_min",
            "mae_max",
            "n",
        ]
    ].sort_values(["budget", "variable", "mae_mean", "policy"])
    return summary


def latex_value(variable: str, value: float) -> str:
    if variable == "snow_mass_flux_kg_m2_s":
        coefficient, exponent = f"{float(value):.2e}".split("e")
        return rf"${coefficient}{{\times}}10^{{{int(exponent)}}}$"
    return f"{float(value):.3f}"


def write_latex_table(focus: pd.DataFrame, *, focus_budget: float, out_path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Final-test per-variable raw forecast MAE in physical units at $B={float(focus_budget):.2f}$",
        r"  under the chronological split protocol ($n=10$ seeds). Lower is better; the static",
        r"  comparator was selected using validation windows only.}",
        r"  \label{tab:p1_physical}",
        r"  \scriptsize",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{@{}llcccccc@{}}",
        r"    \toprule",
        "    Variable & Unit & " + " & ".join(label for _, label in POLICY_ORDER) + r" \\",
        r"    \midrule",
    ]
    for variable in VARIABLE_LABELS:
        label, unit = VARIABLE_LABELS[variable]
        row = focus[focus["variable"] == variable].set_index("policy")
        values = []
        for policy, _ in POLICY_ORDER:
            if policy not in row.index:
                raise ValueError(f"Missing physical-unit row for policy={policy!r}, variable={variable!r}")
            values.append(latex_value(variable, float(row.loc[policy, "mae_mean"])))
        lines.append(f"    {label}")
        lines.append(f"      & {unit}")
        lines.append("      & " + " & ".join(values) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build physical-unit per-variable MAE tables for v2 paper claims.")
    parser.add_argument("--grid-dirs", nargs="+", default=["reports/v31_split_protocol_main/raw"])
    parser.add_argument("--out-dir", default="reports/v31_split_protocol_main/physical_unit_assets")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["air_temperature_c", "wind_speed_ms", "snow_mass_flux_kg_m2_s"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "full_open_unconstrained",
            "validation_selected_static",
            "custom_ppo",
            "round_robin",
            "aoi",
            "random",
        ],
    )
    parser.add_argument("--focus-budget", type=float, default=1.70)
    parser.add_argument("--latex-out", default="paper/tables/physical_unit_mae.tex")
    args = parser.parse_args()

    out_dir = framework_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = collect_by_variable([framework_path(str(path)) for path in args.grid_dirs])
    summary = summarise(table, variables=[str(x) for x in args.variables], policies=[str(x) for x in args.policies])
    all_path = out_dir / "exp_p1_physical_unit_mae.csv"
    summary.to_csv(all_path, index=False)
    focus = summary[summary["budget"].round(6) == round(float(args.focus_budget), 6)].copy()
    focus_path = out_dir / f"exp_p1_physical_unit_mae_budget{budget_tag(float(args.focus_budget))}.csv"
    focus.to_csv(focus_path, index=False)
    if not focus.empty:
        write_latex_table(focus, focus_budget=float(args.focus_budget), out_path=framework_path(str(args.latex_out)))
    print(all_path)
    if not focus.empty:
        print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
