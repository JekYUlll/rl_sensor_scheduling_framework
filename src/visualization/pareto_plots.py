from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_pareto(power_vals: list[float], score_vals: list[float], labels: list[str], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(power_vals, score_vals)
    for x, y, lab in zip(power_vals, score_vals, labels):
        ax.annotate(lab, (x, y), fontsize=8)
    ax.set_xlabel("power")
    ax.set_ylabel("quality")
    ax.set_title("Pareto view")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
