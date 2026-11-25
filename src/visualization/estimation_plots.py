from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_trace_power(trace_hist: list[float], power_hist: list[float], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(trace_hist, label="trace(P)")
    ax2 = ax1.twinx()
    ax2.plot(power_hist, color="tab:orange", label="power", alpha=0.7)
    ax1.set_title("Estimation uncertainty and power")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
