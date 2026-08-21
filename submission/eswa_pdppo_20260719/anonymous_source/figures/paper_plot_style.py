"""Shared plotting style for the manuscript figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#0072B2",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#7A7A7A",
    "dark": "#2F2F2F",
    "light": "#E8EAED",
}


def apply_paper_style(base_size: float = 8.5) -> None:
    """Apply the common ESWA manuscript plotting style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": base_size,
            "axes.titlesize": base_size,
            "axes.titleweight": "bold",
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 1.0,
            "ytick.labelsize": base_size - 1.0,
            "legend.fontsize": base_size - 1.2,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "grid.linewidth": 0.45,
            "axes.axisbelow": True,
        }
    )


def save_pdf_png(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_prefix.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_prefix.with_suffix(".png"))


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
