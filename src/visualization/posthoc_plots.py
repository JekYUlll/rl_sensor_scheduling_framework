from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


def heatmap(
    df,
    out_path: str | Path,
    title: str = "Heatmap",
    vmin: float | None = -1.0,
    vmax: float | None = 1.0,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(df, ax=ax, cmap="coolwarm", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(df.values, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
