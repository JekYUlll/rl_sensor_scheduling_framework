from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_action_hist(action_ids: list[int], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(action_ids, bins=min(30, max(5, len(set(action_ids)))))
    ax.set_title("Action frequency")
    ax.set_xlabel("action id")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
