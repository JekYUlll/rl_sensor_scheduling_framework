from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(rewards: list[float], losses: list[float], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(rewards)
    axes[0].set_title("Reward by episode")
    axes[1].plot(losses)
    axes[1].set_title("DQN loss")
    axes[1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
