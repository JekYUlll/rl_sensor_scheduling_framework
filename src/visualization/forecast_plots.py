from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, title: str = "Forecast") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(y_true.reshape(-1), label="true")
    ax.plot(y_pred.reshape(-1), label="pred", alpha=0.8)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
