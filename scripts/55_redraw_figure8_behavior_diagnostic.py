#!/usr/bin/env python3
"""Redraw the fixed-budget split-protocol behavior diagnostic timeline figure.

The figure uses corrected final-test rollouts at B=1.70, seed 41. It is intentionally
a diagnostic figure: event context, PD-PPO sensor modes, and per-policy rolling oracle
loss are shown together so the reader does not mistake high switching frequency for
useful adaptive scheduling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "reports" / "v31_split_protocol_main" / "raw" / "budget1p70_seed41"
OUT_DIR = ROOT / "paper" / "figures"

POLICIES = [
    ("full_open_unconstrained", "Full obs.", "#595959"),
    ("validation_selected_static", "Val.-selected static", "#4daf4a"),
    ("custom_ppo", "PD-PPO", "#1f78b4"),
    ("round_robin", "Round-robin", "#ff7f00"),
    ("aoi", "AoI", "#984ea3"),
    ("random", "Random", "#e41a1c"),
]


def load_rollout(policy: str) -> dict[str, np.ndarray]:
    path = RUN_DIR / f"rollout_{policy}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def rolling_mean(x: np.ndarray, window: int = 48) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x.astype(float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def nice_sensor_label(sensor_id: str) -> str:
    return (
        sensor_id.replace("_", " ")
        .replace("met station core", "weather core")
        .replace("radiometer basic", "pyranometer")
        .replace("surface temp ir", "surface IR")
        .replace("ultrasonic anemometer hd", "hi-res wind")
        .replace("shielded thermo hygro", "thermo-hygro")
        .replace("snow particle counter", "particle counter")
        .replace("laser disdrometer", "laser disdrometer")
        .replace("fc4 flux", "FC4 flux")
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_steps = 512
    ppo = load_rollout("custom_ppo")
    steps = np.arange(max_steps)
    event = np.asarray(ppo["event_flags"], dtype=float)[:max_steps]
    modes = np.asarray(ppo["mode_ids"], dtype=int)[:max_steps].T
    sensor_ids = [str(x) for x in ppo["sensor_ids"].tolist()]
    sensor_labels = [nice_sensor_label(x) for x in sensor_ids]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.45, 2.7, 2.0],
        hspace=0.22,
        left=0.16,
        right=0.96,
        top=0.95,
        bottom=0.10,
    )

    ax_event = fig.add_subplot(gs[0])
    event_cmap = ListedColormap(["#fbfbfb", "#e66101"])
    ax_event.imshow(event[None, :], aspect="auto", interpolation="nearest", cmap=event_cmap, vmin=0, vmax=1)
    ax_event.set_yticks([])
    ax_event.set_xlim(0, max_steps - 1)
    ax_event.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_event.set_title("Blowing-snow event context")
    for spine in ax_event.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

    ax_modes = fig.add_subplot(gs[1], sharex=ax_event)
    mode_cmap = ListedColormap(["#f7f7f5", "#9ecae1", "#08519c"])
    mode_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], mode_cmap.N)
    ax_modes.imshow(modes, aspect="auto", interpolation="nearest", cmap=mode_cmap, norm=mode_norm)
    ax_modes.set_yticks(np.arange(len(sensor_labels)))
    ax_modes.set_yticklabels(sensor_labels)
    ax_modes.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_modes.set_title("PD-PPO sensor modes")
    ax_modes.set_yticks(np.arange(-0.5, len(sensor_labels), 1.0), minor=True)
    ax_modes.grid(which="minor", axis="y", color="white", linewidth=1.2)
    ax_modes.tick_params(axis="y", which="minor", length=0)
    for spine in ax_modes.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

    ax_loss = fig.add_subplot(gs[2], sharex=ax_event)
    means: list[tuple[str, float, str]] = []
    for policy, label, color in POLICIES:
        rollout = load_rollout(policy)
        losses = np.asarray(rollout["oracle_losses"], dtype=float)[:max_steps]
        y = rolling_mean(losses, window=48)
        ax_loss.plot(steps, y, label=label, color=color, linewidth=1.45)
        means.append((label, float(losses.mean()), color))

    ax_loss.set_xlim(0, max_steps - 1)
    ax_loss.set_xlabel("time index")
    ax_loss.set_ylabel("rolling oracle loss")
    ax_loss.set_xticks(np.arange(0, max_steps + 1, 128))
    ax_loss.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.8, columnspacing=1.0)

    means_sorted = sorted(means, key=lambda item: item[1])
    y0 = 0.96
    for idx, (label, value, color) in enumerate(means_sorted):
        ax_loss.text(
            0.985,
            y0 - idx * 0.115,
            f"{label}: {value:.3f}",
            color=color,
            ha="right",
            va="top",
            transform=ax_loss.transAxes,
            fontsize=8,
        )

    legend_handles = [
        Patch(facecolor="#f7f7f5", edgecolor="#999999", label="OFF"),
        Patch(facecolor="#9ecae1", edgecolor="#999999", label="WARMING"),
        Patch(facecolor="#08519c", edgecolor="#999999", label="ACTIVE"),
        Patch(facecolor="#e66101", edgecolor="#999999", label="event"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.005),
        ncol=4,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.4,
    )

    png = OUT_DIR / "figure5_sensor_timeline.png"
    svg = OUT_DIR / "figure5_sensor_timeline.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
