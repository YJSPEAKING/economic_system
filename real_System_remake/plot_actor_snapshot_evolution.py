"""Plot deterministic Actor-snapshot evolution metrics without importing PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis_plots" / "actor_snapshot_evolution"
DEFAULT_SEEDS = (184, 652, 187)
COLORS = {184: "#2C7FB8", 652: "#F28E2B", 187: "#2CA02C"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )


def plot_per_seed(frames: dict[int, pd.DataFrame], output_path: Path) -> None:
    _style()
    figure, axes = plt.subplots(len(frames), 1, figsize=(9.2, 3.05 * len(frames)), sharex=True)
    if len(frames) == 1:
        axes = [axes]
    for axis, (seed, frame) in zip(axes, frames.items()):
        color = COLORS.get(seed, "#2C7FB8")
        axis.plot(
            frame["evaluation_step"],
            frame["actor_expert_js_divergence"],
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=2.7,
            alpha=0.8,
            label="Per-snapshot JS",
        )
        smooth = frame["actor_expert_js_divergence"].rolling(5, min_periods=1).mean()
        axis.plot(
            frame["evaluation_step"],
            smooth,
            color="#222222",
            linewidth=1.3,
            label="5-step moving mean",
        )
        axis.set_ylim(0.0, 0.72)
        axis.set_ylabel("JS divergence")
        axis.set_title(f"Seed {seed}", loc="left", fontsize=11)
        axis.grid(True, linestyle="--", alpha=0.32)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Evaluation step (100 episodes)")
    figure.suptitle(
        "Deterministic Actor Evolution under a Fixed Selected Discriminator",
        fontsize=13,
        y=0.995,
    )
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.07, top=0.94, hspace=0.24)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_summary(frames: dict[int, pd.DataFrame], output_path: Path) -> None:
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    fields = (
        ("actor_expert_js_divergence", "JS divergence", "(a) Expert-score distribution gap"),
        ("auc_distance_from_0_5", "|AUC - 0.5|", "(b) Discriminator separability gap"),
        ("action_rmse", "Action RMSE", "(c) Direct action gap"),
    )
    for axis, (field, ylabel, title) in zip(axes, fields):
        for seed, frame in frames.items():
            axis.plot(
                frame["evaluation_step"],
                frame[field],
                color=COLORS.get(seed),
                linewidth=1.6,
                label=f"Seed {seed}",
            )
        axis.set_xlabel("Evaluation step (100 episodes)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.32)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_ylim(0.0, 0.72)
    axes[1].set_ylim(bottom=0.0)
    axes[2].set_ylim(bottom=0.0)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.15, top=0.91, wspace=0.28)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_js_mean_sem(
    frames: dict[int, pd.DataFrame], output_dir: Path
) -> None:
    """Plot the cross-seed JS mean with a mean +/- one-SEM band."""
    _style()
    combined = pd.concat(
        [
            frame[["evaluation_step", "actor_expert_js_divergence"]].assign(
                seed=seed
            )
            for seed, frame in frames.items()
        ],
        ignore_index=True,
    )
    counts = combined.groupby("evaluation_step")["seed"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != len(frames):
        raise ValueError("Every evaluation step must contain all requested seeds.")
    grouped = combined.groupby("evaluation_step")["actor_expert_js_divergence"]
    statistics = grouped.agg(sample_count="count", mean="mean", std="std").reset_index()
    statistics["sem"] = statistics["std"] / np.sqrt(statistics["sample_count"])
    statistics["mean_minus_sem"] = np.maximum(
        0.0, statistics["mean"] - statistics["sem"]
    )
    statistics["mean_plus_sem"] = statistics["mean"] + statistics["sem"]
    statistics.to_csv(output_dir / "actor_expert_js_mean_sem.csv", index=False)

    x = statistics["evaluation_step"].to_numpy(dtype=float)
    mean = statistics["mean"].to_numpy(dtype=float)
    lower = statistics["mean_minus_sem"].to_numpy(dtype=float)
    upper = statistics["mean_plus_sem"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    axis.fill_between(
        x,
        lower,
        upper,
        color="#4C78A8",
        alpha=0.22,
        linewidth=0.0,
        label=f"Mean +/- 1 SEM (n={len(frames)})",
    )
    axis.plot(
        x,
        mean,
        color="#1F5A94",
        linestyle="-",
        linewidth=1.8,
        label="Mean JS divergence",
    )
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(0.0, 0.70)
    axis.set_xlabel("Evaluation step (100 episodes)")
    axis.set_ylabel("JS divergence")
    axis.set_title("Actor-Expert JS Divergence over Training")
    axis.grid(True, linestyle="--", alpha=0.32)
    axis.legend(loc="upper right", fontsize=8, frameon=True)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.91)
    figure.savefig(output_dir / "actor_expert_js_mean_sem.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    frames = {
        seed: pd.read_csv(output_dir / f"seed_{seed}_actor_snapshot_metrics.csv")
        for seed in args.seeds
    }
    plot_per_seed(frames, output_dir / "three_seed_actor_snapshot_js.png")
    plot_summary(frames, output_dir / "three_seed_actor_snapshot_evolution_summary.png")
    plot_js_mean_sem(frames, output_dir)
    print(f"Actor snapshot plots written to: {output_dir}")


if __name__ == "__main__":
    main()
