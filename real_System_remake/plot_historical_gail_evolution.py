"""Render phase-2 GAIL evolution plots without importing PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis_plots" / "historical_gail_evolution"
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


def plot_episode_js(
    seed_frames: dict[int, pd.DataFrame], output_path: Path
) -> None:
    _style()
    figure, axes = plt.subplots(
        len(seed_frames), 1, figsize=(9.2, 3.05 * len(seed_frames)), sharex=True
    )
    if len(seed_frames) == 1:
        axes = [axes]
    for axis, (seed, frame) in zip(axes, seed_frames.items()):
        color = COLORS.get(seed, "#2C7FB8")
        axis.plot(
            frame["episode"],
            frame["generated_expert_js_divergence"],
            color=color,
            alpha=0.16,
            linewidth=0.55,
            label="Per-episode JS",
        )
        axis.plot(
            frame["episode"],
            frame["rolling_js_mean"],
            color=color,
            linewidth=1.7,
            label="100-episode rolling mean",
        )
        axis.set_ylim(0.0, 0.72)
        axis.set_ylabel("JS divergence")
        axis.set_title(f"Seed {seed}", loc="left", fontsize=11)
        axis.grid(True, linestyle="--", alpha=0.32)
        axis.legend(loc="upper right", frameon=True, fontsize=8)
    axes[-1].set_xlabel("Training episode")
    figure.suptitle(
        "Historical Generated-vs-Expert Divergence under a Fixed Selected Discriminator",
        fontsize=13,
        y=0.995,
    )
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.07, top=0.94, hspace=0.24)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_summary(
    window_frames: dict[int, pd.DataFrame],
    validation_frames: dict[int, pd.DataFrame],
    output_path: Path,
    window_episodes: int,
) -> None:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for seed, frame in window_frames.items():
        axes[0].plot(
            frame["window_index"],
            frame["pooled_generated_expert_js_divergence"],
            color=COLORS.get(seed),
            linewidth=1.7,
            marker="o",
            markersize=2.5,
            label=f"Seed {seed}",
        )
    axes[0].set_xlabel(f"{window_episodes}-episode window")
    axes[0].set_ylabel("Pooled JS divergence")
    axes[0].set_title("(a) Fixed-discriminator retrospective score")
    axes[0].set_ylim(0.0, 0.72)
    axes[0].grid(True, linestyle="--", alpha=0.32)
    axes[0].legend(loc="best", fontsize=8)

    for seed, frame in validation_frames.items():
        axes[1].plot(
            frame["training_step"],
            frame["random_auc"],
            color=COLORS.get(seed),
            linewidth=1.7,
            marker="o",
            markersize=3,
            label=f"Seed {seed}",
        )
    axes[1].axhline(
        0.8,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="Configured AUC threshold",
    )
    axes[1].set_xlabel("TD3 parameter-update step")
    axes[1].set_ylabel("Expert-vs-random ROC-AUC")
    axes[1].set_title("(b) Online discriminator validation history")
    axes[1].set_ylim(0.5, 1.01)
    axes[1].grid(True, linestyle="--", alpha=0.32)
    axes[1].legend(loc="lower left", fontsize=8)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.92, wspace=0.25)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--window-episodes", type=int, default=100)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    episode_frames = {
        seed: pd.read_csv(output_dir / f"seed_{seed}_per_episode_js.csv")
        for seed in args.seeds
    }
    window_frames = {
        seed: pd.read_csv(
            output_dir / f"seed_{seed}_per_{args.window_episodes}_episode_js.csv"
        )
        for seed in args.seeds
    }
    validation_frames = {
        seed: pd.read_csv(output_dir / f"seed_{seed}_discriminator_validation_history.csv")
        for seed in args.seeds
    }
    plot_episode_js(
        episode_frames, output_dir / "three_seed_per_episode_js_fixed_discriminator.png"
    )
    plot_summary(
        window_frames,
        validation_frames,
        output_dir / "three_seed_gail_evolution_summary.png",
        args.window_episodes,
    )
    print(f"Phase-2 plots written to: {output_dir}")


if __name__ == "__main__":
    main()
