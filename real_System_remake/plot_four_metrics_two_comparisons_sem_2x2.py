from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

import plot_four_metrics_three_algorithms_sem_2x2 as base


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    BASE_DIR
    / "analysis_plots"
    / "four_metrics_two_comparisons_sem_2x2"
)

EXPERT_SAMPLE_ROWS = 168_548
EXPERT_SUCCESSFUL_EPISODES = 1_720
EXPERT_MEAN_SURVIVAL_DAYS = (
    EXPERT_SAMPLE_ROWS / EXPERT_SUCCESSFUL_EPISODES
)

COMPARISONS = [
    {
        "groups": ["TD3", "GAIL+TD3"],
        "title": "TD3 vs GAIL+TD3: Mean Curves with SEM",
        "filename": "td3_vs_gail_td3_four_metrics_sem_2x2.png",
    },
    {
        "groups": ["GAIL+TD3", "Transformer+GAIL+TD3"],
        "title": "GAIL+TD3 vs Trans.+GAIL+TD3: Mean Curves with SEM",
        "filename": "gail_td3_vs_transformer_four_metrics_sem_2x2.png",
    },
]


def plot_metric(
    ax,
    metric_id: str,
    title: str,
    ylabel: str,
    data: dict,
    groups: List[str],
) -> None:
    upper_values: List[float] = []
    for group_name in groups:
        aggregate = data[group_name][metric_id]
        steps = aggregate["steps"]
        means = aggregate["means"]
        sems = aggregate["sems"]
        lower = [
            max(0.0, mean - sem)
            for mean, sem in zip(means, sems)
        ]
        upper = [
            mean + sem
            for mean, sem in zip(means, sems)
        ]
        upper_values.extend(upper)

        color = base.COLORS[group_name]
        ax.fill_between(
            steps,
            lower,
            upper,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            steps,
            means,
            color=color,
            label=base.LEGEND_LABELS[group_name],
            zorder=2,
        )

    if metric_id == "survival":
        ax.axhline(
            y=EXPERT_MEAN_SURVIVAL_DAYS,
            color="#4d4d4d",
            linestyle=(0, (5, 3)),
            linewidth=1.2,
            label=(
                "Expert trajectories "
                f"({EXPERT_MEAN_SURVIVAL_DAYS:.2f} days)"
            ),
            zorder=3,
        )

    ax.set_title(title, pad=8, fontproperties=base.CHINESE_TITLE_FONT)
    ax.set_xlabel("Evaluation Step (100 Episodes)")
    ax.set_ylabel(ylabel, labelpad=11)
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 10))
    base.configure_axis_scale(ax, metric_id, upper_values)
    base.style_axis(ax)

    legend_location = "lower right" if metric_id == "survival" else "upper left"
    if metric_id == "bank":
        legend_location = "upper right"
    ax.legend(
        loc=legend_location,
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        edgecolor="#333333",
        borderpad=0.45,
        labelspacing=0.3,
        handlelength=2.4,
    )


def render_comparison(
    data: Dict[str, Dict[str, dict]],
    groups: List[str],
    title: str,
    filename: str,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
    for ax, (metric_id, metric_title, ylabel) in zip(
        axes.ravel(), base.METRICS
    ):
        plot_metric(
            ax,
            metric_id,
            metric_title,
            ylabel,
            data,
            groups,
        )

    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.08,
        top=0.91,
        wspace=0.25,
        hspace=0.31,
    )

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, facecolor="white", dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    matplotlib.use("Agg")
    base.configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_groups = sorted(
        {
            group_name
            for comparison in COMPARISONS
            for group_name in comparison["groups"]
        }
    )
    loaded = {
        group_name: [
            base.load_run(run_name, group_name)
            for run_name in base.RUNS[group_name]
        ]
        for group_name in required_groups
    }
    data: Dict[str, Dict[str, dict]] = {
        group_name: {
            metric_id: base.aggregate_sem(group_runs, metric_id)
            for metric_id, _title, _ylabel in base.METRICS
        }
        for group_name, group_runs in loaded.items()
    }

    for comparison in COMPARISONS:
        output_path = render_comparison(data=data, **comparison)
        print(output_path.resolve())


if __name__ == "__main__":
    main()
