from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    BASE_DIR
    / "analysis_plots"
    / "post_training_dscr"
    / "three_algorithm_recent_100_survival_gt_90"
)
RISK_CSV = DATA_DIR / "three_algorithm_dscr_below_1_share.csv"
OUTPUT_PATH = DATA_DIR / "three_algorithm_dscr_boxplot_and_risk.png"

ORDER = ["TD3", "GAIL+TD3", "Transformer+GAIL+TD3"]
LABELS = ["TD3", "GAIL+TD3", "Transformer+GAIL+TD3"]
COLORS = {
    "TD3": "#4C86E8",
    "GAIL+TD3": "#F08A5D",
    "Transformer+GAIL+TD3": "#63B995",
}


def read_rows(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["algorithm"]: row for row in csv.DictReader(file)}


def configure_matplotlib() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14,
            "axes.linewidth": 1.0,
            "savefig.dpi": 220,
        }
    )


def style_axis(ax) -> None:
    ax.grid(
        axis="y",
        linestyle="-.",
        linewidth=0.65,
        color="#A8A8A8",
        alpha=0.65,
        zorder=0,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", width=1.0, length=4, colors="black")


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    risk_rows = read_rows(RISK_CSV)

    missing_risk = [name for name in ORDER if name not in risk_rows]
    if missing_risk:
        raise RuntimeError(f"Missing algorithms: {missing_risk}")

    configure_matplotlib()
    fig, ax_bar = plt.subplots(figsize=(6.6, 5.6))

    risk_values = [
        float(risk_rows[name]["dscr_below_1_share_percent_all"])
        for name in ORDER
    ]
    bars = ax_bar.bar(
        range(len(ORDER)),
        risk_values,
        width=0.56,
        color=[COLORS[name] for name in ORDER],
        edgecolor="#555555",
        linewidth=0.8,
        zorder=3,
    )
    bar_upper = max(1.0, math.ceil(max(risk_values) * 1.22))
    ax_bar.set_ylim(0, bar_upper)
    ax_bar.yaxis.set_major_locator(MultipleLocator(1))
    ax_bar.set_xticks(range(len(ORDER)))
    ax_bar.set_xticklabels(LABELS)
    ax_bar.set_ylabel("占比（%）", labelpad=10)
    ax_bar.set_title("DSCR < 1 的企业日占比", pad=10)
    style_axis(ax_bar)

    label_offset = bar_upper * 0.025
    for bar, value in zip(bars, risk_values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.14,
        top=0.91,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white", dpi=220)
    plt.close(fig)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
