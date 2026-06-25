from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = (
    BASE_DIR
    / "analysis_plots"
    / "dscr_description_20260624_five_metrics_sem"
    / "raw_metrics"
)
OUTPUT_DIR = BASE_DIR / "analysis_plots" / "four_metrics_three_algorithms_sem_2x2"
OUTPUT_PATH = OUTPUT_DIR / "td3_gail_transformer_four_metrics_sem_2x2.png"
SUMMARY_PATH = OUTPUT_DIR / "four_metrics_sem_summary.csv"

RUNS = {
    "TD3": [
        "run-20260621_202812-6ig7ywjq5ma7q4r3n4d2k",
        "run-20260621_221653-3oxu02lub6ic88gg2tlyv",
        "run-20260621_235606-cxmnv919c5vgxd18ag5m2",
        "run-20260622_012518-o3wa23ld5k0anno4nhwb9",
        "run-20260622_025851-vc8eettjrkns0ask0j149",
        "run-20260622_043232-p1wsv6nx8qn3fwe50d8dd",
        "run-20260622_063205-sgs3p05et5bj1by7pfa86",
        "run-20260622_080822-vx1bs5edgj8z9ihb6xe1x",
    ],
    "GAIL+TD3": [
        "run-20260621_193255-5pg7iyrvmcx34frn6ale5",
        "run-20260621_223131-dm2s10i96kbbd7fky6owy",
        "run-20260622_004812-zslinp7qvacgwbfuqujtw",
        "run-20260622_031937-daocyl38dnxhsxwoxtkov",
        "run-20260622_053309-3gob8caas2nn2pu2ozthf",
        "run-20260622_075227-p7wj1wfq83b3mhkyo053i",
        "run-20260622_110022-x8utkar2er1eerentcryu",
        "run-20260622_134950-g5g4dxxf5ozv6uufkbfmk",
    ],
    "Transformer+GAIL+TD3": [
        "run-20260622_223056-o8iu1b7pexa77lh7sd8cx",
        "run-20260623_033112-glurxrgn07nhly7dvjkjp",
        "run-20260623_080809-pdrdexgwuux1bd54os63u",
        "run-20260623_131038-y6xubdud4wjny573weehp",
        "run-20260623_172840-t8vokks22cdly7camzcud",
        "run-20260623_212641-cj8cxudsz6o3ilixh1qc8",
        "run-20260624_024311-i0350u1ei1u9bcogckxi4",
        "run-20260624_073818-an8ie0ctj7ii46dc18f2a",
    ],
}

CACHE_GROUPS = {
    "TD3": "TD3",
    "GAIL+TD3": "GAIL_plus_TD3",
    "Transformer+GAIL+TD3": "Transformer_plus_GAIL_plus_TD3",
}

METRICS = [
    ("survival", "(a) 每百回合/存活天数", "Days"),
    ("production", "(b) 每百回合/累计收益/生产企业", "Income"),
    ("consumption", "(c) 每百回合/累计收益/消费企业", "Income"),
    ("bank", "(d) 每百回合/累计收益/银行", "Income"),
]

COLORS = {
    "TD3": "#1f77b4",
    "GAIL+TD3": "#ff7f0e",
    "Transformer+GAIL+TD3": "#2ca02c",
}

LEGEND_LABELS = {
    "TD3": "TD3",
    "GAIL+TD3": "GAIL+TD3",
    "Transformer+GAIL+TD3": "Trans.+GAIL+TD3",
}

CHINESE_TITLE_FONT = None


def load_run(run_name: str, group_name: str) -> dict:
    path = CACHE_DIR / CACHE_GROUPS[group_name] / f"{run_name}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached SwanLab metrics: {path}")

    result = {
        metric_id: {}
        for metric_id, _title, _ylabel in METRICS
    }
    seed = ""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            metric_id = row["metric_id"]
            if metric_id not in result:
                continue
            seed = row["seed"]
            result[metric_id][int(row["step"])] = float(row["value"])
    return {"run": run_name, "seed": seed, "metrics": result}


def aggregate_sem(runs: List[dict], metric_id: str) -> dict:
    available = [
        run for run in runs if run["metrics"][metric_id]
    ]
    if len(available) != 8:
        raise RuntimeError(
            f"{metric_id} for {available[0]['run'] if available else 'unknown'} "
            f"has {len(available)} runs instead of 8."
        )

    steps = sorted(
        set.intersection(
            *(set(run["metrics"][metric_id]) for run in available)
        )
    )
    means: List[float] = []
    sems: List[float] = []
    for step in steps:
        values = [run["metrics"][metric_id][step] for run in available]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        means.append(mean)
        sems.append(math.sqrt(variance) / math.sqrt(len(values)))

    return {
        "steps": steps,
        "means": means,
        "sems": sems,
        "n": len(available),
        "seeds": [run["seed"] for run in available],
    }


def configure_matplotlib() -> None:
    global CHINESE_TITLE_FONT

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    CHINESE_TITLE_FONT = FontProperties(family="SimSun", size=12)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 15,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.8,
            "savefig.dpi": 300,
        }
    )


def integer_formatter(value: float, _position: object) -> str:
    return f"{int(round(value))}"


def configure_axis_scale(ax, metric_id: str, upper_values: List[float]) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.yaxis.set_major_formatter(FuncFormatter(integer_formatter))
    if metric_id == "survival":
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        return

    tick_step = 5000 if metric_id == "bank" else 50000
    upper = max(upper_values or [tick_step])
    axis_upper = math.ceil(upper / tick_step) * tick_step
    ax.set_ylim(0, max(axis_upper, tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))


def style_axis(ax) -> None:
    ax.grid(
        True,
        linestyle="-.",
        linewidth=0.65,
        color="#a8a8a8",
        alpha=0.65,
        zorder=0,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", width=1.0, length=4, colors="black")


def plot_metric(ax, metric_id: str, title: str, ylabel: str, data: dict) -> None:
    upper_values: List[float] = []
    for group_name in RUNS:
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
        color = COLORS[group_name]
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
            label=LEGEND_LABELS[group_name],
            zorder=2,
        )

    ax.set_title(title, pad=8, fontproperties=CHINESE_TITLE_FONT)
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel(ylabel, labelpad=11)
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 10))
    configure_axis_scale(ax, metric_id, upper_values)
    style_axis(ax)

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


def write_summary(data: dict) -> None:
    with SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "algorithm",
                "metric",
                "n",
                "first_step",
                "last_step",
                "last_mean",
                "last_sem",
                "seeds",
            ]
        )
        for group_name in RUNS:
            for metric_id, _title, _ylabel in METRICS:
                aggregate = data[group_name][metric_id]
                writer.writerow(
                    [
                        group_name,
                        metric_id,
                        aggregate["n"],
                        aggregate["steps"][0],
                        aggregate["steps"][-1],
                        aggregate["means"][-1],
                        aggregate["sems"][-1],
                        ",".join(aggregate["seeds"]),
                    ]
                )


def main() -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    loaded = {
        group_name: [
            load_run(run_name, group_name)
            for run_name in run_names
        ]
        for group_name, run_names in RUNS.items()
    }
    data: Dict[str, Dict[str, dict]] = {
        group_name: {
            metric_id: aggregate_sem(group_runs, metric_id)
            for metric_id, _title, _ylabel in METRICS
        }
        for group_name, group_runs in loaded.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
    for ax, (metric_id, title, ylabel) in zip(axes.ravel(), METRICS):
        plot_metric(ax, metric_id, title, ylabel, data)

    fig.suptitle(
        "TD3 vs GAIL+TD3 vs Trans.+GAIL+TD3: Mean Curves with SEM",
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.08,
        top=0.91,
        wspace=0.25,
        hspace=0.31,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white", dpi=300)
    plt.close(fig)
    write_summary(data)
    print(OUTPUT_PATH.resolve())
    print(SUMMARY_PATH.resolve())


if __name__ == "__main__":
    main()
