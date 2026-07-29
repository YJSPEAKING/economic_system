from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
from matplotlib.font_manager import FontProperties


BASE_DIR = Path(__file__).resolve().parent
SWANLOG_DIR = BASE_DIR / "swanlog"
OUT_DIR = (
    BASE_DIR
    / "analysis_plots"
    / "five_metrics_two_comparisons_aulc_sem"
)
RAW_DIR = OUT_DIR / "raw_metrics"


def sample_sem(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))

GROUP_DIRS = {
    "TD3": SWANLOG_DIR / "TD3",
    "GAIL+TD3": SWANLOG_DIR / "GAIL+TD3",
    "Transformer+GAIL+TD3": SWANLOG_DIR / "Transformer+GAIL+TD3",
}

METRIC_ALIASES = {
    "survival": ["每百回合/存活天数"],
    "production": [
        "每百回合/累计收益/生产企业",
        "每百回合/累计奖励/生产企业1",
    ],
    "consumption": [
        "每百回合/累计收益/消费企业",
        "每百回合/累计奖励/消费企业1",
    ],
    "bank": [
        "每百回合/累计收益/银行",
        "每百回合/累计奖励/银行",
    ],
}
METRIC_BY_KEY = {
    key: metric_id
    for metric_id, aliases in METRIC_ALIASES.items()
    for key in aliases
}
BASE_METRICS = ["survival", "production", "consumption", "bank"]
METRIC_ORDER = [*BASE_METRICS, "naulc"]

METRIC_TITLES = {
    "survival": "(a) 每百回合/存活天数",
    "production": "(b) 每百回合/累计收益/生产企业",
    "consumption": "(c) 每百回合/累计收益/消费企业",
    "bank": "(d) 每百回合/累计收益/银行",
    "naulc": "(e) 累计归一化学习曲线下面积",
}
Y_LABELS = {
    "survival": "Days",
    "production": "Income",
    "consumption": "Income",
    "bank": "Income",
    "naulc": "nAULC",
}
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
COMPARISONS = [
    {
        "groups": ["TD3", "GAIL+TD3"],
        "filename": "td3_vs_gail_td3_five_metrics_aulc_sem.png",
    },
    {
        "groups": ["GAIL+TD3", "Transformer+GAIL+TD3"],
        "filename": "gail_td3_vs_transformer_five_metrics_aulc_sem.png",
    },
]

EXPERT_SAMPLE_ROWS = 168_548
EXPERT_SUCCESSFUL_EPISODES = 1_720
EXPERT_MEAN_SURVIVAL_DAYS = EXPERT_SAMPLE_ROWS / EXPERT_SUCCESSFUL_EPISODES
MAX_EPISODE_DAYS = 100.0
EXPECTED_RUNS_PER_GROUP = 8
CHINESE_TITLE_FONT = FontProperties(family="Microsoft YaHei", size=10.5)


def import_swanlab_reader():
    from swanlab.data.porter.datastore import DataStore
    from swanlab.proto.v0 import BaseModel

    return DataStore, BaseModel


def seed_from_config(run_dir: Path) -> str:
    config_path = run_dir / "files" / "config.yaml"
    if not config_path.exists():
        return ""
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"random_seed:\s*([0-9]+)", text)
    return match.group(1) if match else ""


def discover_runs(group_name: str) -> List[Path]:
    group_dir = GROUP_DIRS[group_name]
    runs = sorted(
        path
        for path in group_dir.iterdir()
        if path.is_dir() and path.name.startswith("run-")
    )
    if len(runs) != EXPECTED_RUNS_PER_GROUP:
        raise RuntimeError(
            f"{group_name} should contain {EXPECTED_RUNS_PER_GROUP} runs, "
            f"but {len(runs)} were found in {group_dir}."
        )
    return runs


def cache_paths(run_dir: Path, group_name: str) -> tuple[Path, Path]:
    group_dir = RAW_DIR / group_name.replace("+", "_plus_")
    group_dir.mkdir(parents=True, exist_ok=True)
    return (
        group_dir / f"{run_dir.name}_metrics.csv",
        group_dir / f"{run_dir.name}_meta.json",
    )


def export_run_metrics(run_dir: Path, group_name: str) -> Path:
    csv_path, meta_path = cache_paths(run_dir, group_name)
    if csv_path.exists() and meta_path.exists():
        return csv_path

    backup_path = run_dir / "backup.swanlab"
    if not backup_path.exists():
        raise FileNotFoundError(f"Missing backup.swanlab: {backup_path}")

    DataStore, BaseModel = import_swanlab_reader()
    metrics: Dict[str, Dict[int, float]] = {
        metric_id: {} for metric_id in BASE_METRICS
    }
    datastore = DataStore()
    datastore.open_for_scan(str(backup_path))
    try:
        while True:
            record = datastore.scan()
            if record is None:
                break
            try:
                data = BaseModel.from_record(record).model_dump()
            except Exception:
                continue

            metric_id = METRIC_BY_KEY.get(data.get("key"))
            if metric_id is None:
                continue
            metric = data.get("metric")
            value = metric.get("data") if isinstance(metric, dict) else metric
            try:
                step = int(data.get("step"))
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                metrics[metric_id][step] = value
    finally:
        try:
            datastore.close()
        except Exception:
            pass

    missing = [metric_id for metric_id, values in metrics.items() if not values]
    if missing:
        raise RuntimeError(f"{run_dir.name} is missing metrics: {missing}")

    seed = seed_from_config(run_dir)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "seed", "metric_id", "step", "value"])
        for metric_id in BASE_METRICS:
            for step, value in sorted(metrics[metric_id].items()):
                writer.writerow([run_dir.name, seed, metric_id, step, value])

    meta_path.write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "seed": seed,
                "metric_counts": {
                    metric_id: len(values)
                    for metric_id, values in metrics.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path


def load_run(csv_path: Path) -> dict:
    run_name = ""
    seed = ""
    values: Dict[str, Dict[int, float]] = {
        metric_id: {} for metric_id in BASE_METRICS
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            metric_id = row["metric_id"]
            if metric_id not in values:
                continue
            run_name = row["run"]
            seed = row["seed"]
            values[metric_id][int(row["step"])] = float(row["value"])
    return {"run": run_name, "seed": seed, "values": values}


def add_cumulative_normalized_aulc(run: dict, common_steps: Sequence[int]) -> None:
    survival = run["values"]["survival"]
    if any(step not in survival for step in common_steps):
        raise RuntimeError(f"Incomplete survival series in {run['run']}")
    if len(common_steps) < 2:
        raise RuntimeError("At least two evaluation steps are required for AULC.")

    total_horizon = common_steps[-1] - common_steps[0]
    if total_horizon <= 0:
        raise RuntimeError("Evaluation steps must be strictly increasing.")

    area = 0.0
    values = {common_steps[0]: 0.0}
    for previous_step, step in zip(common_steps[:-1], common_steps[1:]):
        width = step - previous_step
        area += width * (
            survival[previous_step] + survival[step]
        ) / 2.0
        values[step] = area / (total_horizon * MAX_EPISODE_DAYS)
    run["values"]["naulc"] = values


def aggregate_sem(runs: Sequence[dict], metric_id: str) -> Optional[dict]:
    if not runs:
        return None
    steps = sorted(
        set.intersection(
            *(set(run["values"][metric_id].keys()) for run in runs)
        )
    )
    if not steps:
        return None

    means: List[float] = []
    sems: List[float] = []
    for step in steps:
        values = [run["values"][metric_id][step] for run in runs]
        mean = sum(values) / len(values)
        means.append(mean)
        if len(values) <= 1:
            sems.append(0.0)
        else:
            variance = sum((value - mean) ** 2 for value in values) / (
                len(values) - 1
            )
            sems.append(math.sqrt(variance) / math.sqrt(len(values)))

    return {
        "steps": steps,
        "means": means,
        "sems": sems,
        "n_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
    }


def build_data() -> tuple[Dict[str, Dict[str, dict]], Dict[str, List[dict]]]:
    runs_by_group: Dict[str, List[dict]] = {}
    for group_name in GROUP_DIRS:
        csv_paths = [
            export_run_metrics(run_dir, group_name)
            for run_dir in discover_runs(group_name)
        ]
        runs_by_group[group_name] = [load_run(path) for path in csv_paths]

    all_survival = [
        run["values"]["survival"]
        for runs in runs_by_group.values()
        for run in runs
    ]
    common_steps = sorted(
        set.intersection(*(set(values.keys()) for values in all_survival))
    )
    if common_steps != list(range(common_steps[0], common_steps[-1] + 1)):
        raise RuntimeError("Survival evaluation steps are not contiguous.")
    for runs in runs_by_group.values():
        for run in runs:
            add_cumulative_normalized_aulc(run, common_steps)

    data = {
        group_name: {
            metric_id: aggregate_sem(runs, metric_id)
            for metric_id in METRIC_ORDER
        }
        for group_name, runs in runs_by_group.items()
    }
    return data, runs_by_group


def configure_matplotlib() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "SimSun"],
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.8,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.7,
            "savefig.dpi": 300,
        }
    )


def apply_axis_scale(ax, metric_id: str, upper_values: Iterable[float]) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if metric_id == "naulc":
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        return

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{int(round(value))}")
    )
    if metric_id == "survival":
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        return

    upper = max(max(upper_values, default=1.0), 1.0)
    tick_step = 50000 if metric_id in ("production", "consumption") else 5000
    upper = math.ceil(upper / tick_step) * tick_step
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))


def style_axis(ax) -> None:
    ax.grid(
        True,
        linestyle="-.",
        linewidth=0.65,
        color="#a8a8a8",
        alpha=0.65,
    )
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def plot_metric(ax, metric_id: str, groups: Sequence[str], data: dict) -> None:
    upper_values: List[float] = []
    for group_name in groups:
        aggregate = data[group_name][metric_id]
        steps = aggregate["steps"]
        means = aggregate["means"]
        sems = aggregate["sems"]
        lower = [max(0.0, mean - sem) for mean, sem in zip(means, sems)]
        upper = [mean + sem for mean, sem in zip(means, sems)]
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

    if metric_id == "survival":
        ax.axhline(
            y=EXPERT_MEAN_SURVIVAL_DAYS,
            color="#4d4d4d",
            linestyle=(0, (5, 3)),
            linewidth=1.15,
            label=f"Expert trajectories ({EXPERT_MEAN_SURVIVAL_DAYS:.2f} days)",
            zorder=3,
        )

    ax.set_title(
        METRIC_TITLES[metric_id],
        pad=7,
        fontproperties=CHINESE_TITLE_FONT,
    )
    ax.set_xlabel("Evaluation Step (100 Episodes)")
    ax.set_ylabel(Y_LABELS[metric_id], labelpad=10)
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 10))
    apply_axis_scale(ax, metric_id, upper_values)
    style_axis(ax)

    legend_location = "lower right" if metric_id in ("survival", "naulc") else "upper left"
    if metric_id == "bank":
        legend_location = "upper right"
    ax.legend(
        loc=legend_location,
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        edgecolor="#333333",
        borderpad=0.4,
        labelspacing=0.25,
        handlelength=2.2,
    )


def render_comparison(data: dict, groups: Sequence[str], filename: str) -> Path:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.2, 7.5))
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    for ax, metric_id in zip(axes, METRIC_ORDER):
        plot_metric(ax, metric_id, groups, data)

    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        bottom=0.085,
        top=0.955,
        wspace=0.50,
        hspace=0.38,
    )
    output_path = OUT_DIR / filename
    fig.savefig(output_path, facecolor="white", dpi=300)
    plt.close(fig)
    return output_path


def write_summary(data: dict, runs_by_group: Dict[str, List[dict]]) -> Path:
    summary_path = OUT_DIR / "five_metrics_aulc_sem_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "group",
                "metric_id",
                "n_runs",
                "first_step",
                "last_step",
                "last_mean",
                "last_sem",
                "seeds",
            ]
        )
        for group_name in GROUP_DIRS:
            for metric_id in METRIC_ORDER:
                aggregate = data[group_name][metric_id]
                writer.writerow(
                    [
                        group_name,
                        metric_id,
                        aggregate["n_runs"],
                        aggregate["steps"][0],
                        aggregate["steps"][-1],
                        aggregate["means"][-1],
                        aggregate["sems"][-1],
                        ";".join(aggregate["seeds"]),
                    ]
                )

    per_seed_path = OUT_DIR / "naulc_by_seed.csv"
    with per_seed_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "run", "seed", "step", "naulc"])
        for group_name, runs in runs_by_group.items():
            for run in runs:
                for step, value in sorted(run["values"]["naulc"].items()):
                    writer.writerow(
                        [group_name, run["run"], run["seed"], step, value]
                    )
    return summary_path


def write_final_performance_summary(
    runs_by_group: Dict[str, List[dict]],
) -> Path:
    """Write the seed-level final-performance statistics used in Table 6-2."""
    summary_path = OUT_DIR / "final_performance_last20_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "group",
                "metric_id",
                "basis",
                "start_step",
                "end_step",
                "n_runs",
                "mean",
                "sem",
                "relative_sem_percent",
            ]
        )

        for group_name, runs in runs_by_group.items():
            for metric_id in ("survival", "production", "consumption", "bank"):
                common_steps = sorted(
                    set.intersection(
                        *(set(run["values"][metric_id]) for run in runs)
                    )
                )
                stable_steps = common_steps[-20:]
                seed_values = [
                    statistics.mean(
                        run["values"][metric_id][step] for step in stable_steps
                    )
                    for run in runs
                ]
                mean_value = statistics.mean(seed_values)
                sem_value = sample_sem(seed_values)
                relative_sem = (
                    sem_value / abs(mean_value) * 100.0 if mean_value else 0.0
                )
                writer.writerow(
                    [
                        group_name,
                        metric_id,
                        "last_20_evaluation_steps",
                        stable_steps[0],
                        stable_steps[-1],
                        len(seed_values),
                        mean_value,
                        sem_value,
                        relative_sem,
                    ]
                )

            final_step = min(
                max(run["values"]["naulc"]) for run in runs
            )
            seed_values = [
                run["values"]["naulc"][final_step] for run in runs
            ]
            mean_value = statistics.mean(seed_values)
            sem_value = sample_sem(seed_values)
            relative_sem = (
                sem_value / abs(mean_value) * 100.0 if mean_value else 0.0
            )
            writer.writerow(
                [
                    group_name,
                    "naulc",
                    "final_evaluation_step",
                    final_step,
                    final_step,
                    len(seed_values),
                    mean_value,
                    sem_value,
                    relative_sem,
                ]
            )

    return summary_path


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, runs_by_group = build_data()
    summary_path = write_summary(data, runs_by_group)
    final_performance_path = write_final_performance_summary(runs_by_group)
    print(summary_path.resolve())
    print(final_performance_path.resolve())
    for comparison in COMPARISONS:
        output_path = render_comparison(data=data, **comparison)
        print(output_path.resolve())


if __name__ == "__main__":
    main()
