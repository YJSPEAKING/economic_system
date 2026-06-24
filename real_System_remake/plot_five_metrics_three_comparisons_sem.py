from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib


BASE_DIR = Path(__file__).resolve().parent
SWANLOG_DIR = BASE_DIR / "swanlog"
OUT_DIR = BASE_DIR / "analysis_plots" / "dscr_description_20260624_five_metrics_sem"
RAW_DIR = OUT_DIR / "raw_metrics"

TD3_RUNS = [
    "run-20260621_202812-6ig7ywjq5ma7q4r3n4d2k",
    "run-20260621_221653-3oxu02lub6ic88gg2tlyv",
    "run-20260621_235606-cxmnv919c5vgxd18ag5m2",
    "run-20260622_012518-o3wa23ld5k0anno4nhwb9",
    "run-20260622_025851-vc8eettjrkns0ask0j149",
    "run-20260622_043232-p1wsv6nx8qn3fwe50d8dd",
    "run-20260622_063205-sgs3p05et5bj1by7pfa86",
    "run-20260622_080822-vx1bs5edgj8z9ihb6xe1x",
]

GAIL_TD3_RUNS = [
    "run-20260621_193255-5pg7iyrvmcx34frn6ale5",
    "run-20260621_223131-dm2s10i96kbbd7fky6owy",
    "run-20260622_004812-zslinp7qvacgwbfuqujtw",
    "run-20260622_031937-daocyl38dnxhsxwoxtkov",
    "run-20260622_053309-3gob8caas2nn2pu2ozthf",
    "run-20260622_075227-p7wj1wfq83b3mhkyo053i",
    "run-20260622_110022-x8utkar2er1eerentcryu",
    "run-20260622_134950-g5g4dxxf5ozv6uufkbfmk",
]

TRANSFORMER_GAIL_TD3_RUNS = [
    "run-20260622_223056-o8iu1b7pexa77lh7sd8cx",
    "run-20260623_033112-glurxrgn07nhly7dvjkjp",
    "run-20260623_080809-pdrdexgwuux1bd54os63u",
    "run-20260623_131038-y6xubdud4wjny573weehp",
    "run-20260623_172840-t8vokks22cdly7camzcud",
    "run-20260623_212641-cj8cxudsz6o3ilixh1qc8",
    "run-20260624_024311-i0350u1ei1u9bcogckxi4",
    "run-20260624_073818-an8ie0ctj7ii46dc18f2a",
]

GROUPS = {
    "TD3": TD3_RUNS,
    "GAIL+TD3": GAIL_TD3_RUNS,
    "Transformer+GAIL+TD3": TRANSFORMER_GAIL_TD3_RUNS,
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
    "production_dscr": ["每百回合/偿债能力/生产企业"],
}

METRIC_BY_KEY = {
    key: metric_id
    for metric_id, aliases in METRIC_ALIASES.items()
    for key in aliases
}

METRIC_ORDER = [
    "survival",
    "production",
    "consumption",
    "bank",
    "production_dscr",
]

METRIC_TITLES = {
    "survival": "(a) 每百回合/存活天数",
    "production": "(b) 每百回合/累计收益/生产企业",
    "consumption": "(c) 每百回合/累计收益/消费企业",
    "bank": "(d) 每百回合/累计收益/银行",
    "production_dscr": "(e) 每百回合/偿债能力/生产企业",
}

Y_LABELS = {
    "survival": "Days",
    "production": "Income",
    "consumption": "Income",
    "bank": "Income",
    "production_dscr": "DSCR",
}

COLORS = {
    "TD3": "#1f77b4",
    "GAIL+TD3": "#ff7f0e",
    "Transformer+GAIL+TD3": "#2ca02c",
}

COMPARISONS = [
    {
        "name": "td3_vs_gail_td3_five_metrics_sem",
        "title": "TD3 vs GAIL+TD3",
        "groups": ["TD3", "GAIL+TD3"],
    },
    {
        "name": "transformer_gail_td3_vs_gail_td3_five_metrics_sem",
        "title": "Transformer+GAIL+TD3 vs GAIL+TD3",
        "groups": ["GAIL+TD3", "Transformer+GAIL+TD3"],
    },
    {
        "name": "td3_gail_td3_transformer_five_metrics_sem",
        "title": "TD3 vs GAIL+TD3 vs Transformer+GAIL+TD3",
        "groups": ["TD3", "GAIL+TD3", "Transformer+GAIL+TD3"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot five SwanLab metrics for three algorithm comparisons."
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def conda_python_paths() -> tuple[Path, Path]:
    executable = Path(sys.executable).resolve()
    if executable.parent.parent.name.lower() == "envs":
        conda_root = executable.parent.parent.parent
    else:
        conda_root = executable.parent
    return conda_root / "python.exe", conda_root / "envs" / "ppo" / "python.exe"


def cache_is_complete() -> bool:
    for group_name, run_names in GROUPS.items():
        group_dir = RAW_DIR / group_name.replace("+", "_plus_")
        for run_name in run_names:
            if not (group_dir / f"{run_name}_metrics.csv").exists():
                return False
            if not (group_dir / f"{run_name}_meta.json").exists():
                return False
    return True


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


def export_run_metrics(run_dir: Path, group_name: str) -> Path:
    group_dir = RAW_DIR / group_name.replace("+", "_plus_")
    group_dir.mkdir(parents=True, exist_ok=True)
    csv_path = group_dir / f"{run_dir.name}_metrics.csv"
    meta_path = group_dir / f"{run_dir.name}_meta.json"
    if csv_path.exists() and meta_path.exists():
        return csv_path

    backup_path = run_dir / "backup.swanlab"
    if not backup_path.exists():
        raise FileNotFoundError(f"Missing backup.swanlab: {backup_path}")

    DataStore, BaseModel = import_swanlab_reader()
    metrics: Dict[str, Dict[int, float]] = {
        metric_id: {} for metric_id in METRIC_ORDER
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

    seed = seed_from_config(run_dir)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "seed", "metric_id", "step", "value"])
        for metric_id in METRIC_ORDER:
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


def load_series(csv_paths: Iterable[Path]) -> Dict[str, List[dict]]:
    by_metric: Dict[str, List[dict]] = {
        metric_id: [] for metric_id in METRIC_ORDER
    }
    for csv_path in csv_paths:
        run_name = ""
        seed = ""
        values_by_metric: Dict[str, Dict[int, float]] = {
            metric_id: {} for metric_id in METRIC_ORDER
        }
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                metric_id = row["metric_id"]
                if metric_id not in values_by_metric:
                    continue
                run_name = row["run"]
                seed = row["seed"]
                values_by_metric[metric_id][int(row["step"])] = float(row["value"])
        for metric_id, values in values_by_metric.items():
            if values:
                by_metric[metric_id].append(
                    {"run": run_name, "seed": seed, "values": values}
                )
    return by_metric


def aggregate_sem(series: Sequence[dict]) -> Optional[dict]:
    if not series:
        return None
    steps = sorted(
        set.intersection(*(set(item["values"].keys()) for item in series))
    )
    if not steps:
        return None

    means: List[float] = []
    sems: List[float] = []
    ns: List[int] = []
    for step in steps:
        values = [item["values"][step] for item in series]
        n = len(values)
        mean = sum(values) / n
        means.append(mean)
        if n <= 1:
            sems.append(0.0)
        else:
            variance = sum((value - mean) ** 2 for value in values) / (n - 1)
            sems.append(math.sqrt(variance) / math.sqrt(n))
        ns.append(n)

    return {
        "steps": steps,
        "mean": means,
        "sem": sems,
        "n_by_step": ns,
        "n_runs": len(series),
        "seeds": [item["seed"] for item in series],
    }


def build_group_data() -> Dict[str, Dict[str, Optional[dict]]]:
    group_data: Dict[str, Dict[str, Optional[dict]]] = {}
    for group_name, run_names in GROUPS.items():
        csv_paths = [
            export_run_metrics(SWANLOG_DIR / run_name, group_name)
            for run_name in run_names
        ]
        by_metric = load_series(csv_paths)
        group_data[group_name] = {
            metric_id: aggregate_sem(by_metric[metric_id])
            for metric_id in METRIC_ORDER
        }
    return group_data


def configure_matplotlib() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
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
            "savefig.dpi": 220,
        }
    )


def integer_formatter(value: float, _position: object) -> str:
    return f"{int(round(value))}"


def apply_axis_scale(ax, metric_id: str, visible_upper: float) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

    ax.yaxis.set_major_formatter(FuncFormatter(integer_formatter))
    if metric_id == "survival":
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        return

    upper = max(visible_upper, 1.0)
    if metric_id in ("production", "consumption"):
        tick_step = 50000
        upper = math.ceil(upper / tick_step) * tick_step
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    elif metric_id == "bank":
        tick_step = 5000
        upper = math.ceil(upper / tick_step) * tick_step
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    else:
        upper *= 1.05
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_ylim(0, upper)


def plot_metric(ax, metric_id: str, comparison: dict, group_data: dict) -> None:
    values_for_limit: List[float] = []
    for group_name in comparison["groups"]:
        aggregate = group_data[group_name][metric_id]
        if aggregate is None:
            continue
        steps = aggregate["steps"]
        means = aggregate["mean"]
        sems = aggregate["sem"]
        lower = [max(0.0, mean - sem) for mean, sem in zip(means, sems)]
        upper = [mean + sem for mean, sem in zip(means, sems)]
        values_for_limit.extend(upper)
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
            label=f"{group_name} (n={aggregate['n_runs']})",
            zorder=2,
        )

    ax.set_title(METRIC_TITLES[metric_id], pad=8)
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel(Y_LABELS[metric_id], labelpad=12)
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 10))
    ax.grid(
        True,
        linestyle="-.",
        linewidth=0.65,
        color="#a8a8a8",
        alpha=0.65,
    )
    apply_axis_scale(ax, metric_id, max(values_for_limit or [1.0]))

    legend_location = "upper right" if metric_id in ("bank", "production_dscr") else "lower right"
    ax.legend(
        loc=legend_location,
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        edgecolor="#666666",
        borderpad=0.45,
        labelspacing=0.3,
        handlelength=2.4,
    )
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def plot_comparison(comparison: dict, group_data: dict) -> Path:
    import matplotlib.pyplot as plt

    print(f"Drawing {comparison['name']}...", flush=True)
    fig, axes_grid = plt.subplots(3, 2, figsize=(10.0, 12.0))
    axes = list(axes_grid.ravel())
    axes[-1].axis("off")

    for ax, metric_id in zip(axes, METRIC_ORDER):
        print(f"  Plotting {metric_id}...", flush=True)
        plot_metric(ax, metric_id, comparison, group_data)

    fig.suptitle(
        f"{comparison['title']}: Mean Learning Curves with SEM",
        y=0.972,
    )
    fig.text(
        0.5,
        0.944,
        "Curves show the mean across eight seeds; shaded regions show ±1 SEM.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.06,
        top=0.91,
        wspace=0.28,
        hspace=0.42,
    )
    dscr_position = axes[4].get_position()
    axes[4].set_position(
        [
            0.5 - dscr_position.width / 2,
            dscr_position.y0,
            dscr_position.width,
            dscr_position.height,
        ]
    )

    out_path = OUT_DIR / f"{comparison['name']}.png"
    print(f"  Saving {out_path.name}...", flush=True)
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    return out_path


def write_summary(group_data: dict) -> Path:
    summary_path = OUT_DIR / "five_metrics_sem_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "group",
                "metric_id",
                "n_runs",
                "first_step",
                "last_step",
                "last_step_n",
                "last_mean",
                "last_sem",
                "seeds",
            ]
        )
        for group_name, metrics in group_data.items():
            for metric_id in METRIC_ORDER:
                aggregate = metrics[metric_id]
                if aggregate is None:
                    writer.writerow(
                        [group_name, metric_id, 0, "", "", "", "", "", ""]
                    )
                    continue
                writer.writerow(
                    [
                        group_name,
                        metric_id,
                        aggregate["n_runs"],
                        aggregate["steps"][0],
                        aggregate["steps"][-1],
                        aggregate["n_by_step"][-1],
                        aggregate["mean"][-1],
                        aggregate["sem"][-1],
                        ",".join(aggregate["seeds"]),
                    ]
                )
    return summary_path


def validate_group_data(group_data: dict) -> None:
    missing = []
    for group_name, metrics in group_data.items():
        for metric_id, aggregate in metrics.items():
            if aggregate is None:
                missing.append(f"{group_name}: {metric_id}")
    if missing:
        raise RuntimeError(
            "The following group metrics were not found: " + ", ".join(missing)
        )


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_python, ppo_python = conda_python_paths()
    if not cache_is_complete() and not args.extract_only:
        if ppo_python.exists() and Path(sys.executable).resolve() != ppo_python.resolve():
            print("Metric cache is incomplete; exporting with the ppo environment...", flush=True)
            subprocess.run(
                [str(ppo_python), str(Path(__file__).resolve()), "--extract-only"],
                check=True,
            )

    print("Configuring Matplotlib...", flush=True)
    configure_matplotlib()
    print("Loading cached SwanLab metrics...", flush=True)
    group_data = build_group_data()
    print("Validating metrics...", flush=True)
    validate_group_data(group_data)

    if args.extract_only:
        print("Metric cache export completed.", flush=True)
        return

    if (
        not args.render_only
        and base_python.exists()
        and Path(sys.executable).resolve() != base_python.resolve()
    ):
        print("Rendering with the base Conda Matplotlib environment...", flush=True)
        subprocess.run(
            [str(base_python), str(Path(__file__).resolve()), "--render-only"],
            check=True,
        )
        return

    for comparison in COMPARISONS:
        print(plot_comparison(comparison, group_data).resolve())
    print(write_summary(group_data).resolve())


if __name__ == "__main__":
    main()
