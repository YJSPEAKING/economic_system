from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib


TD3_RUNS = [
    "run-20260613_131335-rcsvupwhsy0jexo9dnn6m",
    "run-20260613_143058-f87zfdo74ksv7bruxoi6e",
    "run-20260613_160815-s0l5sj1nywq68z8lkzm8m",
    "run-20260613_175731-c6tzfp7ahu9imqj0v36hq",
    "run-20260613_193440-jrh0ke9mdg7ich8o4gy2g",
    "run-20260613_210917-28qmhzqb6ococo30f91rd",
    "run-20260613_222722-6ofaznjh0i5j0zam317ab",
    "run-20260614_003148-wpakj7okl4b1rydhcq0rd",
]

GAIL_TD3_RUNS = [
    "run-20260617_001528-dzqg3otf025d93ab14is8",
    "run-20260617_024236-xq7h9trakjenqdu0dpv62",
    "run-20260617_042057-j103cbms9rxw5i3rr1mi2",
    "run-20260617_060854-pppsc2jn3bfk8kl5c0ryj",
    "run-20260617_074649-he8eawnf37e0afhejfz4x",
    "run-20260617_092948-zoblbi9mayxi85o5vagsy",
    "run-20260617_140302-zghmj0imechyy1bsd9w8j",
    "run-20260617_161607-7azajx0oxt6k6fengi74d",
]

TRANSFORMER_GAIL_TD3_RUNS = [
    "run-20260614_001155-e6zpgsd0jm43kibz7j3z8",
    "run-20260614_044536-j0jrf03i8njyfh2j3chi0",
    "run-20260614_085443-mqf40ew763rn58yq4mmoq",
    "run-20260614_140746-hlucuhfis4ip5u856ampf",
    "run-20260614_182034-vtyapfw0uwni3wsfbp35i",
    "run-20260615_000419-94dg1mazir8im8rom8p6h",
    "run-20260615_044714-fyooirk0pq46asrqpxjpl",
    "run-20260615_093606-8bgykq9oh12nqmfjeq5x1",
]

METRIC_ALIASES = {
    "survival": ["\u6bcf\u767e\u56de\u5408/\u5b58\u6d3b\u5929\u6570"],
    "production": [
        "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u751f\u4ea7\u4f01\u4e1a",
        "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u751f\u4ea7\u4f01\u4e1a1",
    ],
    "consumption": [
        "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u6d88\u8d39\u4f01\u4e1a",
        "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u6d88\u8d39\u4f01\u4e1a1",
    ],
    "bank": [
        "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u94f6\u884c",
        "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u94f6\u884c",
    ],
}

METRIC_BY_KEY = {
    key: metric_id
    for metric_id, aliases in METRIC_ALIASES.items()
    for key in aliases
}

METRIC_ORDER = ["survival", "production", "consumption", "bank"]

METRIC_TITLES = {
    "survival": "(a) Survival Days",
    "production": "(b) \u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u751f\u4ea7\u4f01\u4e1a",
    "consumption": "(c) \u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u6d88\u8d39\u4f01\u4e1a",
    "bank": "(d) \u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u94f6\u884c",
}

Y_LABELS = {
    "survival": "Days",
    "production": "Income",
    "consumption": "Income",
    "bank": "Income",
}

GROUPS = {
    "TD3": TD3_RUNS,
    "GAIL+TD3": GAIL_TD3_RUNS,
    "GAIL+TD3+Transformer": TRANSFORMER_GAIL_TD3_RUNS,
}

COMPARISONS = [
    {
        "name": "td3_vs_gail_td3",
        "title": "TD3 vs GAIL+TD3",
        "groups": ["TD3", "GAIL+TD3"],
        "colors": {"TD3": "#1f77b4", "GAIL+TD3": "#ff7f0e"},
    },
    {
        "name": "transformer_gail_td3_vs_gail_td3",
        "title": "GAIL+TD3+Transformer vs GAIL+TD3",
        "groups": ["GAIL+TD3", "GAIL+TD3+Transformer"],
        "colors": {"GAIL+TD3": "#ff7f0e", "GAIL+TD3+Transformer": "#2ca02c"},
    },
]

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Draw two 2x2 Matplotlib comparison figures from SwanLab runs.")
    parser.add_argument("--swanlog-dir", type=Path, default=base / "swanlog")
    parser.add_argument("--out-dir", type=Path, default=base / "analysis_plots" / "paper_algorithm_comparisons_matplotlib")
    parser.add_argument("--align", choices=["common", "union"], default="common")
    parser.add_argument("--ci", choices=["t", "normal"], default="t")
    parser.add_argument("--x-min", type=float, default=1.0)
    parser.add_argument("--x-max", type=float, default=60.0)
    parser.add_argument("--formats", default="svg", help="Comma-separated output formats, e.g. svg,png,pdf. SVG is safest.")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--log-income", action="store_true", help="Use log scale for the three income subplots.")
    return parser.parse_args()


def configure_matplotlib(formats: str) -> None:
    requested = {item.strip().lower() for item in formats.split(",") if item.strip()}
    if requested and requested <= {"svg", "pdf"}:
        matplotlib.use("svg")
    else:
        matplotlib.use("Agg")


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


def read_git_info(run_dir: Path) -> str:
    meta_path = run_dir / "files" / "swanlab-metadata.json"
    if not meta_path.exists():
        return ""
    try:
        return json.dumps(json.loads(meta_path.read_text(encoding="utf-8", errors="ignore")).get("git_info"), ensure_ascii=False)
    except Exception:
        return ""


def export_run_metrics(run_dir: Path, cache_dir: Path, refresh_cache: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{run_dir.name}_metrics.csv"
    meta_path = cache_dir / f"{run_dir.name}_meta.json"
    if csv_path.exists() and meta_path.exists() and not refresh_cache:
        return csv_path

    backup_path = run_dir / "backup.swanlab"
    if not backup_path.exists():
        raise FileNotFoundError(f"Missing backup.swanlab: {backup_path}")

    DataStore, BaseModel = import_swanlab_reader()
    metrics: Dict[str, Dict[int, float]] = {metric_id: {} for metric_id in METRIC_ORDER}
    description = ""
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

            desc = data.get("description") or data.get("notes")
            if isinstance(desc, str) and desc:
                description = desc
            config = data.get("config")
            if isinstance(config, dict):
                config_desc = config.get("description") or config.get("notes")
                if isinstance(config_desc, str) and config_desc:
                    description = config_desc

            key = data.get("key")
            metric_id = METRIC_BY_KEY.get(key)
            if metric_id is None:
                continue
            metric = data.get("metric")
            value = metric.get("data") if isinstance(metric, dict) else metric
            try:
                step = int(data.get("step"))
                value = float(value)
            except Exception:
                continue
            metrics[metric_id][step] = value
    finally:
        try:
            datastore.close()
        except Exception:
            pass

    seed = seed_from_config(run_dir)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
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
                "description": description,
                "git_info": read_git_info(run_dir),
                "metric_counts": {metric_id: len(values) for metric_id, values in metrics.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path


def load_series(csv_paths: Iterable[Path]) -> Dict[str, List[dict]]:
    by_metric: Dict[str, List[dict]] = {metric_id: [] for metric_id in METRIC_ORDER}
    for csv_path in csv_paths:
        run_name = ""
        seed = ""
        values_by_metric: Dict[str, Dict[int, float]] = {metric_id: {} for metric_id in METRIC_ORDER}
        with csv_path.open("r", encoding="utf-8", newline="") as file:
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
                by_metric[metric_id].append({"run": run_name, "seed": seed, "values": values})
    return by_metric


def ci_multiplier(n: int, method: str) -> float:
    if n <= 1:
        return 0.0
    if method == "normal":
        return 1.96
    return T_CRITICAL_95.get(n - 1, 1.96)


def aggregate(series: Sequence[dict], align: str, ci_method: str) -> Optional[dict]:
    if not series:
        return None
    if align == "common":
        steps = sorted(set.intersection(*(set(item["values"].keys()) for item in series)))
    else:
        steps = sorted(set.union(*(set(item["values"].keys()) for item in series)))
    if not steps:
        return None

    means: List[float] = []
    cis: List[float] = []
    ns: List[int] = []
    for step in steps:
        values = [item["values"][step] for item in series if step in item["values"]]
        n = len(values)
        mean = sum(values) / n
        means.append(mean)
        if n <= 1:
            cis.append(0.0)
        else:
            variance = sum((value - mean) ** 2 for value in values) / (n - 1)
            cis.append(ci_multiplier(n, ci_method) * math.sqrt(variance) / math.sqrt(n))
        ns.append(n)
    return {
        "steps": steps,
        "mean": means,
        "ci": cis,
        "n_by_step": ns,
        "n_runs": len(series),
        "seeds": [item["seed"] for item in series],
    }


def build_group_data(args: argparse.Namespace) -> Dict[str, Dict[str, Optional[dict]]]:
    group_data: Dict[str, Dict[str, Optional[dict]]] = {}
    for group_name, run_names in GROUPS.items():
        cache_dir = args.out_dir / "raw_metrics" / group_name.replace("+", "_plus_")
        csv_paths = [
            export_run_metrics(args.swanlog_dir / run_name, cache_dir, args.refresh_cache)
            for run_name in run_names
        ]
        by_metric = load_series(csv_paths)
        group_data[group_name] = {
            metric_id: aggregate(by_metric[metric_id], args.align, args.ci)
            for metric_id in METRIC_ORDER
        }
    return group_data


def setup_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def int_formatter(value: float, _pos: object) -> str:
    return f"{int(round(value))}"


def plot_comparison(comparison: dict, group_data: Dict[str, Dict[str, Optional[dict]]], args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    axes = axes.ravel()

    for ax, metric_id in zip(axes, METRIC_ORDER):
        ax.set_title(METRIC_TITLES[metric_id])
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel(Y_LABELS[metric_id])
        ax.grid(True, linestyle="--", linewidth=0.75, alpha=0.55)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(int_formatter))
        ax.set_xlim(args.x_min, args.x_max)

        if args.log_income and metric_id != "survival":
            ax.set_yscale("log")

        for group_name in comparison["groups"]:
            agg = group_data[group_name][metric_id]
            if agg is None:
                continue
            steps = agg["steps"]
            mean = agg["mean"]
            ci = agg["ci"]
            lower = [m - c for m, c in zip(mean, ci)]
            upper = [m + c for m, c in zip(mean, ci)]
            color = comparison["colors"][group_name]
            ax.plot(steps, mean, color=color, linewidth=2.0, label=f"{group_name} (n={agg['n_runs']})")
            ax.fill_between(steps, lower, upper, color=color, alpha=0.18, linewidth=0)

        ax.legend(loc="best", frameon=True)

    fig.suptitle(comparison["title"], y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92, wspace=0.22, hspace=0.32)

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    for fmt in formats:
        out_path = args.out_dir / f"{comparison['name']}.{fmt}"
        fig.savefig(out_path, format=fmt, dpi=args.dpi)
        print(out_path.resolve())
    plt.close(fig)


def write_summary(group_data: Dict[str, Dict[str, Optional[dict]]], args: argparse.Namespace) -> None:
    summary_path = args.out_dir / "comparison_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "metric_id", "n_runs", "first_step", "last_step", "last_step_n", "last_mean", "last_95ci_half_width", "seeds"])
        for group_name, metrics in group_data.items():
            for metric_id in METRIC_ORDER:
                agg = metrics.get(metric_id)
                if agg is None:
                    writer.writerow([group_name, metric_id, 0, "", "", "", "", "", ""])
                    continue
                writer.writerow(
                    [
                        group_name,
                        metric_id,
                        agg["n_runs"],
                        agg["steps"][0],
                        agg["steps"][-1],
                        agg["n_by_step"][-1],
                        agg["mean"][-1],
                        agg["ci"][-1],
                        ",".join(agg["seeds"]),
                    ]
                )
    print(summary_path.resolve())


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.formats)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    group_data = build_group_data(args)

    setup_style()
    for comparison in COMPARISONS:
        plot_comparison(comparison, group_data, args)
    write_summary(group_data, args)


if __name__ == "__main__":
    main()
