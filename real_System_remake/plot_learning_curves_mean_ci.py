"""
Plot mean learning curves with 95% confidence intervals from SwanLab runs.

Default behavior:
1. Read the current experiment notes from Environment.py.
2. Find the latest contiguous SwanLab run block whose notes match that value.
3. Export four metrics from backup.swanlab into cached CSV files.
4. Draw one 2x2 summary figure and four separate figures.

Example:
    python plot_learning_curves_mean_ci.py

Use explicit run names:
    python plot_learning_curves_mean_ci.py --runs run-xxx run-yyy run-zzz run-aaa

Use an explicit notes string:
    python plot_learning_curves_mean_ci.py --notes "online GAIL+TD3 v1.20, smoother daily bank reward and stronger terminal survival penalty"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METRIC_KEYS = {
    "survival": "\u6bcf\u767e\u56de\u5408/\u5b58\u6d3b\u5929\u6570",
    "production": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u751f\u4ea7\u4f01\u4e1a1",
    "consumption": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u6d88\u8d39\u4f01\u4e1a1",
    "bank": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u94f6\u884c",
}

METRIC_TITLES = {
    "survival": "System Survival Days",
    "production": "每百回合/累计利润/生产企业",
    "consumption": "每百回合/累计利润/消费企业",
    "bank": "每百回合/累计奖励/银行",
}

Y_LABELS = {
    "survival": "Survival days",
    "production": "Comprehensive income",
    "consumption": "Comprehensive income",
    "bank": "Interest income",
}

COLORS = {
    "survival": "#1f77b4",
    "production": "#2ca02c",
    "consumption": "#ff7f0e",
    "bank": "#9467bd",
}

METRIC_ORDER = ["survival", "production", "consumption", "bank"]

# Two-sided 95% t critical values. The fallback is normal 1.96 for larger n.
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
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot mean learning curves with 95% confidence intervals.")
    parser.add_argument("--swanlog-dir", type=Path, default=script_dir / "swanlog")
    parser.add_argument("--environment", type=Path, default=script_dir / "Environment.py")
    parser.add_argument("--out-dir", type=Path, default=script_dir / "analysis_plots" / "learning_curves_mean_ci")
    parser.add_argument("--runs", nargs="*", default=None, help="Run names or run directories. If omitted, runs are selected by notes.")
    parser.add_argument("--notes", default="current", help='Notes filter. Use "current" to read Environment.py.')
    parser.add_argument("--max-scan-runs", type=int, default=120, help="Maximum recent runs to inspect when selecting by notes.")
    parser.add_argument("--all-matching", action="store_true", help="Use all matching runs instead of the latest contiguous matching block.")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-read backup.swanlab even when cached CSV exists.")
    parser.add_argument("--align", choices=["common", "union"], default="common", help="Step alignment. common uses only steps present in every run.")
    parser.add_argument("--ci", choices=["t", "normal"], default="t", help="Use t interval or normal 1.96 interval.")
    return parser.parse_args()


def read_current_notes(environment_path: Path) -> Optional[str]:
    if not environment_path.exists():
        return None
    text = environment_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"notes\s*=\s*[\"']([^\"']+)[\"']", text)
    return match.group(1) if match else None


def import_swanlab_reader():
    from swanlab.data.porter.datastore import DataStore
    from swanlab.proto.v0 import BaseModel

    return DataStore, BaseModel


def scan_run_description(run_dir: Path, max_records: int = 5000) -> Optional[str]:
    backup_path = run_dir / "backup.swanlab"
    if not backup_path.exists():
        return None

    DataStore, BaseModel = import_swanlab_reader()
    datastore = DataStore()
    datastore.open_for_scan(str(backup_path))
    description = None
    try:
        for _ in range(max_records):
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
                break
            config = data.get("config")
            if isinstance(config, dict):
                config_desc = config.get("description") or config.get("notes")
                if isinstance(config_desc, str) and config_desc:
                    description = config_desc
                    break
    finally:
        try:
            datastore.close()
        except Exception:
            pass
    return description


def seed_from_config(run_dir: Path) -> str:
    config_path = run_dir / "files" / "config.yaml"
    if not config_path.exists():
        return ""
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"random_seed:\s*([0-9]+)", text)
    return match.group(1) if match else ""


def git_from_meta(run_dir: Path) -> str:
    meta_path = run_dir / "files" / "swanlab-metadata.json"
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    return json.dumps(meta.get("git_info"), ensure_ascii=False)


def resolve_run_dir(swanlog_dir: Path, run_arg: str) -> Path:
    path = Path(run_arg)
    if path.exists():
        return path
    return swanlog_dir / run_arg


def select_runs(args: argparse.Namespace) -> Tuple[List[Path], str]:
    if args.runs:
        run_dirs = [resolve_run_dir(args.swanlog_dir, run_arg) for run_arg in args.runs]
        missing = [str(path) for path in run_dirs if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Run directory not found: {missing}")
        return run_dirs, "manual run list"

    target_notes = read_current_notes(args.environment) if args.notes == "current" else args.notes
    if not target_notes:
        raise ValueError("No notes filter was provided, and Environment.py notes could not be read.")

    run_dirs = sorted(
        [path for path in args.swanlog_dir.iterdir() if path.is_dir() and path.name.startswith("run-")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    scanned = run_dirs[: args.max_scan_runs]
    matches = []
    latest_block = []
    block_started = False

    for run_dir in scanned:
        description = scan_run_description(run_dir)
        is_match = description == target_notes
        if is_match:
            matches.append(run_dir)
            if not block_started:
                block_started = True
            latest_block.append(run_dir)
        elif block_started:
            break

    selected = matches if args.all_matching else latest_block
    if not selected:
        raise ValueError(f"No SwanLab runs matched notes: {target_notes}")
    return selected, target_notes


def export_metrics(run_dir: Path, cache_dir: Path, refresh_cache: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{run_dir.name}_metrics.csv"
    meta_path = cache_dir / f"{run_dir.name}_meta.json"
    if csv_path.exists() and meta_path.exists() and not refresh_cache:
        return csv_path

    DataStore, BaseModel = import_swanlab_reader()
    backup_path = run_dir / "backup.swanlab"
    metrics: Dict[str, Dict[int, float]] = {metric_id: {} for metric_id in METRIC_KEYS}
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
            if key not in METRIC_KEYS.values():
                continue
            metric = data.get("metric")
            value = metric.get("data") if isinstance(metric, dict) else metric
            try:
                step = int(data.get("step"))
                value = float(value)
            except Exception:
                continue

            for metric_id, metric_key in METRIC_KEYS.items():
                if key == metric_key:
                    metrics[metric_id][step] = value
                    break
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

    meta = {
        "run": run_dir.name,
        "seed": seed,
        "description": description,
        "git_info": git_from_meta(run_dir),
        "metric_counts": {metric_id: len(values) for metric_id, values in metrics.items()},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path


def load_cached_series(csv_paths: Iterable[Path]) -> Dict[str, List[dict]]:
    by_metric: Dict[str, List[dict]] = {metric_id: [] for metric_id in METRIC_ORDER}
    for csv_path in csv_paths:
        values_by_metric: Dict[str, Dict[int, float]] = {metric_id: {} for metric_id in METRIC_ORDER}
        run_name = ""
        seed = ""
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

    means = []
    cis = []
    ns = []
    for step in steps:
        values = np.array([item["values"][step] for item in series if step in item["values"]], dtype=float)
        n = len(values)
        means.append(float(values.mean()))
        if n <= 1:
            cis.append(0.0)
        else:
            cis.append(float(ci_multiplier(n, ci_method) * values.std(ddof=1) / math.sqrt(n)))
        ns.append(n)

    return {
        "steps": np.array(steps),
        "mean": np.array(means),
        "ci": np.array(cis),
        "n_by_step": np.array(ns),
        "n_runs": len(series),
        "seeds": [item["seed"] for item in series],
    }


def write_summary(out_dir: Path, aggregated: Dict[str, Optional[dict]], run_dirs: Sequence[Path], notes: str) -> None:
    source_path = out_dir / "source_runs.txt"
    with source_path.open("w", encoding="utf-8") as file:
        file.write(f"notes_or_selection={notes}\n")
        for run_dir in run_dirs:
            file.write(f"{run_dir.name}\tseed={seed_from_config(run_dir)}\tpath={run_dir}\n")

    summary_path = out_dir / "learning_curves_mean_ci_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "metric_id",
                "n_runs",
                "first_step",
                "last_step",
                "last_step_n",
                "last_mean",
                "last_95ci_half_width",
                "seeds",
            ]
        )
        for metric_id in METRIC_ORDER:
            agg = aggregated.get(metric_id)
            if agg is None:
                writer.writerow([metric_id, 0, "", "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    metric_id,
                    agg["n_runs"],
                    int(agg["steps"][0]),
                    int(agg["steps"][-1]),
                    int(agg["n_by_step"][-1]),
                    float(agg["mean"][-1]),
                    float(agg["ci"][-1]),
                    ",".join(str(seed) for seed in agg["seeds"]),
                ]
            )


def plot_curves(out_dir: Path, aggregated: Dict[str, Optional[dict]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for ax, metric_id, letter in zip(axes.ravel(), METRIC_ORDER, letters):
        agg = aggregated.get(metric_id)
        ax.set_title(f"{letter} {METRIC_TITLES[metric_id]}")
        if agg is None:
            ax.text(0.5, 0.5, "Metric not found", transform=ax.transAxes, ha="center", va="center")
        else:
            color = COLORS[metric_id]
            ax.plot(agg["steps"], agg["mean"], color=color, linewidth=2.0, label=f"Mean (n={agg['n_runs']})")
            ax.fill_between(
                agg["steps"],
                agg["mean"] - agg["ci"],
                agg["mean"] + agg["ci"],
                color=color,
                alpha=0.18,
                linewidth=0,
                label="95% CI",
            )
            ax.legend(loc="best", frameon=True)
        ax.set_xlabel("Evaluation step (per 100 episodes)")
        ax.set_ylabel(Y_LABELS[metric_id])
        ax.grid(True, alpha=0.28)

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.savefig(out_dir / "learning_curves_mean_ci_4plots.png", bbox_inches="tight")
    plt.close(fig)

    for metric_id in METRIC_ORDER:
        agg = aggregated.get(metric_id)
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        ax.set_title(METRIC_TITLES[metric_id])
        if agg is None:
            ax.text(0.5, 0.5, "Metric not found", transform=ax.transAxes, ha="center", va="center")
        else:
            color = COLORS[metric_id]
            ax.plot(agg["steps"], agg["mean"], color=color, linewidth=2.1, label=f"Mean (n={agg['n_runs']})")
            ax.fill_between(
                agg["steps"],
                agg["mean"] - agg["ci"],
                agg["mean"] + agg["ci"],
                color=color,
                alpha=0.2,
                linewidth=0,
                label="95% CI",
            )
            ax.legend(loc="best")
        ax.set_xlabel("Evaluation step (per 100 episodes)")
        ax.set_ylabel(Y_LABELS[metric_id])
        ax.grid(True, alpha=0.28)
        fig.savefig(out_dir / f"{metric_id}_mean_ci.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dirs, selection_label = select_runs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "raw_metrics"

    csv_paths = [export_metrics(run_dir, cache_dir, args.refresh_cache) for run_dir in run_dirs]
    by_metric = load_cached_series(csv_paths)
    aggregated = {metric_id: aggregate(by_metric[metric_id], args.align, args.ci) for metric_id in METRIC_ORDER}

    title = f"Learning Curves: mean with 95% CI ({selection_label})"
    plot_curves(args.out_dir, aggregated, title)
    write_summary(args.out_dir, aggregated, run_dirs, selection_label)

    print(f"Selected runs: {len(run_dirs)}")
    for run_dir in run_dirs:
        print(f"  {run_dir.name} seed={seed_from_config(run_dir)}")
    print(f"Output directory: {args.out_dir.resolve()}")
    print("Main figure: learning_curves_mean_ci_4plots.png")


if __name__ == "__main__":
    main()
