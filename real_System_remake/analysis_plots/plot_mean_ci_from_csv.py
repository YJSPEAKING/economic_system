from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_TITLES = {
    "survival": "System Survival Days",
    "production": "Production Agent Cumulative Reward",
    "consumption": "Consumption Agent Cumulative Reward",
    "bank": "Bank Cumulative Reward / Profit Proxy",
}

YLABELS = {
    "survival": "Survival days",
    "production": "Cumulative reward",
    "consumption": "Cumulative reward",
    "bank": "Cumulative reward",
}

COLORS = {
    "survival": "#1f77b4",
    "production": "#2ca02c",
    "consumption": "#ff7f0e",
    "bank": "#9467bd",
}


def load_series(raw_dir: Path):
    by_metric = {metric_id: [] for metric_id in METRIC_TITLES}
    source_runs = []

    for csv_path in sorted(raw_dir.glob("*_metrics.csv")):
        by_run_metric = {metric_id: {} for metric_id in METRIC_TITLES}
        run_name = ""
        seed = ""
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                metric_id = row["metric_id"]
                if metric_id not in by_run_metric:
                    continue
                run_name = row["run"]
                seed = row["seed"]
                by_run_metric[metric_id][int(row["step"])] = float(row["value"])

        source_runs.append((run_name, seed, csv_path.name))
        for metric_id, values in by_run_metric.items():
            if values:
                by_metric[metric_id].append({"run": run_name, "seed": seed, "values": values})

    return by_metric, source_runs


def aggregate(series):
    if not series:
        return None
    common_steps = sorted(set.intersection(*(set(item["values"].keys()) for item in series)))
    if not common_steps:
        return None
    data = np.array([[item["values"][step] for step in common_steps] for item in series], dtype=float)
    mean = data.mean(axis=0)
    ci = 1.96 * data.std(axis=0, ddof=1) / math.sqrt(data.shape[0]) if data.shape[0] > 1 else np.zeros_like(mean)
    return {
        "steps": np.array(common_steps),
        "mean": mean,
        "ci": ci,
        "n": data.shape[0],
        "seeds": [item["seed"] for item in series],
        "last_values": data[:, -1],
    }


def main():
    project_dir = Path.cwd() / "real_System_remake"
    out_dir = project_dir / "analysis_plots" / "v1_20_mean_ci"
    raw_dir = out_dir / "raw_metrics"
    by_metric, source_runs = load_series(raw_dir)
    aggregated = {metric_id: aggregate(series) for metric_id, series in by_metric.items()}

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    axes = axes.ravel()
    letters = ["(a)", "(b)", "(c)", "(d)"]
    metric_order = ["survival", "production", "consumption", "bank"]

    for ax, metric_id, letter in zip(axes, metric_order, letters):
        agg = aggregated[metric_id]
        ax.set_title(f"{letter} {METRIC_TITLES[metric_id]}")
        if agg is None:
            ax.text(0.5, 0.5, "Metric not found", transform=ax.transAxes, ha="center", va="center")
        else:
            color = COLORS[metric_id]
            ax.plot(agg["steps"], agg["mean"], color=color, linewidth=2.0, label=f"Mean (n={agg['n']})")
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
        ax.set_ylabel(YLABELS[metric_id])
        ax.grid(True, alpha=0.28)

    fig.suptitle("GAIL+TD3 v1.20 Results: mean with 95% confidence interval", y=1.02, fontsize=13)
    combined_path = out_dir / "v1_20_mean_ci_4plots.png"
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)

    for metric_id in metric_order:
        agg = aggregated[metric_id]
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        ax.set_title(METRIC_TITLES[metric_id])
        if agg is None:
            ax.text(0.5, 0.5, "Metric not found", transform=ax.transAxes, ha="center", va="center")
        else:
            color = COLORS[metric_id]
            ax.plot(agg["steps"], agg["mean"], color=color, linewidth=2.1, label=f"Mean (n={agg['n']})")
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
        ax.set_ylabel(YLABELS[metric_id])
        ax.grid(True, alpha=0.28)
        fig.savefig(out_dir / f"{metric_id}_mean_ci.png", bbox_inches="tight")
        plt.close(fig)

    summary_path = out_dir / "v1_20_mean_ci_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "metric_id",
                "n_runs",
                "n_common_steps",
                "first_step",
                "last_step",
                "last_mean",
                "last_95ci_half_width",
                "seeds",
                "last_values_by_seed",
            ]
        )
        for metric_id in metric_order:
            agg = aggregated[metric_id]
            if agg is None:
                writer.writerow([metric_id, 0, 0, "", "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    metric_id,
                    agg["n"],
                    len(agg["steps"]),
                    int(agg["steps"][0]),
                    int(agg["steps"][-1]),
                    float(agg["mean"][-1]),
                    float(agg["ci"][-1]),
                    ",".join(str(seed) for seed in agg["seeds"]),
                    ",".join(f"{value:.6f}" for value in agg["last_values"]),
                ]
            )

    source_path = out_dir / "source_runs.txt"
    with source_path.open("w", encoding="utf-8") as file:
        for run_name, seed, csv_name in source_runs:
            file.write(f"{run_name}\tseed={seed}\tcsv={csv_name}\n")

    print(combined_path.resolve())
    print(summary_path.resolve())
    print(source_path.resolve())


if __name__ == "__main__":
    main()
