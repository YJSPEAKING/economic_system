# -*- coding: utf-8 -*-
"""Plot algorithm comparisons with DSCR metrics from local swanlab backups.

The script reads backup.swanlab files directly and generates three SEM-shaded
comparison figures:
1. TD3 vs GAIL+TD3
2. Trans.+GAIL+TD3 vs GAIL+TD3
3. TD3 vs GAIL+TD3 vs Trans.+GAIL+TD3
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator


ROOT = Path(__file__).resolve().parent
SWANLOG_DIR = ROOT / "swanlog"
OUT_DIR = ROOT / "analysis_plots" / "dscr_algorithm_comparisons_sem"
STATS_DIR = OUT_DIR / "stats_csv"
MAX_STEP = 60


GROUPS = {
    "TD3": [
        "run-20260618_211323-8zf0k46bcjbni1vjrbfys",
        "run-20260618_230355-v1k5fck6mzbg2kjictnlc",
        "run-20260619_005703-4b94z2dtxcp4syckcmle1",
        "run-20260619_023417-mwsqdqkj0m8eax76pso9p",
        "run-20260619_040947-x5upzsw6v5ghcl3w3z077",
        "run-20260619_054347-kbfbo3b86dvmmd74o1m6m",
        "run-20260619_064516-l317net6sx2b973jmu76e",
        "run-20260619_083216-z8r33j2vxpww35ip5v1wd",
    ],
    "GAIL+TD3": [
        "run-20260618_203534-64q12my1ctp79icf5147x",
        "run-20260618_231754-jn2btqpxneq994ue8z223",
        "run-20260619_012114-3oxqc5rogbsp7l10y2q3r",
        "run-20260619_033517-avf5jzkd6p3z024gw4uvc",
        "run-20260619_053345-ihojjvlo2utnu2nssuk3c",
        "run-20260619_073444-ktguwhitx3svowm9n1ezd",
        "run-20260619_105341-348k186sfnohcuz4bommc",
        "run-20260619_140303-sa9gtcyxsazuhe8ixd2aa",
    ],
    "Trans.+GAIL+TD3": [
        "run-20260619_164141-wzfka5d4w6l7g6a4p79dc",
        "run-20260619_205135-3o5c20j57k5ydicjqgp0c",
        "run-20260620_003508-0b413tloayakr3iw5anxw",
        "run-20260620_044710-mj8r3zkl2tzd6t15iiofi",
        "run-20260620_082257-olsvq38qota1rsd8fkdbg",
        "run-20260620_114030-m9iyp051e087xm6gx3p9j",
        "run-20260620_160510-e5h5bli8x4w629b2bdarm",
        "run-20260620_201216-2wa0tnubzincke6uj2cnk",
    ],
}


METRICS = [
    {
        "key": "每百回合/存活天数",
        "title": "(a) 每百回合/存活天数",
        "ylabel": "Days",
        "kind": "survival",
    },
    {
        "key": "每百回合/累计收益/生产企业",
        "title": "(b) 每百回合/累计收益/生产企业",
        "ylabel": "Income",
        "kind": "enterprise_income",
    },
    {
        "key": "每百回合/累计收益/消费企业",
        "title": "(c) 每百回合/累计收益/消费企业",
        "ylabel": "Income",
        "kind": "enterprise_income",
    },
    {
        "key": "每百回合/累计收益/银行",
        "title": "(d) 每百回合/累计收益/银行",
        "ylabel": "Income",
        "kind": "bank_income",
    },
    {
        "key": "每百回合/偿债能力/生产企业",
        "title": "(e) 每百回合/偿债能力/生产企业",
        "ylabel": "DSCR",
        "kind": "dscr",
    },
    {
        "key": "每百回合/偿债能力中位数/生产企业",
        "title": "(f) 每百回合/偿债能力中位数/生产企业",
        "ylabel": "DSCR",
        "kind": "dscr",
    },
    {
        "key": "每百回合/偿债能力/消费企业",
        "title": "(g) 每百回合/偿债能力/消费企业",
        "ylabel": "DSCR",
        "kind": "dscr",
    },
    {
        "key": "每百回合/偿债能力中位数/消费企业",
        "title": "(h) 每百回合/偿债能力中位数/消费企业",
        "ylabel": "DSCR",
        "kind": "dscr",
    },
]


COMPARISONS = [
    (
        "td3_vs_gail_td3_dscr_sem.png",
        "TD3 vs GAIL+TD3: Mean Curves with SEM",
        ["TD3", "GAIL+TD3"],
    ),
    (
        "transformer_gail_td3_vs_gail_td3_dscr_sem.png",
        "Trans.+GAIL+TD3 vs GAIL+TD3: Mean Curves with SEM",
        ["GAIL+TD3", "Trans.+GAIL+TD3"],
    ),
    (
        "td3_gail_td3_transformer_dscr_sem.png",
        "TD3 vs GAIL+TD3 vs Trans.+GAIL+TD3: Mean Curves with SEM",
        ["TD3", "GAIL+TD3", "Trans.+GAIL+TD3"],
    ),
]


COLORS = {
    "TD3": "#1f77b4",
    "GAIL+TD3": "#ff7f0e",
    "Trans.+GAIL+TD3": "#2ca02c",
}


def load_run_scalars(run_name: str) -> dict[str, dict[int, float]]:
    backup_path = SWANLOG_DIR / run_name / "backup.swanlab"
    if not backup_path.exists():
        raise FileNotFoundError(f"Missing swanlab backup: {backup_path}")

    text = backup_path.read_bytes().decode("utf-8", errors="ignore")
    metrics: dict[str, dict[int, float]] = {}
    for metric in METRICS:
        key = metric["key"]
        needle = f'"key": "{key}"'
        search_from = 0
        while True:
            key_pos = text.find(needle, search_from)
            if key_pos < 0:
                break
            search_from = key_pos + len(needle)

            scalar_start = text.rfind('{"model_type": "Scalar"', 0, key_pos)
            if scalar_start < 0:
                continue

            value_marker = '"data": '
            value_pos = text.rfind(value_marker, scalar_start, key_pos)
            step_marker = '"step": '
            step_pos = text.find(step_marker, key_pos, key_pos + 260)
            if value_pos < 0 or step_pos < 0:
                continue

            try:
                value_f = parse_float_at(text, value_pos + len(value_marker))
                step_i = parse_int_at(text, step_pos + len(step_marker))
            except ValueError:
                continue
            if 0 <= step_i <= MAX_STEP:
                metrics.setdefault(key, {})[step_i] = value_f
    return metrics


def parse_float_at(text: str, start: int) -> float:
    while start < len(text) and text[start].isspace():
        start += 1
    end = start
    allowed = set("-+.0123456789eE")
    while end < len(text) and text[end] in allowed:
        end += 1
    if end == start:
        raise ValueError("No float at position")
    return float(text[start:end])


def parse_int_at(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    end = start
    if end < len(text) and text[end] in "+-":
        end += 1
    while end < len(text) and text[end].isdigit():
        end += 1
    if end == start:
        raise ValueError("No int at position")
    return int(text[start:end])


def load_all_runs() -> dict[str, dict[str, dict[str, dict[int, float]]]]:
    all_data: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for group, runs in GROUPS.items():
        all_data[group] = {}
        for run in runs:
            all_data[group][run] = load_run_scalars(run)
    return all_data


def group_metric_matrix(
    all_data: dict[str, dict[str, dict[str, dict[int, float]]]],
    group: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    steps = np.arange(1, MAX_STEP + 1, dtype=int)
    rows = []
    for run in GROUPS[group]:
        series = all_data[group][run].get(metric_key, {})
        rows.append([series.get(int(step), np.nan) for step in steps])
    return steps, np.array(rows, dtype=float)


def summarize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_counts = np.sum(~np.isnan(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(valid_counts), out=np.zeros_like(std), where=valid_counts > 1)
    mean[valid_counts == 0] = np.nan
    sem[valid_counts == 0] = np.nan
    return mean, sem, valid_counts


def nice_upper(value: float, base: int) -> int:
    if not np.isfinite(value) or value <= 0:
        return base
    return int(math.ceil(value / base) * base)


def configure_axis(ax, metric, plotted_values: list[np.ndarray]) -> None:
    ax.set_xlim(0, MAX_STEP)
    ax.set_xlabel("Evaluation Step", labelpad=4)
    ax.set_ylabel(metric["ylabel"], labelpad=22)
    ax.tick_params(direction="out", length=3.5, width=0.9)
    ax.grid(True, linestyle="--", linewidth=0.65, color="#bcbcbc", alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    finite_values = []
    for values in plotted_values:
        finite_values.extend(values[np.isfinite(values)].tolist())
    max_value = max(finite_values) if finite_values else 1.0

    kind = metric["kind"]
    if kind == "survival":
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
    elif kind == "enterprise_income":
        ax.set_ylim(bottom=0, top=nice_upper(max_value * 1.05, 50000))
        ax.yaxis.set_major_locator(MultipleLocator(50000))
    elif kind == "bank_income":
        ax.set_ylim(bottom=0, top=nice_upper(max_value * 1.10, 5000))
        ax.yaxis.set_major_locator(MultipleLocator(5000))
    else:
        upper = nice_upper(max_value * 1.12, 10)
        ax.set_ylim(bottom=0, top=upper)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def save_stats_csv(
    all_data: dict[str, dict[str, dict[str, dict[int, float]]]],
    groups: list[str],
    filename_stem: str,
) -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        path = STATS_DIR / f"{filename_stem}_{safe_filename(metric['key'])}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "metric", "group", "mean", "sem", "n"])
            for group in groups:
                steps, matrix = group_metric_matrix(all_data, group, metric["key"])
                mean, sem, counts = summarize_matrix(matrix)
                for step, mean_v, sem_v, n in zip(steps, mean, sem, counts):
                    if n == 0:
                        continue
                    writer.writerow([step, metric["key"], group, mean_v, sem_v, int(n)])


def safe_filename(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def plot_comparison(
    all_data: dict[str, dict[str, dict[str, dict[int, float]]]],
    filename: str,
    title: str,
    groups: list[str],
) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.6), dpi=300)
    axes = axes.ravel()
    fig.suptitle(title, fontsize=15, y=0.985)

    for ax, metric in zip(axes, METRICS):
        plotted = []
        for group in groups:
            steps, matrix = group_metric_matrix(all_data, group, metric["key"])
            mean, sem, counts = summarize_matrix(matrix)
            color = COLORS[group]
            mask = np.isfinite(mean)
            if not np.any(mask):
                continue
            lower = np.maximum(mean - sem, 0)
            upper = mean + sem
            ax.fill_between(
                steps[mask],
                lower[mask],
                upper[mask],
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                steps[mask],
                mean[mask],
                color=color,
                linewidth=1.6,
                label=group,
                zorder=3,
            )
            plotted.append(upper[mask])

        ax.set_title(metric["title"], fontsize=11, pad=7)
        configure_axis(ax, metric, plotted)
        ax.legend(
            loc="best",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=0.78,
            borderpad=0.35,
            handlelength=2.0,
        )

    fig.text(
        0.5,
        0.018,
        "Shaded area indicates +/-1 SEM; each algorithm uses n=8 seeds.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.075, top=0.92, wspace=0.27, hspace=0.34)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    save_stats_csv(all_data, groups, Path(filename).stem)
    return out_path


def report_missing_metrics(all_data: dict[str, dict[str, dict[str, dict[int, float]]]]) -> list[str]:
    warnings = []
    for group, runs in GROUPS.items():
        for run in runs:
            scalars = all_data[group].get(run, {})
            for metric in METRICS:
                if metric["key"] not in scalars:
                    warnings.append(f"{group}/{run} missing: {metric['key']}")
    return warnings


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    all_data = load_all_runs()
    warnings = report_missing_metrics(all_data)
    if warnings:
        print("Missing metrics:")
        for item in warnings:
            print("  " + item)

    output_paths = []
    for filename, title, groups in COMPARISONS:
        output_paths.append(plot_comparison(all_data, filename, title, groups))

    print("Generated figures:")
    for path in output_paths:
        print(path)
    print(f"Stats CSV directory: {STATS_DIR}")


if __name__ == "__main__":
    main()
