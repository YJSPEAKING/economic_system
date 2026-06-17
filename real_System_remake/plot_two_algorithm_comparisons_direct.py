from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from plot_two_algorithm_comparisons_matplotlib import GROUPS


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "analysis_plots" / "paper_algorithm_comparisons_matplotlib"
RAW_DIR = OUT_DIR / "raw_metrics"

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

GROUP_DIRS = {
    "TD3": RAW_DIR / "TD3",
    "GAIL+TD3": RAW_DIR / "GAIL_plus_TD3",
    "GAIL+TD3+Transformer": RAW_DIR / "GAIL_plus_TD3_plus_Transformer",
}

COMPARISONS = [
    {
        "name": "td3_vs_gail_td3",
        "title": "TD3 vs GAIL+TD3",
        "groups": ["TD3", "GAIL+TD3"],
        "colors": {"TD3": (31, 119, 180), "GAIL+TD3": (255, 127, 14)},
    },
    {
        "name": "transformer_gail_td3_vs_gail_td3",
        "title": "GAIL+TD3+Transformer vs GAIL+TD3",
        "groups": ["GAIL+TD3", "GAIL+TD3+Transformer"],
        "colors": {"GAIL+TD3": (255, 127, 14), "GAIL+TD3+Transformer": (44, 160, 44)},
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

FIG_WIDTH = 2008
FIG_HEIGHT = 1500
PLOT_RECTS = [
    (70, 105, 980, 700),
    (1030, 105, 1940, 700),
    (70, 810, 980, 1405),
    (1030, 810, 1940, 1405),
]


def find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_series(raw_dir: Path, run_names: Optional[Sequence[str]] = None) -> Dict[str, List[dict]]:
    by_metric: Dict[str, List[dict]] = {metric_id: [] for metric_id in METRIC_ORDER}
    if run_names is None:
        csv_paths = sorted(raw_dir.glob("*_metrics.csv"))
    else:
        csv_paths = [raw_dir / f"{run_name}_metrics.csv" for run_name in run_names]
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        run = ""
        seed = ""
        values_by_metric: Dict[str, Dict[int, float]] = {metric_id: {} for metric_id in METRIC_ORDER}
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                metric_id = row["metric_id"]
                if metric_id not in values_by_metric:
                    continue
                run = row["run"]
                seed = row["seed"]
                values_by_metric[metric_id][int(row["step"])] = float(row["value"])
        for metric_id, values in values_by_metric.items():
            if values:
                by_metric[metric_id].append({"run": run, "seed": seed, "values": values})
    return by_metric


def aggregate(series: Sequence[dict]) -> Optional[dict]:
    if not series:
        return None
    steps = sorted(set.intersection(*(set(item["values"].keys()) for item in series)))
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
            t_value = T_CRITICAL_95.get(n - 1, 1.96)
            cis.append(t_value * math.sqrt(variance) / math.sqrt(n))
        ns.append(n)
    return {
        "steps": steps,
        "mean": means,
        "ci": cis,
        "n_by_step": ns,
        "n_runs": len(series),
        "seeds": [item["seed"] for item in series],
    }


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def nice_number(value: float) -> str:
    return f"{int(round(value))}"


def build_y_ticks(metric_id: str, values: Sequence[float]) -> List[int]:
    data_max = max(values)
    if metric_id == "survival":
        return [0, 20, 40, 60, 80, 100]

    tick_step = 50000 if metric_id in ("production", "consumption") else 5000
    top = int(math.ceil(max(data_max, 0) / tick_step) * tick_step)
    if top == 0:
        top = tick_step
    return list(range(0, top + tick_step, tick_step))


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Tuple[int, int, int],
    width: int = 1,
    dash: int = 8,
    gap: int = 6,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        segment_end = min(pos + dash, length)
        draw.line(
            (
                x1 + dx * pos,
                y1 + dy * pos,
                x1 + dx * segment_end,
                y1 + dy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        pos += dash + gap


def draw_plot(
    image: Image.Image,
    rect: Tuple[int, int, int, int],
    metric_id: str,
    group_data: Dict[str, Optional[dict]],
    colors: Dict[str, Tuple[int, int, int]],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = rect
    title = METRIC_TITLES[metric_id]
    title_w, _ = text_size(draw, title, font)
    draw.text(((left + right - title_w) / 2, top), title, fill=(25, 25, 25), font=font)

    plot_left = left + 112
    plot_top = top + 66
    plot_right = right - 32
    plot_bottom = bottom - 76

    all_steps: List[int] = []
    y_values: List[float] = []
    for agg in group_data.values():
        if agg is None:
            continue
        all_steps.extend(agg["steps"])
        y_values.extend([m - c for m, c in zip(agg["mean"], agg["ci"])])
        y_values.extend([m + c for m, c in zip(agg["mean"], agg["ci"])])
    if not all_steps or not y_values:
        draw.text((plot_left, plot_top), "No data", fill=(80, 80, 80), font=font)
        return

    x_min = 1
    x_max = 60
    y_ticks = build_y_ticks(metric_id, y_values)
    y_min = y_ticks[0]
    y_max = y_ticks[-1]

    def map_x(x: float) -> float:
        return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def map_y(y: float) -> float:
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    grid_color = (205, 205, 205)
    axis_color = (0, 0, 0)
    text_color = (0, 0, 0)

    for tick in range(6):
        ratio = tick / 5
        x = plot_left + ratio * (plot_right - plot_left)
        draw_dashed_line(draw, (x, plot_top), (x, plot_bottom), fill=grid_color, width=1)
        x_value = x_min + ratio * (x_max - x_min)
        draw.text((x - 12, plot_bottom + 10), nice_number(x_value), fill=text_color, font=small_font)

    for y_value in y_ticks:
        y = map_y(y_value)
        draw_dashed_line(draw, (plot_left, y), (plot_right, y), fill=grid_color, width=1)
        y_label = nice_number(y_value)
        y_w, y_h = text_size(draw, y_label, small_font)
        draw.text((plot_left - y_w - 8, y - y_h / 2), y_label, fill=text_color, font=small_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)
    xlabel = "Evaluation Step"
    xlabel_w, _ = text_size(draw, xlabel, small_font)
    draw.text(((plot_left + plot_right - xlabel_w) / 2, bottom - 32), xlabel, fill=text_color, font=small_font)
    draw.text((left + 6, plot_top - 30), Y_LABELS[metric_id], fill=text_color, font=small_font)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    legend_x = plot_right - 260
    legend_y = plot_bottom - 70

    for index, (group_name, agg) in enumerate(group_data.items()):
        if agg is None:
            continue
        color = colors[group_name]
        upper = [(map_x(step), map_y(mean + ci)) for step, mean, ci in zip(agg["steps"], agg["mean"], agg["ci"])]
        lower = [(map_x(step), map_y(mean - ci)) for step, mean, ci in zip(agg["steps"], agg["mean"], agg["ci"])]
        overlay_draw.polygon(upper + list(reversed(lower)), fill=(*color, 42))
        points = [(map_x(step), map_y(mean)) for step, mean in zip(agg["steps"], agg["mean"])]
        image.alpha_composite(overlay)
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        y0 = legend_y + index * 30
        draw.line((legend_x, y0 + 12, legend_x + 34, y0 + 12), fill=color, width=3)
        draw.text((legend_x + 44, y0), f"{group_name} (n={agg['n_runs']})", fill=text_color, font=small_font)


def write_summary(group_data: Dict[str, Dict[str, Optional[dict]]]) -> None:
    with (OUT_DIR / "direct_comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "metric_id", "n_runs", "first_step", "last_step", "last_step_n", "last_mean", "last_95ci_half_width", "seeds"])
        for group_name, metrics in group_data.items():
            for metric_id in METRIC_ORDER:
                agg = metrics.get(metric_id)
                if agg is None:
                    writer.writerow([group_name, metric_id, 0, "", "", "", "", "", ""])
                    continue
                writer.writerow([
                    group_name,
                    metric_id,
                    agg["n_runs"],
                    agg["steps"][0],
                    agg["steps"][-1],
                    agg["n_by_step"][-1],
                    agg["mean"][-1],
                    agg["ci"][-1],
                    ",".join(agg["seeds"]),
                ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_data = {
        group: load_series(path, GROUPS.get(group))
        for group, path in GROUP_DIRS.items()
    }
    group_data = {
        group: {metric_id: aggregate(raw_data[group][metric_id]) for metric_id in METRIC_ORDER}
        for group in GROUP_DIRS
    }

    font = find_font(32)
    small_font = find_font(26)
    title_font = find_font(38)

    for comparison in COMPARISONS:
        image = Image.new("RGBA", (FIG_WIDTH, FIG_HEIGHT), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        title = comparison["title"]
        title_w, _ = text_size(draw, title, title_font)
        draw.text(((FIG_WIDTH - title_w) / 2, 32), title, fill=(0, 0, 0), font=title_font)
        for metric_id, rect in zip(METRIC_ORDER, PLOT_RECTS):
            draw_plot(
                image,
                rect,
                metric_id,
                {group: group_data[group][metric_id] for group in comparison["groups"]},
                comparison["colors"],
                font,
                small_font,
            )
        out_path = OUT_DIR / f"{comparison['name']}.png"
        image.convert("RGB").save(out_path, quality=95, dpi=(300, 300))
        print(out_path)

    write_summary(group_data)


if __name__ == "__main__":
    main()
