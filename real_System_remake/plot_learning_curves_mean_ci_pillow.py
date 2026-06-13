from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


METRIC_ORDER = ["survival", "production", "consumption", "bank"]

METRIC_TITLES = {
    "survival": "System Survival Days",
    "production": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u751f\u4ea7\u4f01\u4e1a",
    "consumption": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u6d88\u8d39\u4f01\u4e1a",
    "bank": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u94f6\u884c",
}

Y_LABELS = {
    "survival": "Survival days",
    "production": "Income",
    "consumption": "Income",
    "bank": "Income",
}

COLORS = {
    "survival": (31, 119, 180),
    "production": (44, 160, 44),
    "consumption": (255, 127, 14),
    "bank": (148, 103, 189),
}

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
    default_out = base / "analysis_plots" / "v1_21_8seeds_mean_ci"
    parser = argparse.ArgumentParser(description="Draw mean and 95% CI learning curves with Pillow.")
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--align", choices=["common", "union"], default="common")
    parser.add_argument("--ci", choices=["t", "normal"], default="t")
    return parser.parse_args()


def find_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_series(raw_dir: Path) -> Dict[str, List[dict]]:
    by_metric: Dict[str, List[dict]] = {metric: [] for metric in METRIC_ORDER}
    for csv_path in sorted(raw_dir.glob("*_metrics.csv")):
        run_name = ""
        seed = ""
        values_by_metric: Dict[str, Dict[int, float]] = {metric: {} for metric in METRIC_ORDER}
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


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def nice_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3g}"


def draw_plot(
    image: Image.Image,
    rect: Tuple[int, int, int, int],
    metric_id: str,
    agg: Optional[dict],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = rect
    title = METRIC_TITLES[metric_id]
    title_w, _ = text_size(draw, title, font)
    draw.text(((left + right - title_w) / 2, top), title, fill=(25, 25, 25), font=font)

    plot_left = left + 88
    plot_top = top + 55
    plot_right = right - 28
    plot_bottom = bottom - 58

    if agg is None:
        msg = "Metric not found"
        msg_w, msg_h = text_size(draw, msg, font)
        draw.text(((left + right - msg_w) / 2, (top + bottom - msg_h) / 2), msg, fill=(80, 80, 80), font=font)
        return

    steps = agg["steps"]
    mean = agg["mean"]
    ci = agg["ci"]
    x_min = min(steps)
    x_max = max(steps)
    y_values = [m - c for m, c in zip(mean, ci)] + [m + c for m, c in zip(mean, ci)]
    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.1
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def map_x(x: float) -> float:
        if x_max == x_min:
            return (plot_left + plot_right) / 2
        return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def map_y(y: float) -> float:
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    grid_color = (232, 232, 232)
    axis_color = (170, 170, 170)
    text_color = (80, 80, 80)

    for tick in range(6):
        ratio = tick / 5
        x = plot_left + ratio * (plot_right - plot_left)
        y = plot_bottom - ratio * (plot_bottom - plot_top)
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        x_val = x_min + ratio * (x_max - x_min)
        y_val = y_min + ratio * (y_max - y_min)
        draw.text((x - 10, plot_bottom + 8), nice_number(x_val), fill=text_color, font=small_font)
        y_label = nice_number(y_val)
        y_w, y_h = text_size(draw, y_label, small_font)
        draw.text((plot_left - y_w - 8, y - y_h / 2), y_label, fill=text_color, font=small_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)

    xlabel = "Evaluation step (per 100 episodes)"
    xlabel_w, _ = text_size(draw, xlabel, small_font)
    draw.text(((plot_left + plot_right - xlabel_w) / 2, bottom - 28), xlabel, fill=text_color, font=small_font)
    ylabel = Y_LABELS[metric_id]
    draw.text((left + 8, plot_top - 25), ylabel, fill=text_color, font=small_font)

    color = COLORS[metric_id]
    upper = [(map_x(step), map_y(m + c)) for step, m, c in zip(steps, mean, ci)]
    lower = [(map_x(step), map_y(m - c)) for step, m, c in zip(steps, mean, ci)]
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(upper + list(reversed(lower)), fill=(*color, 42))
    image.alpha_composite(overlay)

    points = [(map_x(step), map_y(value)) for step, value in zip(steps, mean)]
    if len(points) >= 2:
        draw.line(points, fill=color, width=4, joint="curve")
    for point in points:
        x, y = point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    legend = f"Mean (n={agg['n_runs']})   95% CI"
    draw.text((plot_right - 190, plot_top + 8), legend, fill=text_color, font=small_font)


def save_summary(out_dir: Path, aggregated: Dict[str, Optional[dict]]) -> None:
    summary_path = out_dir / "learning_curves_mean_ci_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric_id", "n_runs", "first_step", "last_step", "last_step_n", "last_mean", "last_95ci_half_width", "seeds"])
        for metric_id in METRIC_ORDER:
            agg = aggregated.get(metric_id)
            if agg is None:
                writer.writerow([metric_id, 0, "", "", "", "", "", ""])
                continue
            writer.writerow([
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
    args = parse_args()
    out_dir = args.out_dir
    raw_dir = args.raw_dir or out_dir / "raw_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_metric = load_series(raw_dir)
    aggregated = {metric_id: aggregate(by_metric[metric_id], args.align, args.ci) for metric_id in METRIC_ORDER}

    image = Image.new("RGBA", (1800, 1200), (255, 255, 255, 255))
    font = find_font(28)
    small_font = find_font(20)
    title_font = find_font(32)
    draw = ImageDraw.Draw(image)
    main_title = "GAIL+TD3 v1.21: Mean Learning Curves with 95% CI"
    title_w, _ = text_size(draw, main_title, title_font)
    draw.text(((1800 - title_w) / 2, 28), main_title, fill=(25, 25, 25), font=title_font)

    rects = [
        (55, 90, 875, 585),
        (925, 90, 1745, 585),
        (55, 650, 875, 1145),
        (925, 650, 1745, 1145),
    ]
    for metric_id, rect in zip(METRIC_ORDER, rects):
        draw_plot(image, rect, metric_id, aggregated[metric_id], font, small_font)

    image.convert("RGB").save(out_dir / "learning_curves_mean_ci_4plots.png", quality=95)

    for metric_id in METRIC_ORDER:
        single = Image.new("RGBA", (900, 600), (255, 255, 255, 255))
        draw_plot(single, (20, 20, 880, 580), metric_id, aggregated[metric_id], font, small_font)
        single.convert("RGB").save(out_dir / f"{metric_id}_mean_ci.png", quality=95)

    save_summary(out_dir, aggregated)
    print(out_dir / "learning_curves_mean_ci_4plots.png")


if __name__ == "__main__":
    main()
