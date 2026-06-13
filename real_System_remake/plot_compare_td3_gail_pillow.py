from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


TD3_RUNS = [
    "run-20260611_000002-n3od24x0neyzw18oirlk6",
    "run-20260611_011610-nsebmrbwgob5dzg68bki1",
    "run-20260611_022829-vh58qcjl9pq5dhjrjzku1",
    "run-20260611_034129-m6wp1sysrveov4tarc29s",
    "run-20260611_045354-oe1vi3kximynewatvt20r",
    "run-20260611_060852-rsw0uy2r11i1b0bd9agvz",
    "run-20260611_072248-k7demplrp6ersto1hykhn",
    "run-20260611_083727-9ovnv1xy4a8eptpoy4tfw",
]

GAIL_RUNS = [
    "run-20260611_191409-z6r9akkau687teps4qlhu",
    "run-20260611_222957-znvy698zfqrz1a81fnbgu",
    "run-20260612_003943-fybyo5u7zhpnzhqxlfgq7",
    "run-20260612_024959-7c3qf4o108bbi8dvqa9v1",
    "run-20260612_074153-0yhjd4spum2e0yedvjbf2",
    "run-20260612_102339-d41qetvzo5js3yvemkmps",
    "run-20260612_180041-333xdgw1vzwlocfrd2js7",
    "run-20260612_235915-cfvkp9kcimzdwmvt30dka",
]

METRIC_KEYS = {
    "survival": "\u6bcf\u767e\u56de\u5408/\u5b58\u6d3b\u5929\u6570",
    "production": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u751f\u4ea7\u4f01\u4e1a",
    "consumption": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u6d88\u8d39\u4f01\u4e1a",
    "bank": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u6536\u76ca/\u94f6\u884c",
}

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

GROUP_STYLES = {
    "TD3": {"color": (31, 119, 180), "alpha": 36},
    "GAIL+TD3": {"color": (255, 127, 14), "alpha": 42},
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
    parser = argparse.ArgumentParser(description="Compare TD3 and GAIL+TD3 learning curves with Pillow.")
    parser.add_argument("--swanlog-dir", type=Path, default=base / "swanlog")
    parser.add_argument("--out-dir", type=Path, default=base / "analysis_plots" / "td3_vs_gail_v1_21_8seeds")
    parser.add_argument("--align", choices=["common", "union"], default="common")
    parser.add_argument("--ci", choices=["t", "normal"], default="t")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


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


def export_metrics(run_dir: Path, cache_dir: Path, refresh_cache: bool) -> Path:
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
    return {"steps": steps, "mean": means, "ci": cis, "n_by_step": ns, "n_runs": len(series), "seeds": [item["seed"] for item in series]}


def find_font(size: int) -> ImageFont.ImageFont:
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


def draw_metric(
    image: Image.Image,
    rect: Tuple[int, int, int, int],
    metric_id: str,
    group_data: Dict[str, Optional[dict]],
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

    all_steps: List[int] = []
    y_values: List[float] = []
    for agg in group_data.values():
        if agg is None:
            continue
        all_steps.extend(agg["steps"])
        y_values.extend([m - c for m, c in zip(agg["mean"], agg["ci"])])
        y_values.extend([m + c for m, c in zip(agg["mean"], agg["ci"])])

    if not all_steps or not y_values:
        msg = "Metric not found"
        msg_w, msg_h = text_size(draw, msg, font)
        draw.text(((left + right - msg_w) / 2, (top + bottom - msg_h) / 2), msg, fill=(80, 80, 80), font=font)
        return

    x_min, x_max = min(all_steps), max(all_steps)
    y_min, y_max = min(y_values), max(y_values)
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
        draw.text((x - 10, plot_bottom + 8), nice_number(x_min + ratio * (x_max - x_min)), fill=text_color, font=small_font)
        y_label = nice_number(y_min + ratio * (y_max - y_min))
        y_w, y_h = text_size(draw, y_label, small_font)
        draw.text((plot_left - y_w - 8, y - y_h / 2), y_label, fill=text_color, font=small_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)
    xlabel = "Evaluation step (per 100 episodes)"
    xlabel_w, _ = text_size(draw, xlabel, small_font)
    draw.text(((plot_left + plot_right - xlabel_w) / 2, bottom - 28), xlabel, fill=text_color, font=small_font)
    draw.text((left + 8, plot_top - 25), Y_LABELS[metric_id], fill=text_color, font=small_font)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    legend_x = plot_right - 240
    legend_y = plot_top + 8
    for idx, (label, agg) in enumerate(group_data.items()):
        style = GROUP_STYLES[label]
        color = style["color"]
        if agg is None:
            continue
        upper = [(map_x(step), map_y(m + c)) for step, m, c in zip(agg["steps"], agg["mean"], agg["ci"])]
        lower = [(map_x(step), map_y(m - c)) for step, m, c in zip(agg["steps"], agg["mean"], agg["ci"])]
        overlay_draw.polygon(upper + list(reversed(lower)), fill=(*color, style["alpha"]))
        image.alpha_composite(overlay)

        points = [(map_x(step), map_y(value)) for step, value in zip(agg["steps"], agg["mean"])]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        y0 = legend_y + idx * 28
        draw.line((legend_x, y0 + 10, legend_x + 32, y0 + 10), fill=color, width=4)
        draw.text((legend_x + 42, y0), f"{label} (n={agg['n_runs']})", fill=text_color, font=small_font)


def write_summary(out_dir: Path, groups: Dict[str, Dict[str, Optional[dict]]]) -> None:
    with (out_dir / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "metric_id", "n_runs", "first_step", "last_step", "last_step_n", "last_mean", "last_95ci_half_width", "seeds"])
        for group_name, metric_data in groups.items():
            for metric_id in METRIC_ORDER:
                agg = metric_data.get(metric_id)
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


def export_group(label: str, runs: Sequence[str], swanlog_dir: Path, out_dir: Path, refresh_cache: bool) -> Dict[str, Optional[dict]]:
    cache_dir = out_dir / "raw_metrics" / label.replace("+", "_plus_")
    csv_paths = []
    for run in runs:
        csv_paths.append(export_metrics(swanlog_dir / run, cache_dir, refresh_cache))
    by_metric = load_series(csv_paths)
    return by_metric


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    td3_series = export_group("TD3", TD3_RUNS, args.swanlog_dir, args.out_dir, args.refresh_cache)
    gail_series = export_group("GAIL+TD3", GAIL_RUNS, args.swanlog_dir, args.out_dir, args.refresh_cache)
    groups = {
        "TD3": {metric: aggregate(td3_series[metric], args.align, args.ci) for metric in METRIC_ORDER},
        "GAIL+TD3": {metric: aggregate(gail_series[metric], args.align, args.ci) for metric in METRIC_ORDER},
    }

    image = Image.new("RGBA", (1800, 1200), (255, 255, 255, 255))
    font = find_font(28)
    small_font = find_font(20)
    title_font = find_font(34)
    draw = ImageDraw.Draw(image)
    main_title = "TD3 vs GAIL+TD3: Mean Learning Curves with 95% CI"
    title_w, _ = text_size(draw, main_title, title_font)
    draw.text(((1800 - title_w) / 2, 28), main_title, fill=(25, 25, 25), font=title_font)

    rects = [
        (55, 90, 875, 585),
        (925, 90, 1745, 585),
        (55, 650, 875, 1145),
        (925, 650, 1745, 1145),
    ]
    for metric_id, rect in zip(METRIC_ORDER, rects):
        draw_metric(image, rect, metric_id, {group_name: data[metric_id] for group_name, data in groups.items()}, font, small_font)

    image.convert("RGB").save(args.out_dir / "td3_vs_gail_mean_ci_4plots.png", quality=95)
    write_summary(args.out_dir, groups)
    print(args.out_dir / "td3_vs_gail_mean_ci_4plots.png")


if __name__ == "__main__":
    main()
