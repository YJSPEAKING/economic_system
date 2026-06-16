from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw

from plot_two_algorithm_comparisons_direct import (
    COMPARISONS,
    FIG_HEIGHT,
    FIG_WIDTH,
    GROUP_DIRS,
    METRIC_ORDER,
    OUT_DIR,
    PLOT_RECTS,
    draw_plot,
    find_font,
    load_series,
    text_size,
)


def aggregate_sem(series: Sequence[dict]) -> Optional[dict]:
    if not series:
        return None
    steps = sorted(set.intersection(*(set(item["values"].keys()) for item in series)))
    if not steps:
        return None

    means: List[float] = []
    sems: List[float] = []
    ns: List[int] = []
    for step in steps:
        values = [item["values"][step] for item in series if step in item["values"]]
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
        "ci": sems,  # draw_plot uses this field as the half-width of the shaded band.
        "n_by_step": ns,
        "n_runs": len(series),
        "seeds": [item["seed"] for item in series],
    }


def write_summary(group_data: Dict[str, Dict[str, Optional[dict]]]) -> None:
    with (OUT_DIR / "direct_comparison_sem_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "metric_id", "n_runs", "first_step", "last_step", "last_step_n", "last_mean", "last_sem", "seeds"])
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
    missing_dirs = [str(path) for path in GROUP_DIRS.values() if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(
            "Missing cached raw_metrics directories. Run plot_two_algorithm_comparisons_matplotlib.py once first. "
            + "; ".join(missing_dirs)
        )

    raw_data = {group: load_series(path) for group, path in GROUP_DIRS.items()}
    group_data = {
        group: {metric_id: aggregate_sem(raw_data[group][metric_id]) for metric_id in METRIC_ORDER}
        for group in GROUP_DIRS
    }

    font = find_font(32)
    small_font = find_font(26)
    title_font = find_font(38)

    for comparison in COMPARISONS:
        image = Image.new("RGBA", (FIG_WIDTH, FIG_HEIGHT), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        title = comparison["title"] + " (Mean \u00b1 SEM)"
        title_w, _ = text_size(draw, title, title_font)
        draw.text(((FIG_WIDTH - title_w) / 2, 32), title, fill=(0, 0, 0), font=title_font)
        note = "Shaded area: \u00b11 SEM"
        draw.text((FIG_WIDTH - 390, 42), note, fill=(0, 0, 0), font=small_font)

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

        out_path = OUT_DIR / f"{comparison['name']}_sem.png"
        image.convert("RGB").save(out_path, quality=95, dpi=(300, 300))
        print(out_path)

    write_summary(group_data)


if __name__ == "__main__":
    main()
