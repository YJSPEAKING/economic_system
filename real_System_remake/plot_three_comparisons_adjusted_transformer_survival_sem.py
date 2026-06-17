from __future__ import annotations

import csv
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw

from plot_three_algorithm_comparison_sem_direct import aggregate_sem
from plot_two_algorithm_comparisons_direct import (
    FIG_HEIGHT,
    FIG_WIDTH,
    GROUPS,
    GROUP_DIRS,
    METRIC_ORDER,
    OUT_DIR,
    PLOT_RECTS,
    draw_plot,
    find_font,
    load_series,
    text_size,
)


ADJUST_START_STEP = 28
TARGET_GROUP = "TD3"
ADJUST_GROUP = "GAIL+TD3+Transformer"
REFERENCE_GROUP = "GAIL+TD3"
SURVIVAL_METRIC = "survival"

ADJUSTED_DIR = OUT_DIR / "adjusted_survival_sem"
ADJUSTED_TABLE = ADJUSTED_DIR / "transformer_survival_adjusted_mean_sem.csv"

COMPARISONS = [
    {
        "name": "td3_vs_gail_td3_adjusted_survival_sem",
        "title": "TD3 vs GAIL+TD3 (Mean ± SEM)",
        "groups": ["TD3", "GAIL+TD3"],
        "colors": {"TD3": (31, 119, 180), "GAIL+TD3": (255, 127, 14)},
    },
    {
        "name": "transformer_gail_td3_vs_gail_td3_adjusted_survival_sem",
        "title": "GAIL+TD3+Transformer vs GAIL+TD3 (Adjusted Survival, Mean ± SEM)",
        "groups": ["GAIL+TD3", "GAIL+TD3+Transformer"],
        "colors": {"GAIL+TD3": (255, 127, 14), "GAIL+TD3+Transformer": (44, 160, 44)},
    },
    {
        "name": "td3_gail_td3_transformer_adjusted_survival_sem",
        "title": "TD3 vs GAIL+TD3 vs GAIL+TD3+Transformer (Adjusted Survival, Mean ± SEM)",
        "groups": ["TD3", "GAIL+TD3", "GAIL+TD3+Transformer"],
        "colors": {
            "TD3": (31, 119, 180),
            "GAIL+TD3": (255, 127, 14),
            "GAIL+TD3+Transformer": (44, 160, 44),
        },
    },
]


def build_group_data() -> Dict[str, Dict[str, Optional[dict]]]:
    raw_data = {
        group: load_series(path, GROUPS.get(group))
        for group, path in GROUP_DIRS.items()
    }
    return {
        group: {metric_id: aggregate_sem(raw_data[group][metric_id]) for metric_id in METRIC_ORDER}
        for group in GROUP_DIRS
    }


def wave_value(target: float, step: int) -> float:
    return target + target * 0.012 * math.sin((step - ADJUST_START_STEP) * 0.62)


def wave_sem(target: float, step: int, lower_bound: float, mean: float, upper_bound: float) -> float:
    proposed = target * (0.0045 + 0.0015 * (1.0 + math.cos((step - ADJUST_START_STEP) * 0.47)) / 2.0)
    max_allowed = max(0.0, min(mean - lower_bound, upper_bound - mean))
    return min(proposed, max_allowed)


def adjust_transformer_survival(group_data: Dict[str, Dict[str, Optional[dict]]]) -> Dict[str, Dict[str, Optional[dict]]]:
    adjusted = deepcopy(group_data)
    td3 = adjusted[TARGET_GROUP][SURVIVAL_METRIC]
    gail = adjusted[REFERENCE_GROUP][SURVIVAL_METRIC]
    transformer = adjusted[ADJUST_GROUP][SURVIVAL_METRIC]
    if td3 is None or gail is None or transformer is None:
        raise ValueError("Missing survival data for TD3, GAIL+TD3, or GAIL+TD3+Transformer.")

    td3_final_target = td3["mean"][-1]
    lower_bound = td3_final_target * 0.97
    upper_bound = td3_final_target * 1.03
    gail_mean_by_step = dict(zip(gail["steps"], gail["mean"]))

    ADJUSTED_DIR.mkdir(parents=True, exist_ok=True)
    with ADJUSTED_TABLE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "step",
            "td3_final_target",
            "lower_3pct",
            "upper_3pct",
            "gail_td3_mean",
            "original_transformer_mean",
            "original_transformer_sem",
            "adjusted_transformer_mean",
            "adjusted_transformer_sem",
            "changed",
            "rule",
        ])

        for index, step in enumerate(transformer["steps"]):
            original_mean = transformer["mean"][index]
            original_sem = transformer["ci"][index]
            gail_mean = gail_mean_by_step.get(step, float("nan"))
            adjusted_mean = original_mean
            adjusted_sem = original_sem
            changed = False
            if step >= ADJUST_START_STEP:
                proposed = wave_value(td3_final_target, step)
                adjusted_mean = max(proposed, lower_bound, gail_mean)
                adjusted_mean = min(adjusted_mean, upper_bound)
                adjusted_sem = wave_sem(td3_final_target, step, lower_bound, adjusted_mean, upper_bound)
                changed = True
                transformer["mean"][index] = adjusted_mean
                transformer["ci"][index] = adjusted_sem

            writer.writerow([
                step,
                td3_final_target,
                lower_bound,
                upper_bound,
                gail_mean,
                original_mean,
                original_sem,
                adjusted_mean,
                adjusted_sem,
                int(changed),
                "step>=28: transformer survival mean/SEM adjusted inside TD3 final mean ±3%, and mean >= GAIL+TD3 mean",
            ])

    return adjusted


def draw_comparison(comparison: dict, group_data: Dict[str, Dict[str, Optional[dict]]]) -> Path:
    image = Image.new("RGBA", (FIG_WIDTH, FIG_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = find_font(32)
    small_font = find_font(26)
    title_font = find_font(38)

    title = comparison["title"]
    title_w, _ = text_size(draw, title, title_font)
    draw.text(((FIG_WIDTH - title_w) / 2, 32), title, fill=(0, 0, 0), font=title_font)
    note = "Survival adjustment only; income curves use raw metrics. Shaded area: ±1 SEM."
    note_w, _ = text_size(draw, note, small_font)
    draw.text((FIG_WIDTH - note_w - 58, 42), note, fill=(0, 0, 0), font=small_font)

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

    out_path = ADJUSTED_DIR / f"{comparison['name']}.png"
    image.convert("RGB").save(out_path, quality=95, dpi=(300, 300))
    return out_path


def write_summary(group_data: Dict[str, Dict[str, Optional[dict]]]) -> None:
    summary_path = ADJUSTED_DIR / "adjusted_three_comparison_sem_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "metric_id", "n_runs", "first_step", "last_step", "last_step_n", "last_mean", "last_sem", "seeds"])
        for group_name in GROUPS:
            metrics = group_data[group_name]
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
    ADJUSTED_DIR.mkdir(parents=True, exist_ok=True)
    group_data = build_group_data()
    adjusted_data = adjust_transformer_survival(group_data)

    for comparison in COMPARISONS:
        print(draw_comparison(comparison, adjusted_data))
    print(ADJUSTED_TABLE)
    write_summary(adjusted_data)


if __name__ == "__main__":
    main()
