from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw

import plot_three_comparisons_sci_style_adjusted_survival_sem as sci_style
from plot_three_comparisons_adjusted_transformer_survival_sem import build_group_data
from plot_two_algorithm_comparisons_direct import FIG_HEIGHT, FIG_WIDTH, METRIC_ORDER, OUT_DIR, PLOT_RECTS, text_size


COMPARISONS = [
    {
        "name": "td3_vs_gail_td3_sem",
        "title": "TD3 vs GAIL+TD3 (Mean ± SEM)",
        "groups": ["TD3", "GAIL+TD3"],
        "colors": {"TD3": (31, 119, 180), "GAIL+TD3": (255, 127, 14)},
    },
    {
        "name": "transformer_gail_td3_vs_gail_td3_sem",
        "title": "GAIL+TD3+Transformer vs GAIL+TD3 (Mean ± SEM)",
        "groups": ["GAIL+TD3", "GAIL+TD3+Transformer"],
        "colors": {"GAIL+TD3": (255, 127, 14), "GAIL+TD3+Transformer": (44, 160, 44)},
    },
    {
        "name": "td3_gail_td3_transformer_sem",
        "title": "TD3 vs GAIL+TD3 vs GAIL+TD3+Transformer (Mean ± SEM)",
        "groups": ["TD3", "GAIL+TD3", "GAIL+TD3+Transformer"],
        "colors": {
            "TD3": (31, 119, 180),
            "GAIL+TD3": (255, 127, 14),
            "GAIL+TD3+Transformer": (44, 160, 44),
        },
    },
]


def draw_comparison(comparison: dict, group_data: Dict[str, Dict[str, Optional[dict]]]) -> Path:
    image = Image.new("RGBA", (FIG_WIDTH, FIG_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    fonts = {
        "suptitle": sci_style.find_font(38, prefer_times=True),
        "title": sci_style.find_font(32),
        "title_en": sci_style.find_font(34, prefer_times=True),
        "label": sci_style.find_font(30, prefer_times=True),
        "tick": sci_style.find_font(27, prefer_times=True),
        "small_tick": sci_style.find_font(21, prefer_times=True),
        "legend": sci_style.find_font(20, prefer_times=True),
        "note": sci_style.find_font(23, prefer_times=True),
    }

    title = comparison["title"]
    title_w, _ = text_size(draw, title, fonts["suptitle"])
    draw.text(((FIG_WIDTH - title_w) / 2, 26), title, fill=(0, 0, 0), font=fonts["suptitle"])
    note = "Transparent shaded area: +/-1 SEM; n=8 per group."
    note_w, _ = text_size(draw, note, fonts["note"])
    draw.text(((FIG_WIDTH - note_w) / 2, 70), note, fill=(35, 35, 35), font=fonts["note"])

    for metric_id, rect in zip(METRIC_ORDER, PLOT_RECTS):
        sci_style.draw_plot_sci(
            image,
            rect,
            metric_id,
            {group: group_data[group][metric_id] for group in comparison["groups"]},
            comparison["colors"],
            fonts,
        )

    out_path = OUT_DIR / f"{comparison['name']}.png"
    image.convert("RGB").save(out_path, quality=95, dpi=(300, 300))
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sci_style.METRIC_TITLES["survival"] = "(a) 每百回合/存活天数"
    group_data = build_group_data()
    for comparison in COMPARISONS:
        print(draw_comparison(comparison, group_data))


if __name__ == "__main__":
    main()
