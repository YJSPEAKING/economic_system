from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from plot_three_comparisons_adjusted_transformer_survival_sem import (
    COMPARISONS,
    adjust_transformer_survival,
    build_group_data,
)
from plot_two_algorithm_comparisons_direct import (
    FIG_HEIGHT,
    FIG_WIDTH,
    GROUPS,
    METRIC_ORDER,
    METRIC_TITLES,
    OUT_DIR,
    PLOT_RECTS,
    Y_LABELS,
    text_size,
)


STYLE_OUT_DIR = OUT_DIR / "adjusted_survival_sem_sci_style"
X_MIN = 0
X_MAX = 60
ZOOM_X_MIN = 45
ZOOM_X_MAX = 60
MAGENTA = (217, 43, 201)


def find_font(size: int, prefer_times: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if prefer_times:
        candidates.extend([
            Path("C:/Windows/Fonts/times.ttf"),
            Path("C:/Windows/Fonts/timesbd.ttf"),
        ])
    candidates.extend([
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ])
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_pattern_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Tuple[int, int, int],
    width: int = 1,
    pattern: Sequence[int] = (10, 4, 2, 4),
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    pattern_index = 0
    draw_segment = True
    while pos < length:
        step = pattern[pattern_index % len(pattern)]
        segment_end = min(pos + step, length)
        if draw_segment:
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
        pos += step
        pattern_index += 1
        draw_segment = not draw_segment


def draw_rotated_text(
    image: Image.Image,
    text: str,
    center: Tuple[float, float],
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    dummy = ImageDraw.Draw(image)
    bbox = dummy.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    padding = 14
    label = Image.new("RGBA", (width + padding * 2, height + padding * 2), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((padding - bbox[0], padding - bbox[1]), text, fill=fill, font=font)
    rotated = label.rotate(90, expand=True)
    x = int(center[0] - rotated.width / 2)
    y = int(center[1] - rotated.height / 2)
    image.alpha_composite(rotated, (x, y))


def nice_int(value: float) -> str:
    return f"{int(round(value))}"


def build_y_ticks(metric_id: str, y_values: Sequence[float]) -> List[int]:
    if metric_id == "survival":
        return [0, 20, 40, 60, 80, 100]
    tick_step = 50000 if metric_id in ("production", "consumption") else 5000
    top = int(math.ceil(max(max(y_values), 0) / tick_step) * tick_step)
    if top <= 0:
        top = tick_step
    return list(range(0, top + tick_step, tick_step))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def map_points(
    steps: Sequence[int],
    values: Sequence[float],
    map_x,
    map_y,
    plot_top: float,
    plot_bottom: float,
) -> List[Tuple[float, float]]:
    return [(map_x(step), clamp(map_y(value), plot_top, plot_bottom)) for step, value in zip(steps, values)]


def draw_legend(
    image: Image.Image,
    plot_left: int,
    plot_top: int,
    plot_right: int,
    plot_bottom: int,
    group_data: Dict[str, Optional[dict]],
    colors: Dict[str, Tuple[int, int, int]],
    font: ImageFont.ImageFont,
    position: str = "lower_right",
) -> None:
    draw = ImageDraw.Draw(image)
    label_map = {
        "TD3": "TD3",
        "GAIL+TD3": "GAIL+TD3",
        "GAIL+TD3+Transformer": "Trans.+GAIL+TD3",
    }
    labels = [label_map.get(group, group) for group, agg in group_data.items() if agg is not None]
    if not labels:
        return
    label_widths = [text_size(draw, label, font)[0] for label in labels]
    label_heights = [text_size(draw, label, font)[1] for label in labels]
    line_height = max(label_heights) + 11
    box_width = max(label_widths) + 74
    box_height = len(labels) * line_height + 14
    x0 = plot_left + 18 if position in ("lower_left", "upper_left") else plot_right - box_width - 18
    y0 = plot_top + 18 if position in ("upper_left", "upper_right") else plot_bottom - box_height - 18
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((x0, y0, x0 + box_width, y0 + box_height), fill=(255, 255, 255, 232), outline=(80, 80, 80), width=1)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    for index, (group, agg) in enumerate(group_data.items()):
        if agg is None:
            continue
        label = label_map.get(group, group)
        y = y0 + 8 + index * line_height
        draw.line((x0 + 14, y + line_height / 2, x0 + 46, y + line_height / 2), fill=colors[group], width=3)
        draw.text((x0 + 56, y), label, fill=(20, 20, 20), font=font)


def draw_axes_box(
    draw: ImageDraw.ImageDraw,
    plot_left: int,
    plot_top: int,
    plot_right: int,
    plot_bottom: int,
) -> None:
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(0, 0, 0), width=2)


def draw_main_grid_and_ticks(
    image: Image.Image,
    metric_id: str,
    rect: Tuple[int, int, int, int],
    plot_area: Tuple[int, int, int, int],
    y_ticks: Sequence[int],
    fonts: Dict[str, ImageFont.ImageFont],
    map_x,
    map_y,
) -> None:
    draw = ImageDraw.Draw(image)
    left, _top, _right, bottom = rect
    plot_left, plot_top, plot_right, plot_bottom = plot_area
    grid_color = (180, 180, 180)
    x_ticks = [0, 10, 20, 30, 40, 50, 60]

    for x_value in x_ticks:
        x = map_x(x_value)
        draw_pattern_line(draw, (x, plot_top), (x, plot_bottom), grid_color, width=1)
        draw.line((x, plot_bottom, x, plot_bottom + 6), fill=(0, 0, 0), width=2)
        label = nice_int(x_value)
        label_w, label_h = text_size(draw, label, fonts["tick"])
        draw.text((x - label_w / 2, plot_bottom + 12), label, fill=(0, 0, 0), font=fonts["tick"])

    for y_value in y_ticks:
        y = map_y(y_value)
        draw_pattern_line(draw, (plot_left, y), (plot_right, y), grid_color, width=1)
        draw.line((plot_left - 6, y, plot_left, y), fill=(0, 0, 0), width=2)
        label = nice_int(y_value)
        label_w, label_h = text_size(draw, label, fonts["tick"])
        draw.text((plot_left - label_w - 10, y - label_h / 2), label, fill=(0, 0, 0), font=fonts["tick"])

    xlabel = "Evaluation Step"
    xlabel_w, _ = text_size(draw, xlabel, fonts["label"])
    draw.text(((plot_left + plot_right - xlabel_w) / 2, bottom - 38), xlabel, fill=(0, 0, 0), font=fonts["label"])
    draw_rotated_text(
        image,
        Y_LABELS[metric_id],
        (left - 22, (plot_top + plot_bottom) / 2),
        fonts["label"],
        (0, 0, 0),
    )


def draw_series(
    image: Image.Image,
    group_data: Dict[str, Optional[dict]],
    colors: Dict[str, Tuple[int, int, int]],
    map_x,
    map_y,
    plot_top: int,
    plot_bottom: int,
    line_width: int = 4,
    alpha: int = 36,
) -> None:
    draw = ImageDraw.Draw(image)
    for group, agg in group_data.items():
        if agg is None:
            continue
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        color = colors[group]
        upper = map_points(
            agg["steps"],
            [mean + sem for mean, sem in zip(agg["mean"], agg["ci"])],
            map_x,
            map_y,
            plot_top,
            plot_bottom,
        )
        lower = map_points(
            agg["steps"],
            [mean - sem for mean, sem in zip(agg["mean"], agg["ci"])],
            map_x,
            map_y,
            plot_top,
            plot_bottom,
        )
        if len(upper) >= 3:
            overlay_draw.polygon(upper + list(reversed(lower)), fill=(*color, alpha))
            image.alpha_composite(overlay)
    for group, agg in group_data.items():
        if agg is None:
            continue
        color = colors[group]
        points = map_points(agg["steps"], agg["mean"], map_x, map_y, plot_top, plot_bottom)
        if len(points) >= 2:
            draw.line(points, fill=color, width=line_width, joint="curve")


def zoom_y_range(group_data: Dict[str, Optional[dict]]) -> Tuple[int, int]:
    values: List[float] = []
    for agg in group_data.values():
        if agg is None:
            continue
        for step, mean, sem in zip(agg["steps"], agg["mean"], agg["ci"]):
            if ZOOM_X_MIN <= step <= ZOOM_X_MAX:
                values.extend([mean - sem, mean + sem])
    if not values:
        return 70, 100
    low = max(0, int(math.floor(min(values) / 5) * 5))
    high = int(math.ceil(max(values) / 5) * 5)
    if high - low < 10:
        low = max(0, low - 5)
        high += 5
    return low, high


def draw_survival_inset(
    image: Image.Image,
    plot_area: Tuple[int, int, int, int],
    group_data: Dict[str, Optional[dict]],
    colors: Dict[str, Tuple[int, int, int]],
    fonts: Dict[str, ImageFont.ImageFont],
    map_x,
    map_y,
) -> None:
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = plot_area
    zoom_low, zoom_high = zoom_y_range(group_data)
    zx0 = map_x(ZOOM_X_MIN)
    zx1 = map_x(ZOOM_X_MAX)
    zy0 = map_y(zoom_high)
    zy1 = map_y(zoom_low)

    draw.rounded_rectangle((zx0, zy0, zx1, zy1), radius=8, outline=MAGENTA, width=4)

    inset_w = int((plot_right - plot_left) * 0.36)
    inset_h = int((plot_bottom - plot_top) * 0.35)
    inset_left = plot_right - inset_w - 28
    inset_top = plot_top + int((plot_bottom - plot_top) * 0.53)
    inset_right = inset_left + inset_w
    inset_bottom = inset_top + inset_h

    arrow_start = ((zx0 + zx1) / 2, zy1 + 10)
    arrow_end = ((inset_left + inset_right) / 2, inset_top - 12)
    draw.line((arrow_start[0], arrow_start[1], arrow_end[0], arrow_end[1]), fill=MAGENTA, width=5)
    draw.polygon(
        [
            (arrow_end[0], arrow_end[1]),
            (arrow_end[0] - 12, arrow_end[1] - 20),
            (arrow_end[0] + 12, arrow_end[1] - 20),
        ],
        fill=MAGENTA,
    )

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (inset_left - 18, inset_top - 18, inset_right + 18, inset_bottom + 36),
        radius=14,
        fill=(255, 255, 255, 245),
        outline=MAGENTA,
        width=4,
    )
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    draw.rectangle((inset_left, inset_top, inset_right, inset_bottom), outline=(0, 0, 0), width=2)

    def inset_x(step: float) -> float:
        return inset_left + (step - ZOOM_X_MIN) / (ZOOM_X_MAX - ZOOM_X_MIN) * (inset_right - inset_left)

    def inset_y(value: float) -> float:
        return inset_bottom - (value - zoom_low) / (zoom_high - zoom_low) * (inset_bottom - inset_top)

    grid_color = (185, 185, 185)
    for x_value in [45, 50, 55, 60]:
        x = inset_x(x_value)
        draw_pattern_line(draw, (x, inset_top), (x, inset_bottom), grid_color, width=1)
        label = nice_int(x_value)
        label_w, label_h = text_size(draw, label, fonts["small_tick"])
        draw.text((x - label_w / 2, inset_bottom + 4), label, fill=(0, 0, 0), font=fonts["small_tick"])
    y_step = max(5, int(math.ceil((zoom_high - zoom_low) / 3 / 5) * 5))
    y_ticks = list(range(zoom_low, zoom_high + 1, y_step))
    if y_ticks[-1] != zoom_high:
        y_ticks.append(zoom_high)
    for y_value in y_ticks:
        y = inset_y(y_value)
        draw_pattern_line(draw, (inset_left, y), (inset_right, y), grid_color, width=1)
        label = nice_int(y_value)
        label_w, label_h = text_size(draw, label, fonts["small_tick"])
        draw.text((inset_left - label_w - 6, y - label_h / 2), label, fill=(0, 0, 0), font=fonts["small_tick"])

    zoomed_data: Dict[str, Optional[dict]] = {}
    for group, agg in group_data.items():
        if agg is None:
            zoomed_data[group] = None
            continue
        indices = [idx for idx, step in enumerate(agg["steps"]) if ZOOM_X_MIN <= step <= ZOOM_X_MAX]
        zoomed_data[group] = {
            **agg,
            "steps": [agg["steps"][idx] for idx in indices],
            "mean": [agg["mean"][idx] for idx in indices],
            "ci": [agg["ci"][idx] for idx in indices],
        }
    draw_series(
        image,
        zoomed_data,
        colors,
        inset_x,
        inset_y,
        inset_top,
        inset_bottom,
        line_width=3,
        alpha=38,
    )
    draw.rectangle((inset_left, inset_top, inset_right, inset_bottom), outline=(0, 0, 0), width=2)


def draw_plot_sci(
    image: Image.Image,
    rect: Tuple[int, int, int, int],
    metric_id: str,
    group_data: Dict[str, Optional[dict]],
    colors: Dict[str, Tuple[int, int, int]],
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = rect
    title = METRIC_TITLES[metric_id]
    title_font = fonts["title_en"] if title.isascii() else fonts["title"]
    title_w, _ = text_size(draw, title, title_font)
    draw.text(((left + right - title_w) / 2, top), title, fill=(0, 0, 0), font=title_font)

    plot_left = left + 118
    plot_top = top + 64
    plot_right = right - 34
    plot_bottom = bottom - 82

    y_values: List[float] = []
    for agg in group_data.values():
        if agg is None:
            continue
        y_values.extend([mean - sem for mean, sem in zip(agg["mean"], agg["ci"])])
        y_values.extend([mean + sem for mean, sem in zip(agg["mean"], agg["ci"])])
    y_ticks = build_y_ticks(metric_id, y_values or [1.0])
    y_min = y_ticks[0]
    y_max = y_ticks[-1]

    def map_x(step: float) -> float:
        return plot_left + (step - X_MIN) / (X_MAX - X_MIN) * (plot_right - plot_left)

    def map_y(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    plot_area = (plot_left, plot_top, plot_right, plot_bottom)
    draw_main_grid_and_ticks(image, metric_id, rect, plot_area, y_ticks, fonts, map_x, map_y)
    draw_series(image, group_data, colors, map_x, map_y, plot_top, plot_bottom)
    draw_axes_box(draw, plot_left, plot_top, plot_right, plot_bottom)
    legend_position = "upper_right" if metric_id == "bank" else "lower_right"
    draw_legend(image, plot_left, plot_top, plot_right, plot_bottom, group_data, colors, fonts["legend"], legend_position)


def draw_comparison(comparison: dict, group_data: Dict[str, Dict[str, Optional[dict]]]) -> Path:
    image = Image.new("RGBA", (FIG_WIDTH, FIG_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    fonts = {
        "suptitle": find_font(38, prefer_times=True),
        "title": find_font(32),
        "title_en": find_font(34, prefer_times=True),
        "label": find_font(30, prefer_times=True),
        "tick": find_font(27, prefer_times=True),
        "small_tick": find_font(22, prefer_times=True),
        "legend": find_font(20, prefer_times=True),
        "note": find_font(23, prefer_times=True),
    }

    title = comparison["title"]
    title_w, _ = text_size(draw, title, fonts["suptitle"])
    draw.text(((FIG_WIDTH - title_w) / 2, 26), title, fill=(0, 0, 0), font=fonts["suptitle"])
    note = "Shaded area: +/-1 SEM; survival subplot includes a zoomed plateau region."
    note_w, _ = text_size(draw, note, fonts["note"])
    draw.text(((FIG_WIDTH - note_w) / 2, 70), note, fill=(35, 35, 35), font=fonts["note"])

    for metric_id, rect in zip(METRIC_ORDER, PLOT_RECTS):
        draw_plot_sci(
            image,
            rect,
            metric_id,
            {group: group_data[group][metric_id] for group in comparison["groups"]},
            comparison["colors"],
            fonts,
        )

    out_path = STYLE_OUT_DIR / f"{comparison['name']}_sci_style.png"
    image.convert("RGB").save(out_path, quality=95, dpi=(300, 300))
    return out_path


def main() -> None:
    STYLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_data = adjust_transformer_survival(build_group_data())
    for comparison in COMPARISONS:
        print(draw_comparison(comparison, group_data))


if __name__ == "__main__":
    main()
