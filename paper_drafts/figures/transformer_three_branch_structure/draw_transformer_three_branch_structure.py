from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "transformer_three_branch_structure.png"
SVG_PATH = OUTPUT_DIR / "transformer_three_branch_structure.svg"

TIMES = font_manager.FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
TIMES_BOLD = font_manager.FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf")

TEXT = "#20252B"
EDGE = "#3F566E"
FLOW = "#2F3740"
SEPARATOR = "#CDD4DA"
INPUT_FILL = "#DDECF7"
TRANSFORMER_FILL = "#FFF0BD"
NORM_FILL = "#E9E0F5"
GROUP_FILL = "#F6F1FA"
OUTPUT_FILL = "#F8D9B8"
NEUTRAL_FILL = "#F4F6F8"
CONCAT_EDGE = "#778491"
FONT_BUMP = 2.8


def add_text(
    ax,
    x,
    y,
    label,
    *,
    size=10,
    bold=False,
    ha="center",
    va="center",
    color=TEXT,
    zorder=7,
):
    ax.text(
        x,
        y,
        label,
        ha=ha,
        va=va,
        color=color,
        fontsize=size + FONT_BUMP,
        fontproperties=TIMES_BOLD,
        fontweight="bold",
        linespacing=1.08,
        zorder=zorder,
    )


def add_box(
    ax,
    center,
    width,
    height,
    label,
    *,
    fill,
    fontsize=9.5,
    linewidth=1.55,
    rounding=0.08,
    bold=False,
    edgecolor=EDGE,
    zorder=4,
):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.03,rounding_size={rounding}",
        facecolor=fill,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    add_text(ax, x, y, label, size=fontsize, bold=bold, zorder=zorder + 1)
    return patch


def add_title_subtitle_box(
    ax,
    center,
    width,
    height,
    title,
    subtitle,
    *,
    fill,
    title_size=10.1,
    subtitle_size=8.3,
    title_offset=0.18,
    subtitle_offset=0.20,
):
    x, y = center
    patch = add_box(
        ax,
        center,
        width,
        height,
        "",
        fill=fill,
    )
    add_text(ax, x, y + title_offset, title, size=title_size, bold=True, zorder=6)
    add_text(
        ax,
        x,
        y - subtitle_offset,
        f"({subtitle})",
        size=subtitle_size,
        zorder=6,
    )
    return patch


def add_arrow(
    ax,
    start,
    end,
    *,
    linewidth=1.50,
    color=FLOW,
    connectionstyle="arc3",
    zorder=3,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_routed_arrow(ax, points, *, linewidth=1.50, color=FLOW):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=linewidth,
            zorder=2.8,
        )
    add_arrow(ax, points[-2], points[-1], linewidth=linewidth, color=color)


def connect_vertical(ax, x, upper_y, upper_height, lower_y, lower_height):
    add_arrow(
        ax,
        (x, upper_y - upper_height / 2),
        (x, lower_y + lower_height / 2),
    )


def add_branch_heading(ax, x, label):
    add_text(ax, x, 21.45, label, size=13.5, bold=True)
    ax.plot(
        [x - 2.35, x + 2.35],
        [21.08, 21.08],
        color=EDGE,
        linewidth=1.8,
        zorder=2,
    )


def add_representation_module(ax, x, input_label, module_symbol):
    input_y, input_h = 20.05, 0.90
    module_y, module_h = 18.35, 1.18
    add_box(
        ax,
        (x, input_y),
        5.05,
        input_h,
        input_label,
        fill=INPUT_FILL,
        fontsize=9.2,
    )
    add_box(
        ax,
        (x, module_y),
        4.55,
        module_h,
        "",
        fill=NORM_FILL,
    )
    add_text(
        ax,
        x,
        module_y + 0.20,
        "Historical Sequence\nRepresentation Module",
        size=9.3,
        bold=True,
    )
    add_text(ax, x, module_y - 0.34, module_symbol, size=10.8, bold=True)
    connect_vertical(ax, x, input_y, input_h, module_y, module_h)
    return module_y, module_h


def add_concat_module(
    ax,
    x,
    source_y,
    source_h,
    *,
    module_title,
    current_label,
    historical_label,
    output_label,
):
    module_left, module_bottom = x - 2.53, 12.55
    module_width, module_height = 5.06, 4.55
    border = FancyBboxPatch(
        (module_left, module_bottom),
        module_width,
        module_height,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="white",
        edgecolor=CONCAT_EDGE,
        linewidth=1.60,
        linestyle=(0, (5, 3)),
        zorder=1,
    )
    ax.add_patch(border)
    add_text(
        ax,
        x,
        module_bottom + module_height - 0.24,
        module_title,
        size=9.5,
        bold=True,
        zorder=6,
    )

    input_y, input_h, input_w = 15.75, 1.00, 2.30
    left_x, right_x = x - 1.25, x + 1.25
    add_box(
        ax,
        (left_x, input_y),
        input_w,
        input_h,
        current_label,
        fill=INPUT_FILL,
        fontsize=8.7,
    )
    add_box(
        ax,
        (right_x, input_y),
        input_w,
        input_h,
        historical_label,
        fill=INPUT_FILL,
        fontsize=8.5,
    )
    target_x = right_x + input_w / 4
    add_routed_arrow(
        ax,
        [
            (x, source_y - source_h / 2),
            (x, 17.35),
            (target_x, 17.35),
            (target_x, input_y + input_h / 2),
        ],
    )

    concat_y = 14.35
    ax.scatter(
        [x],
        [concat_y],
        s=1220,
        marker="o",
        facecolors=OUTPUT_FILL,
        edgecolors=EDGE,
        linewidths=1.55,
        zorder=5,
    )
    add_text(ax, x, concat_y, "Concat", size=8.0, zorder=6)
    add_arrow(
        ax,
        (left_x, input_y - input_h / 2),
        (x - 0.18, concat_y + 0.25),
    )
    add_arrow(
        ax,
        (right_x, input_y - input_h / 2),
        (x + 0.18, concat_y + 0.25),
    )

    output_y, output_h = 13.20, 0.72
    add_box(
        ax,
        (x, output_y),
        2.55,
        output_h,
        output_label,
        fill=NEUTRAL_FILL,
        fontsize=9.5,
    )
    add_arrow(ax, (x, concat_y - 0.27), (x, output_y + output_h / 2))
    return output_y, output_h


def draw_actor_branch(ax, x):
    add_branch_heading(ax, x, "Actor Branch")
    module_y, module_h = add_representation_module(
        ax,
        x,
        "Historical State Sequence\n" + r"$H_t^s=(s_{t-l+1},\ldots,s_t)$",
        r"$\mathcal{F}^{A}$",
    )
    output_y, output_h = add_concat_module(
        ax,
        x,
        module_y,
        module_h,
        module_title="State-Feature Concatenation",
        current_label="Current State\n$s_t$",
        historical_label="Historical State Feature\n$z_t^A$",
        output_label=r"$[s_t\,;\,z_t^A]$",
    )
    head_y, head_h = 11.25, 0.80
    action_y, action_h = 10.02, 0.80
    add_box(
        ax,
        (x, head_y),
        3.40,
        head_h,
        r"FC Action Head  $\mu_\psi(\cdot)$",
        fill=OUTPUT_FILL,
        fontsize=9.5,
    )
    connect_vertical(ax, x, output_y, output_h, head_y, head_h)
    add_box(
        ax,
        (x, action_y),
        3.10,
        action_h,
        "Deterministic Action\n$a_t$",
        fill=OUTPUT_FILL,
        fontsize=9.5,
        bold=True,
    )
    connect_vertical(ax, x, head_y, head_h, action_y, action_h)


def draw_critic_branch(ax, x):
    add_branch_heading(ax, x, "Critic Branch")
    module_y, module_h = add_representation_module(
        ax,
        x,
        "Historical State-Action Sequence\n"
        + r"$H_t^{sa}=([s_{t-l+1},a_{t-l+1}],\ldots,[s_t,a_t])$",
        r"$\mathcal{F}^{Q}$",
    )
    output_y, output_h = add_concat_module(
        ax,
        x,
        module_y,
        module_h,
        module_title="State-Action-Feature Concatenation",
        current_label="Current State-Action\nPair\n$[s_t,a_t]$",
        historical_label="Historical State-Action\nFeature\n$z_t^Q$",
        output_label=r"$[s_t,a_t\,;\,z_t^Q]$",
    )
    head_y, head_h = 11.25, 0.80
    output_q_y, output_q_h = 9.98, 0.78
    add_box(
        ax,
        (x, head_y),
        3.10,
        head_h,
        "Twin Q Heads",
        fill=OUTPUT_FILL,
        fontsize=9.6,
        bold=True,
    )
    connect_vertical(ax, x, output_y, output_h, head_y, head_h)
    q_offset, q_width = 1.02, 1.78
    add_box(
        ax,
        (x - q_offset, output_q_y),
        q_width,
        output_q_h,
        "$Q_1(s_t,a_t)$",
        fill=OUTPUT_FILL,
        fontsize=9.5,
    )
    add_box(
        ax,
        (x + q_offset, output_q_y),
        q_width,
        output_q_h,
        "$Q_2(s_t,a_t)$",
        fill=OUTPUT_FILL,
        fontsize=9.5,
    )
    add_arrow(
        ax,
        (x - 0.25, head_y - head_h / 2),
        (x - q_offset, output_q_y + output_q_h / 2),
    )
    add_arrow(
        ax,
        (x + 0.25, head_y - head_h / 2),
        (x + q_offset, output_q_y + output_q_h / 2),
    )


def draw_discriminator_branch(ax, x):
    add_branch_heading(ax, x, "Discriminator Branch")
    module_y, module_h = add_representation_module(
        ax,
        x,
        "Expert / Generated State-Action Sequence\n$H_t^{sa}$",
        r"$\mathcal{F}^{D}$",
    )
    head_y, head_h = 15.55, 1.00
    probability_y, probability_h = 13.55, 0.92
    add_title_subtitle_box(
        ax,
        (x, head_y),
        3.85,
        head_h,
        "Discriminator Head",
        "Linear Projection + Sigmoid",
        fill=OUTPUT_FILL,
        title_size=10.0,
        subtitle_size=8.4,
        title_offset=0.17,
        subtitle_offset=0.19,
    )
    connect_vertical(ax, x, module_y, module_h, head_y, head_h)
    add_box(
        ax,
        (x, probability_y),
        4.00,
        probability_h,
        "Expert Probability\n$D(H_t^{sa})$",
        fill=OUTPUT_FILL,
        fontsize=9.5,
        bold=True,
    )
    connect_vertical(ax, x, head_y, head_h, probability_y, probability_h)


def add_common_module_definition(ax):
    left, bottom, width, height = 0.55, 4.45, 16.90, 3.25
    border = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=GROUP_FILL,
        edgecolor=CONCAT_EDGE,
        linewidth=1.65,
        linestyle=(0, (5, 3)),
        zorder=1,
    )
    ax.add_patch(border)
    add_text(
        ax,
        9.0,
        7.36,
        "Transformer-Based Historical Sequence Representation Module",
        size=12.0,
        bold=True,
    )
    add_text(
        ax,
        9.0,
        6.88,
        r"$\mathcal{F}^{(b)},\quad b\in\{A,Q,D\}$",
        size=10.0,
        bold=True,
    )

    flow_y = 5.63
    embedding_w, embedding_h = 3.30, 1.42
    transformer_w, transformer_h = 2.70, 1.02
    readout_w, readout_h = 3.30, 1.42
    feature_w, feature_h = 2.35, 1.02
    gap = 0.65
    flow_width = embedding_w + transformer_w + readout_w + feature_w + 3 * gap
    flow_left = left + (width - flow_width) / 2
    embedding_x = flow_left + embedding_w / 2
    transformer_x = flow_left + embedding_w + gap + transformer_w / 2
    readout_x = (
        flow_left
        + embedding_w
        + gap
        + transformer_w
        + gap
        + readout_w / 2
    )
    feature_x = (
        flow_left
        + embedding_w
        + gap
        + transformer_w
        + gap
        + readout_w
        + gap
        + feature_w / 2
    )

    add_title_subtitle_box(
        ax,
        (embedding_x, flow_y),
        embedding_w,
        embedding_h,
        "Sequence Embedding",
        "Linear Projection + Learnable\nPositional Embedding",
        fill=INPUT_FILL,
        title_size=9.4,
        subtitle_size=7.4,
        title_offset=0.29,
        subtitle_offset=0.25,
    )
    add_box(
        ax,
        (transformer_x, flow_y),
        transformer_w,
        transformer_h,
        "Transformer Encoder",
        fill=TRANSFORMER_FILL,
        fontsize=9.1,
        bold=True,
    )
    add_title_subtitle_box(
        ax,
        (readout_x, flow_y),
        readout_w,
        readout_h,
        "Last-Step Temporal Readout",
        "Last-Step Selection + LayerNorm",
        fill=NORM_FILL,
        title_size=8.8,
        subtitle_size=7.4,
        title_offset=0.27,
        subtitle_offset=0.25,
    )
    add_box(
        ax,
        (feature_x, flow_y),
        feature_w,
        feature_h,
        "Historical Feature\n$z_t^{(b)}$",
        fill=NEUTRAL_FILL,
        fontsize=9.1,
    )

    input_arrow_gap = 0.95
    input_arrow_start = embedding_x - embedding_w / 2 - input_arrow_gap
    input_arrow_end = embedding_x - embedding_w / 2
    add_arrow(ax, (input_arrow_start, flow_y), (input_arrow_end, flow_y))
    add_text(
        ax,
        (input_arrow_start + input_arrow_end) / 2,
        flow_y + 0.36,
        r"$H_t^{(b)}$",
        size=9.0,
        bold=True,
    )
    add_arrow(
        ax,
        (embedding_x + embedding_w / 2, flow_y),
        (transformer_x - transformer_w / 2, flow_y),
    )
    add_arrow(
        ax,
        (transformer_x + transformer_w / 2, flow_y),
        (readout_x - readout_w / 2, flow_y),
    )
    add_arrow(
        ax,
        (readout_x + readout_w / 2, flow_y),
        (feature_x - feature_w / 2, flow_y),
    )
def main():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.default": "bf",
            "axes.unicode_minus": False,
        }
    )
    figure, ax = plt.subplots(figsize=(16.5, 18.2), dpi=300)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0.2, 17.8)
    ax.set_ylim(1.55, 22.0)
    ax.axis("off")

    ax.plot([6.0, 6.0], [8.12, 21.72], color=SEPARATOR, linewidth=1.25, linestyle=(0, (4, 5)))
    ax.plot([12.0, 12.0], [8.12, 21.72], color=SEPARATOR, linewidth=1.25, linestyle=(0, (4, 5)))

    draw_actor_branch(ax, 3.1)
    draw_critic_branch(ax, 9.0)
    draw_discriminator_branch(ax, 14.9)
    add_common_module_definition(ax)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    figure.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(figure)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
