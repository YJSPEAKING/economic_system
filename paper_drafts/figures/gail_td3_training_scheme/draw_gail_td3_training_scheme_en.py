from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "gail_td3_training_scheme_en.png"
SVG_PATH = OUTPUT_DIR / "gail_td3_training_scheme_en.svg"

TIMES = font_manager.FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
TIMES_BOLD = font_manager.FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf")

EDGE = "#3E5F88"
NODE_FILL = "#DCE8F7"
DATA_FILL = "#D5E3F7"
ENV_FILL = "#F3F5F7"
FLOW = "#20252B"
UPDATE = "#5B6573"
FONT_BUMP = 1.2


def text(
    ax,
    x,
    y,
    value,
    *,
    size=11,
    bold=False,
    ha="center",
    va="center",
    color="#111111",
    background=False,
):
    ax.text(
        x,
        y,
        value,
        fontsize=size + FONT_BUMP,
        fontproperties=TIMES_BOLD,
        fontweight="bold",
        ha=ha,
        va=va,
        color=color,
        linespacing=1.05,
        bbox=(
            dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.96)
            if background
            else None
        ),
        zorder=8,
    )


def rounded_node(ax, center, width, height, label):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=NODE_FILL,
        edgecolor=EDGE,
        linewidth=1.65,
        zorder=4,
    )
    ax.add_patch(patch)
    text(ax, x, y, label, size=12)


def cylinder(ax, center, width, height, label):
    x, y = center
    ellipse_h = 0.30
    body = Rectangle(
        (x - width / 2, y - height / 2 + ellipse_h / 2),
        width,
        height - ellipse_h,
        facecolor=DATA_FILL,
        edgecolor=EDGE,
        linewidth=1.55,
        zorder=4,
    )
    bottom = Ellipse(
        (x, y - height / 2 + ellipse_h / 2),
        width,
        ellipse_h,
        facecolor=DATA_FILL,
        edgecolor=EDGE,
        linewidth=1.55,
        zorder=4,
    )
    top = Ellipse(
        (x, y + height / 2 - ellipse_h / 2),
        width,
        ellipse_h,
        facecolor="#E8F0FB",
        edgecolor=EDGE,
        linewidth=1.55,
        zorder=5,
    )
    ax.add_patch(body)
    ax.add_patch(bottom)
    ax.add_patch(top)
    text(ax, x, y, label, size=12)


def environment_node(ax, center, width, height):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.45",
        facecolor=ENV_FILL,
        edgecolor=FLOW,
        linewidth=1.55,
        zorder=4,
    )
    ax.add_patch(patch)
    text(ax, x, y, "Simulation\nEnvironment", size=11.5)


def expert_icon(ax, center):
    x, y = center
    head_y = [y + 0.42, y + 0.54, y + 0.42]
    head_x = [x - 0.34, x, x + 0.34]
    ax.scatter(
        head_x,
        head_y,
        s=95,
        marker="o",
        facecolors="#B9D2F3",
        edgecolors=FLOW,
        linewidths=1.35,
        zorder=5,
    )
    for hx, hy in zip(head_x, head_y):
        ax.plot([hx, hx], [hy - 0.16, hy - 0.66], color=FLOW, linewidth=2.15, zorder=4)
        ax.plot([hx - 0.15, hx + 0.15], [hy - 0.28, hy - 0.28], color=FLOW, linewidth=1.85, zorder=4)
        ax.plot([hx, hx - 0.12], [hy - 0.66, hy - 0.88], color=FLOW, linewidth=1.85, zorder=4)
        ax.plot([hx, hx + 0.12], [hy - 0.66, hy - 0.88], color=FLOW, linewidth=1.85, zorder=4)


def arrow(ax, start, end, *, dashed=False, connectionstyle="arc3", linewidth=1.55, color=None):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        linestyle=(0, (4, 3)) if dashed else "solid",
        color=color or (UPDATE if dashed else FLOW),
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=3,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(ax, points, *, dashed=False, color=None):
    line_color = color or (UPDATE if dashed else FLOW)
    style = (0, (4, 3)) if dashed else "solid"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=line_color,
            linewidth=1.55,
            linestyle=style,
            zorder=2.8,
        )
    arrow(ax, points[-2], points[-1], dashed=dashed, color=line_color)


def main():
    fig, ax = plt.subplots(figsize=(18.05, 7.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0.45, 20.15)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Main modules.
    expert_icon(ax, (2.96, 6.75))
    cylinder(ax, (6.90, 6.75), 1.60, 1.45, "Expert\nDataset")
    rounded_node(ax, (13.10, 6.75), 1.85, 1.32, "D\n(Discriminator)")

    rounded_node(ax, (1.57, 4.15), 1.80, 1.32, "Actor /\nGenerator")
    environment_node(ax, (5.79, 4.15), 1.95, 1.22)
    cylinder(ax, (10.00, 4.15), 1.70, 1.45, "Generated\nDataset")
    rounded_node(ax, (13.10, 1.55), 1.85, 1.32, "Critic\n(Value Network)")

    # Expert data path.
    arrow(ax, (3.61, 6.75), (6.10, 6.75))
    text(ax, 4.88, 7.08, "Data Collection", size=10.2, background=True)
    arrow(ax, (7.70, 6.75), (9.03, 6.75))
    text(ax, 10.00, 6.75, "<State, Action>\n(Expert)", size=10.5, background=True)
    arrow(ax, (10.97, 6.75), (12.18, 6.75))

    # Actor-environment-generated data path.
    arrow(ax, (2.47, 4.15), (4.81, 4.15))
    text(ax, 3.64, 4.48, "Generated Action", size=9.8, background=True)
    arrow(ax, (6.77, 4.15), (9.15, 4.15))
    text(ax, 7.96, 4.48, "Environment State", size=9.8, background=True)
    text(ax, 7.96, 3.82, "Environment Reward", size=9.8, background=True)

    routed_arrow(
        ax,
        [(3.64, 4.04), (3.64, 3.18), (10.00, 3.18), (10.00, 3.43)],
    )

    # Generated samples feed the discriminator, critic, and reward-fusion node.
    arrow(ax, (10.85, 4.15), (11.78, 4.15))
    arrow(ax, (14.42, 4.15), (18.68, 4.15))
    arrow(ax, (13.10, 4.15), (13.10, 6.09))
    arrow(ax, (13.10, 4.15), (13.10, 2.21))
    text(
        ax,
        13.10,
        4.15,
        "<State, Action, Reward>\n(Generated)",
        size=10.1,
        background=True,
    )

    # Discriminator output and reward fusion.
    arrow(ax, (14.03, 6.75), (15.00, 6.75))
    text(ax, 16.15, 6.75, "Discriminator Output", size=10.3, background=True)
    arrow(ax, (17.30, 6.75), (18.18, 6.75))
    text(ax, 19.00, 6.75, "Imitation Reward", size=10.3, background=True)
    routed_arrow(ax, [(19.00, 6.46), (19.00, 4.15)])

    # Discriminator parameter update marker.
    routed_arrow(
        ax,
        [(16.15, 6.75), (16.15, 7.58), (13.10, 7.58), (13.10, 7.41)],
        dashed=True,
    )
    text(ax, 14.63, 7.84, "Update", size=10.2, color=UPDATE)

    # Environment reward joins imitation reward.
    text(ax, 16.55, 4.49, "Environment Reward", size=10.5, background=True)
    ax.scatter(
        [19.00],
        [4.15],
        s=520,
        marker="o",
        facecolors="white",
        edgecolors=FLOW,
        linewidths=1.55,
        zorder=7,
    )
    text(ax, 19.00, 4.15, "+", size=18)

    # Target and estimated value comparison.
    arrow(ax, (19.00, 3.86), (19.00, 3.08))
    text(ax, 19.00, 2.86, "Target Q-Value", size=10.5, background=True)
    arrow(ax, (19.00, 2.64), (19.00, 1.86))

    routed_arrow(ax, [(13.10, 0.89), (13.10, 0.48), (14.92, 0.48)])
    text(ax, 16.05, 0.48, "Estimated Q-Value", size=10.5, background=True)
    routed_arrow(ax, [(17.18, 0.48), (19.00, 0.48), (19.00, 1.24)])

    ax.scatter(
        [19.00],
        [1.55],
        s=620,
        marker="^",
        facecolors="white",
        edgecolors=FLOW,
        linewidths=1.55,
        zorder=7,
    )

    # Critic and actor parameter updates.
    arrow(ax, (18.70, 1.55), (14.03, 1.55), dashed=True)
    text(ax, 16.32, 1.82, "Update Critic", size=10.2, color=UPDATE)
    routed_arrow(
        ax,
        [(12.18, 1.55), (1.57, 1.55), (1.57, 3.49)],
        dashed=True,
    )
    text(ax, 6.88, 1.82, "Update Actor", size=10.2, color=UPDATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
