from pathlib import Path
import sys

BUNDLED_SITE_PACKAGES = Path(
    r"C:\Users\legion\.cache\codex-runtimes\codex-primary-runtime"
    r"\dependencies\python\Lib\site-packages"
)
if str(BUNDLED_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(BUNDLED_SITE_PACKAGES))

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "production_critic_network_structure.png"
SVG_PATH = OUTPUT_DIR / "production_critic_network_structure.svg"

LATIN_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
LATIN_BOLD_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf")

EDGE = "#34495E"
ARROW = "#465A6E"
STATE = "#E7F0F7"
ACTION = "#E9E3F2"
LINEAR = "#D7E9F4"
ACTIVATION = "#F4E8C9"
OUTPUT = "#DCEFD7"
BRANCH_BG = "#F8FAFC"


def add_text(ax, x, y, text, *, size=12.5, bold=False, ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontproperties=LATIN_BOLD_FONT if bold else LATIN_FONT,
        fontweight="bold" if bold else "normal",
        ha=ha,
        va=va,
        color="#111827",
        linespacing=1.18,
        zorder=8,
    )


def add_box(ax, x, y, width, height, text, *, color, size=11.8, radius=0.06):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=color,
        edgecolor=EDGE,
        linewidth=1.7,
        zorder=3,
    )
    ax.add_patch(patch)
    add_text(ax, x, y, text, size=size, bold=True)
    return patch


def add_arrow(ax, start, end, *, linewidth=1.7, mutation=12, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            linestyle=style,
            color=ARROW,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


BRANCH_SHIFT = 1.10
SPLIT_X = 5.00


def add_branch_panel(ax, y, label):
    panel = FancyBboxPatch(
        (4.05 + BRANCH_SHIFT, y - 1.02),
        10.65,
        2.04,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=BRANCH_BG,
        edgecolor="#718096",
        linewidth=1.45,
        linestyle=(0, (5, 4)),
        zorder=1,
    )
    ax.add_patch(panel)
    add_text(ax, 4.32 + BRANCH_SHIFT, y + 0.74, label, size=12.4, bold=True, ha="left")


def draw_q_branch(ax, y, index):
    add_branch_panel(ax, y, rf"Q{index} Branch")
    modules = [
        (5.15 + BRANCH_SHIFT, 1.55, "Linear Layer\n$37\\rightarrow128$", LINEAR, 11.6),
        (7.15 + BRANCH_SHIFT, 1.25, "Leaky\nReLU", ACTIVATION, 11.4),
        (9.10 + BRANCH_SHIFT, 1.55, "Linear Layer\n$128\\rightarrow32$", LINEAR, 11.6),
        (11.10 + BRANCH_SHIFT, 1.25, "Leaky\nReLU", ACTIVATION, 11.4),
        (13.00 + BRANCH_SHIFT, 1.45, "Output Layer\n$32\\rightarrow1$", OUTPUT, 11.5),
    ]
    previous_right = SPLIT_X
    for x, width, label, color, size in modules:
        left = x - width / 2
        add_arrow(ax, (previous_right, y), (left, y))
        add_box(ax, x, y, width, 0.90, label, color=color, size=size)
        previous_right = x + width / 2

    add_arrow(ax, (previous_right, y), (14.93 + BRANCH_SHIFT, y))
    output = Circle(
        (15.23 + BRANCH_SHIFT, y),
        radius=0.30,
        facecolor=OUTPUT,
        edgecolor=EDGE,
        linewidth=1.7,
        zorder=4,
    )
    ax.add_patch(output)
    add_text(
        ax,
        15.68 + BRANCH_SHIFT,
        y,
        rf"$Q_{index}(\hat{{\mathbf{{s}}}}_i,\mathbf{{a}}_i)$",
        size=12.2,
        bold=True,
        ha="left",
    )


def main():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(17.8, 6.75), dpi=300)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0.0, 18.75)
    axis.set_ylim(0.35, 7.05)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    add_text(axis, 1.15, 6.55, "Replay Mini-batch", size=13.5, bold=True)
    add_box(
        axis,
        1.15,
        5.05,
        2.00,
        1.02,
        "Standardized State\n" + r"$\hat{\mathbf{s}}_i\in\mathbb{R}^{33}$",
        color=STATE,
        size=11.5,
    )
    add_box(
        axis,
        1.15,
        2.25,
        2.00,
        1.02,
        "Action\n" + r"$\mathbf{a}_i\in\mathbb{R}^{4}$",
        color=ACTION,
        size=11.5,
    )

    add_box(
        axis,
        3.25,
        3.65,
        1.75,
        1.18,
        "Concatenated Pair\n" + r"$[\hat{\mathbf{s}}_i;\mathbf{a}_i]\in\mathbb{R}^{37}$",
        color="#F7E8C8",
        size=10.7,
    )
    pair_top = 3.65 + 1.18 / 2
    pair_bottom = 3.65 - 1.18 / 2
    add_arrow(axis, (2.15, 5.05), (3.25, pair_top))
    add_arrow(axis, (2.15, 2.25), (3.25, pair_bottom))
    add_arrow(axis, (4.13, 3.65), (SPLIT_X, 3.65))
    axis.scatter([SPLIT_X], [3.65], s=78, color="#64748B", zorder=7)
    axis.plot([SPLIT_X, SPLIT_X], [1.80, 5.50], color=ARROW, linewidth=1.7, zorder=5)

    draw_q_branch(axis, 5.50, 1)
    draw_q_branch(axis, 1.80, 2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    figure.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(figure)

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
