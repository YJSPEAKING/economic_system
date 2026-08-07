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
PNG_PATH = OUTPUT_DIR / "production_discriminator_network_structure.png"
SVG_PATH = OUTPUT_DIR / "production_discriminator_network_structure.svg"

LATIN_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
LATIN_BOLD_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf")

EDGE = "#34495E"
ARROW = "#465A6E"
EXPERT = "#DCEFD7"
GENERATED = "#F7DEDA"
INPUT = "#E7F0F7"
LINEAR = "#D7E9F4"
ACTIVATION = "#F4E8C9"
LOGIT = "#E8E1F2"
OUTPUT = "#DCEFD7"
LOSS = "#F5DDD1"
NETWORK_BG = "#F8FAFC"


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


def add_box(ax, x, y, width, height, text, *, color, size=11.6, radius=0.06):
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


def add_polyline_arrow(ax, points, *, linewidth=1.7, mutation=12, style="-"):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=ARROW,
            linewidth=linewidth,
            linestyle=style,
            zorder=5,
        )
    add_arrow(
        ax,
        points[-2],
        points[-1],
        linewidth=linewidth,
        mutation=mutation,
        style=style,
    )


def add_network_panel(ax):
    panel = FancyBboxPatch(
        (6.20, 1.25),
        9.00,
        4.90,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=NETWORK_BG,
        edgecolor="#718096",
        linewidth=1.55,
        linestyle=(0, (5, 4)),
        zorder=1,
    )
    ax.add_patch(panel)
    add_text(
        ax,
        6.50,
        5.82,
        r"Shared Discriminator Network $D_{\varphi}$",
        size=12.8,
        bold=True,
        ha="left",
    )

    modules = [
        (7.08, 1.36, "Linear Layer\n$37\\rightarrow100$", LINEAR, 10.9),
        (8.60, 0.92, r"$\tanh$", ACTIVATION, 12.0),
        (10.15, 1.40, "Linear Layer\n$100\\rightarrow100$", LINEAR, 10.9),
        (11.72, 0.92, r"$\tanh$", ACTIVATION, 12.0),
        (13.20, 1.30, "Logit Layer\n$100\\rightarrow1$", LOGIT, 10.8),
        (14.55, 0.96, "Sigmoid", ACTIVATION, 10.8),
    ]
    previous_right = None
    for x, width, label, color, size in modules:
        left = x - width / 2
        if previous_right is not None:
            add_arrow(ax, (previous_right, 3.68), (left, 3.68))
        add_box(ax, x, 3.68, width, 0.94, label, color=color, size=size)
        previous_right = x + width / 2


def main():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(18.0, 5.75), dpi=300)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0.0, 21.45)
    axis.set_ylim(0.35, 6.85)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    expert_y = 5.10
    generated_y = 2.15
    add_box(
        axis,
        1.40,
        expert_y,
        2.35,
        1.02,
        "Expert Sample\n" + r"$(\hat{\mathbf{s}}_i^{E},\hat{\mathbf{a}}_i^{E}),\;y_i=1$",
        color=EXPERT,
        size=11.3,
    )
    add_box(
        axis,
        1.40,
        generated_y,
        2.35,
        1.02,
        "Generated Sample\n" + r"$(\hat{\mathbf{s}}_i^{G},\hat{\mathbf{a}}_i^{G}),\;y_i=0$",
        color=GENERATED,
        size=11.3,
    )

    for y in (expert_y, generated_y):
        merge = Circle(
            (3.15, y),
            radius=0.40,
            facecolor="#F7E8C8",
            edgecolor=EDGE,
            linewidth=1.7,
            zorder=4,
        )
        axis.add_patch(merge)
        add_text(axis, 3.15, y, "Concat", size=8.9, bold=True)
        add_arrow(axis, (2.60, y), (2.75, y))

    add_arrow(axis, (3.55, expert_y), (3.87, expert_y))
    add_box(
        axis,
        4.87,
        expert_y,
        1.95,
        1.08,
        "Expert Pair\n"
        + r"$[\hat{\mathbf{s}}_i^{E};\hat{\mathbf{a}}_i^{E}]\in\mathbb{R}^{37}$",
        color=INPUT,
        size=9.7,
    )
    add_arrow(axis, (3.55, generated_y), (3.87, generated_y))
    add_box(
        axis,
        4.87,
        generated_y,
        1.95,
        1.08,
        "Generated Pair\n"
        + r"$[\hat{\mathbf{s}}_i^{G};\hat{\mathbf{a}}_i^{G}]\in\mathbb{R}^{37}$",
        color=INPUT,
        size=9.5,
    )

    add_network_panel(axis)
    add_arrow(axis, (5.87, expert_y), (6.20, expert_y))
    add_arrow(axis, (5.87, generated_y), (6.20, generated_y))

    add_arrow(axis, (15.20, expert_y), (16.02, expert_y))
    add_box(
        axis,
        17.15,
        expert_y,
        2.20,
        1.08,
        "Expert Score\n" + r"$d_i^{E}=D_{\varphi}(\hat{\mathbf{s}}_i^{E},\hat{\mathbf{a}}_i^{E})$",
        color=EXPERT,
        size=9.1,
    )
    add_arrow(axis, (15.20, generated_y), (16.02, generated_y))
    add_box(
        axis,
        17.15,
        generated_y,
        2.20,
        1.08,
        "Generated Score\n" + r"$d_i^{G}=D_{\varphi}(\hat{\mathbf{s}}_i^{G},\hat{\mathbf{a}}_i^{G})$",
        color=GENERATED,
        size=8.9,
    )

    add_box(
        axis,
        20.05,
        3.68,
        2.25,
        1.70,
        "Discriminator Loss\nBinary Cross-Entropy\n+ Entropy Regularization\n"
        + r"$\mathcal{L}_{D}$",
        color=LOSS,
        size=9.6,
    )
    add_arrow(axis, (18.27, expert_y), (18.93, 4.15))
    add_arrow(axis, (18.27, generated_y), (18.93, 3.21))

    add_polyline_arrow(
        axis,
        [(20.05, 2.83), (20.05, 0.62), (10.70, 0.62), (10.70, 1.25)],
        style=(0, (5, 4)),
    )
    add_text(axis, 15.35, 0.83, "Parameter Update", size=10.2, bold=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    figure.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(figure)

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
