from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "critic_discriminator_network_structure.png"
SVG_PATH = OUTPUT_DIR / "critic_discriminator_network_structure.svg"

LATIN_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
LATIN_BOLD_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf")

EDGE = "#34495E"
ARROW = "#465A6E"
INPUT = "#E8F1F8"
HIDDEN = "#D9EAF5"
OUTPUT = "#DDEFD8"
PANEL = "#F8FAFC"


def add_text(ax, x, y, text, *, size=11.5, bold=False, ha="center", va="center"):
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
        zorder=6,
    )


def add_box(ax, x, y, width, height, text, *, color=HIDDEN, size=11.0):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=color,
        edgecolor=EDGE,
        linewidth=1.25,
        zorder=2,
    )
    ax.add_patch(box)
    add_text(ax, x, y, text, size=size, bold=True)
    return box


def add_arrow(ax, start, end, *, mutation=10, linewidth=1.25):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            color=ARROW,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def add_panel(ax, x, y, width, height, title):
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=PANEL,
        edgecolor="#64748B",
        linewidth=1.35,
        zorder=0,
    )
    ax.add_patch(panel)
    add_text(ax, x + width / 2, y + height - 0.38, title, size=14.0, bold=True)


def draw_critic(ax):
    add_panel(ax, 0.25, 0.35, 7.35, 6.45, "Critic (Twin Q Network)")

    add_box(ax, 1.25, 3.55, 1.55, 0.82, r"$[\hat{\mathbf{s}}_t,\,\mathbf{a}_t]\in\mathbb{R}^{37}$", color=INPUT, size=10.0)
    add_arrow(ax, (2.03, 3.55), (2.55, 3.55))

    split_x = 2.72
    ax.scatter([split_x], [3.55], s=85, color="#64748B", zorder=5)
    ax.plot([split_x, split_x], [2.25, 4.85], color=ARROW, linewidth=1.25, zorder=3)

    branch_y = (4.85, 2.25)
    for index, y in enumerate(branch_y, start=1):
        add_arrow(ax, (split_x, y), (3.18, y))
        add_box(ax, 3.92, y, 1.38, 0.78, "Linear\n37 → 128", size=10.7)
        add_arrow(ax, (4.63, y), (4.98, y))
        add_box(ax, 5.55, y, 1.06, 0.78, "Leaky\nReLU", color="#EEE7F6", size=10.2)
        add_arrow(ax, (6.08, y), (6.18, y))
        add_box(ax, 6.85, y, 1.32, 0.88, "Linear 128 → 32\nLeaky ReLU\nLinear 32 → 1", color=OUTPUT, size=9.1)
        add_text(ax, 6.85, y - 0.68, rf"$Q_{index}(\hat{{\mathbf{{s}}}}_t,\mathbf{{a}}_t)$", size=11.0, bold=True)

    add_text(
        ax,
        4.95,
        1.05,
        "The two Q branches have independent parameters.",
        size=10.5,
    )


def draw_discriminator(ax):
    add_panel(ax, 7.95, 0.35, 7.75, 6.45, "Discriminator")

    x_positions = [8.95, 10.65, 12.25, 13.75, 15.05]
    add_box(
        ax,
        x_positions[0],
        3.55,
        1.45,
        0.88,
        r"$[\hat{\mathbf{s}}_t,\,\hat{\mathbf{a}}_t]\in\mathbb{R}^{37}$",
        color=INPUT,
        size=10.6,
    )
    add_box(ax, x_positions[1], 3.55, 1.35, 0.88, "Linear\n37 → 100", size=10.7)
    add_box(ax, x_positions[2], 3.55, 1.30, 0.88, "Linear\n100 → 100", size=10.4)
    add_box(ax, x_positions[3], 3.55, 1.18, 0.88, "Linear\n100 → 1", color=OUTPUT, size=10.3)
    add_box(ax, x_positions[4], 3.55, 0.95, 0.88, "Sigmoid", color="#F9E5C8", size=10.5)

    half_widths = [0.73, 0.68, 0.65, 0.59, 0.48]
    for i in range(len(x_positions) - 1):
        start = (x_positions[i] + half_widths[i], 3.55)
        end = (x_positions[i + 1] - half_widths[i + 1], 3.55)
        add_arrow(ax, start, end)

    add_text(ax, 11.45, 4.20, "tanh", size=11.0, bold=True)
    add_text(ax, 13.00, 4.20, "tanh", size=11.0, bold=True)
    add_text(ax, 13.75, 2.87, r"Logit $f_{\varphi}(\hat{\mathbf{s}}_t,\hat{\mathbf{a}}_t)$", size=10.4)
    add_text(ax, 15.05, 2.87, r"$D_{\varphi}(\hat{\mathbf{s}}_t,\hat{\mathbf{a}}_t)$", size=10.8, bold=True)
    add_text(ax, 11.85, 1.12, "Standardized state-action input", size=10.5)


def main():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(16.0, 7.2), dpi=300)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 7.15)
    axis.axis("off")

    draw_critic(axis)
    draw_discriminator(axis)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    figure.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(figure)

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
