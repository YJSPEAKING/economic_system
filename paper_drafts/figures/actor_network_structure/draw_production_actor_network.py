from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "production_actor_network_structure.png"
SVG_PATH = OUTPUT_DIR / "production_actor_network_structure.svg"

LATIN_FONT_PATH = r"C:\Windows\Fonts\times.ttf"
LATIN_BOLD_FONT_PATH = r"C:\Windows\Fonts\timesbd.ttf"

LATIN_FONT = font_manager.FontProperties(fname=LATIN_FONT_PATH)
LATIN_BOLD_FONT = font_manager.FontProperties(fname=LATIN_BOLD_FONT_PATH)


def add_text(ax, x, y, text, *, size=12, weight="normal", ha="center", va="center", color="#111111"):
    font = LATIN_BOLD_FONT if weight == "bold" else LATIN_FONT
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontproperties=font,
        ha=ha,
        va=va,
        color=color,
    )


def add_box(ax, x, y, width, height, *, facecolor, edgecolor="#333333", linewidth=1.0, radius=0.08):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def draw_nodes(ax, x, y_values, *, facecolor, edgecolor):
    ax.scatter(
        [x] * len(y_values),
        y_values,
        s=360,
        marker="o",
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=1.0,
        zorder=4,
    )


def connect_layers(ax, x1, ys1, x2, ys2, *, color="#50677D"):
    for y1 in ys1:
        for y2 in ys2:
            ax.plot(
                [x1 + 0.16, x2 - 0.16],
                [y1, y2],
                color=color,
                linewidth=0.72,
                alpha=0.48,
                zorder=2,
            )


def add_activation(ax, x, label):
    ax.text(
        x,
        5.32,
        label,
        fontsize=13.0,
        fontproperties=LATIN_BOLD_FONT,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111111",
        zorder=5,
    )
    ax.add_patch(
        FancyArrowPatch(
            (x, 5.08),
            (x, 4.72),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="#374151",
            zorder=5,
        )
    )


def main():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["mathtext.default"] = "bf"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14.2, 7.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 14.6)
    ax.set_ylim(0.0, 6.3)
    ax.axis("off")

    input_x = 1.7
    hidden1_x = 5.0
    hidden2_x = 8.3
    output_x = 11.6

    input_ys = [4.55, 3.95, 3.35, 2.15, 1.55, 0.95]
    hidden1_ys = [4.55, 3.95, 3.35, 2.15, 1.55, 0.95]
    hidden2_ys = [4.55, 3.95, 3.35, 2.15, 1.55, 0.95]
    output_ys = [4.25, 3.25, 2.25, 1.25]

    layer_box_y = 2.75
    layer_box_h = 4.25
    for x, width, color in (
        (input_x, 1.55, "#F1F4F7"),
        (hidden1_x, 1.55, "#EAF2F8"),
        (hidden2_x, 1.55, "#EAF2F8"),
        (output_x, 1.55, "#EAF6EE"),
    ):
        add_box(
            ax,
            x,
            layer_box_y,
            width,
            layer_box_h,
            facecolor=color,
            edgecolor="#374151",
            linewidth=1.1,
            radius=0.06,
        )

    connect_layers(ax, input_x, input_ys, hidden1_x, hidden1_ys)
    connect_layers(ax, hidden1_x, hidden1_ys, hidden2_x, hidden2_ys)
    connect_layers(ax, hidden2_x, hidden2_ys, output_x, output_ys)

    draw_nodes(ax, input_x, input_ys, facecolor="#D9E1E8", edgecolor="#455A64")
    draw_nodes(ax, hidden1_x, hidden1_ys, facecolor="#BFD7EA", edgecolor="#35698A")
    draw_nodes(ax, hidden2_x, hidden2_ys, facecolor="#9FC5E0", edgecolor="#35698A")
    draw_nodes(ax, output_x, output_ys, facecolor="#B8DCC3", edgecolor="#39734A")

    for x in (input_x, hidden1_x, hidden2_x):
        ax.scatter(
            [x, x, x],
            [2.58, 2.75, 2.92],
            s=7,
            marker="o",
            facecolors="#444444",
            edgecolors="none",
            zorder=5,
        )

    add_text(ax, input_x, 5.48, "Input Layer", size=13, weight="bold")
    ax.text(
        input_x,
        0.42,
        r"$\hat{\mathbf{s}}_t \in \mathbb{R}^{33}$",
        fontsize=13,
        fontproperties=LATIN_BOLD_FONT,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111111",
    )

    add_text(ax, hidden1_x, 5.48, "Hidden Layer 1", size=13, weight="bold")
    ax.text(
        hidden1_x,
        0.42,
        r"$\mathbf{h}^{(1)}_t \in \mathbb{R}^{128}$",
        fontsize=13,
        fontproperties=LATIN_BOLD_FONT,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111111",
    )

    add_text(ax, hidden2_x, 5.48, "Hidden Layer 2", size=13, weight="bold")
    ax.text(
        hidden2_x,
        0.42,
        r"$\mathbf{h}^{(2)}_t \in \mathbb{R}^{32}$",
        fontsize=13,
        fontproperties=LATIN_BOLD_FONT,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111111",
    )

    add_text(ax, output_x, 5.48, "Output Layer", size=13, weight="bold")
    ax.text(
        output_x,
        0.42,
        r"$\mathbf{a}_t \in [-0.5,\,0.5]^4$",
        fontsize=13,
        fontproperties=LATIN_BOLD_FONT,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111111",
    )

    add_activation(ax, 3.35, "tanh")
    add_activation(ax, 6.65, "Leaky ReLU")
    add_activation(ax, 9.95, r"tanh × $a_{\mathrm{max}}$")

    action_labels = [
        (4.25, "WNDF", "Loan Willingness"),
        (3.25, "K", "Production-Material\nPurchase Willingness"),
        (2.25, "L", "Consumer-Goods\nPurchase Willingness"),
        (1.25, "P", "Next-Day Pricing"),
    ]
    for y, symbol, description in action_labels:
        arrow = FancyArrowPatch(
            (output_x + 0.18, y),
            (12.88, y),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#374151",
            zorder=3,
        )
        ax.add_patch(arrow)
        ax.text(
            13.28,
            y,
            symbol,
            fontsize=10.5,
            fontproperties=LATIN_BOLD_FONT,
            fontweight="bold",
            ha="center",
            va="center",
            color="#111111",
            zorder=5,
        )
        add_text(ax, 13.28, y - 0.27, description, size=9.6, ha="center")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
