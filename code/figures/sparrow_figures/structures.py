"""Twenty-seven selected main-form syllables with common display scales.

The three rows are a compact catalogue, not a continuous vocal sequence.
All spectra retain their original sample-derived time/frequency coordinates.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

FLOOR_DB, CEILING_DB = -45.0, 0.0
XMAX = 0.7
INK = "#303B42"
WIDTH_MM = 180.0
LEFT_MM, RIGHT_MM, GUTTER_MM = 12.0, 4.0, 2.0
NROWS, NCOLS = 3, 9
ASPECT = 1.974375 / 1.647
CELL_WIDTH_MM = (WIDTH_MM - LEFT_MM - RIGHT_MM - (NCOLS - 1) * GUTTER_MM) / NCOLS
CELL_HEIGHT_MM = CELL_WIDTH_MM * ASPECT
ROW_GAP_MM = 6.2
TOP_MM, FOOTER_MM = 3.5, 18.5
HEIGHT_MM = TOP_MM + NROWS * CELL_HEIGHT_MM + (NROWS - 1) * ROW_GAP_MM + FOOTER_MM
EXPECTED_FAMILIES = tuple(
    f"{structure}_group_{index:02d}"
    for structure, count in (("single", 14), ("double", 6), ("triple", 7))
    for index in range(1, count + 1)
)


def short_label(family):
    prefix = {"single": "S", "double": "D", "triple": "T"}[family.split("_")[0]]
    return prefix + family.rsplit("_", 1)[-1]


def draw(selected, spectra):
    """Draw the accepted nine-column plate from ordered, validated inputs."""
    selected = list(selected)
    if tuple(row["family"] for row in selected) != EXPECTED_FAMILIES:
        raise ValueError("Figure 5 requires all 27 families in Single/Double/Triple order")
    if len({row["stable_id"] for row in selected}) != 27:
        raise ValueError("Figure 5 requires one distinct selected syllable per family")
    if any(row["stable_id"] not in spectra for row in selected):
        raise ValueError("Figure 5 is missing a selected spectrogram")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.0,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "axes.unicode_minus": True,
        }
    )
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))

    def rect(x, y, width, height):
        return [x / WIDTH_MM, y / HEIGHT_MM, width / WIDTH_MM, height / HEIGHT_MM]

    def text(x, y, value, **kwargs):
        kwargs.setdefault("color", INK)
        return fig.text(x / WIDTH_MM, y / HEIGHT_MM, value, **kwargs)

    tops = [
        HEIGHT_MM - TOP_MM - index * (CELL_HEIGHT_MM + ROW_GAP_MM)
        for index in range(NROWS)
    ]
    for index, row in enumerate(selected):
        row_index, column_index = divmod(index, NCOLS)
        left = LEFT_MM + column_index * (CELL_WIDTH_MM + GUTTER_MM)
        bottom = tops[row_index] - CELL_HEIGHT_MM
        label = short_label(row["family"])
        axis = fig.add_axes(rect(left, bottom, CELL_WIDTH_MM, CELL_HEIGHT_MM), label=label)
        spectrum = spectra[row["stable_id"]]
        axis.imshow(
            spectrum["db"],
            origin="lower",
            aspect="auto",
            extent=spectrum["extent"],
            interpolation="nearest",
            cmap="Greys",
            vmin=FLOOR_DB,
            vmax=CEILING_DB,
            rasterized=True,
        )
        axis.set_xlim(0, XMAX)
        axis.set_ylim(0, 16)
        axis.set_xticks([])
        axis.set_yticks([0, 8, 16] if column_index == 0 else [])
        axis.tick_params(
            axis="y", labelsize=6.5, pad=2.5, length=2,
            color="#626B71", labelcolor=INK,
        )
        for edge, spine in axis.spines.items():
            spine.set_visible(column_index == 0 and edge == "left")
            spine.set_color("#626B71")
        text(
            left + 0.5, bottom - 1.0, label,
            fontsize=6.3, fontweight="bold", va="top", ha="left",
        )

    # These row baselines do not imply continuous timing between examples.
    for row_index, top in enumerate(tops):
        bottom = top - CELL_HEIGHT_MM
        baseline = Line2D(
            [LEFT_MM / WIDTH_MM, (WIDTH_MM - RIGHT_MM) / WIDTH_MM],
            [bottom / HEIGHT_MM] * 2,
            transform=fig.transFigure, color="#626B71", lw=0.5,
            solid_capstyle="butt", gid=f"row_baseline_{row_index + 1}",
        )
        fig.add_artist(baseline)

    data_bottom = tops[-1] - CELL_HEIGHT_MM
    data_center = (tops[0] + data_bottom) / 2
    text(
        2, data_center, "Frequency (kHz)", fontsize=7.0,
        rotation=90, rotation_mode="anchor", ha="center", va="center",
    )

    # Shared references stay outside all data panels. The time bar is measured
    # on the same physical scale as the 0–0.7 s panel axes.
    footer_y = 6.0
    time_center_x = 60.0
    bar_width = CELL_WIDTH_MM * 0.4 / XMAX
    time_bar = Line2D(
        [(time_center_x - bar_width / 2) / WIDTH_MM,
         (time_center_x + bar_width / 2) / WIDTH_MM],
        [footer_y / HEIGHT_MM] * 2,
        transform=fig.transFigure, color=INK, lw=1.15,
        solid_capstyle="butt", gid="shared_time_scale",
    )
    fig.add_artist(time_bar)
    text(time_center_x, 2.2, "0.4 s", fontsize=6.8, ha="center", va="bottom")
    text(time_center_x, 9.6, "Time", fontsize=6.5, ha="center", va="bottom")
    colorbar_x, colorbar_width, colorbar_height = 99.0, 32.0, 1.8
    colorbar_axis = fig.add_axes(
        rect(colorbar_x, footer_y - colorbar_height / 2, colorbar_width, colorbar_height),
        label="shared_power_scale",
    )
    colorbar = ColorbarBase(
        colorbar_axis, cmap="Greys", norm=Normalize(FLOOR_DB, CEILING_DB),
        orientation="horizontal", ticks=[-45, -20, 0],
    )
    colorbar.ax.tick_params(labelsize=6.5, pad=1.8, length=1.8)
    colorbar.outline.set_linewidth(0.45)
    text(
        colorbar_x + colorbar_width / 2, 9.6, "Relative power (dB)",
        fontsize=6.5, ha="center", va="bottom",
    )
    fig.canvas.draw()
    return fig
