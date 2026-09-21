"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

FLOOR_DB, CEILING_DB = (-45.0, 0.0)
STRUCTURES = ("single", "double", "triple")
XMAX = 0.5
INK = "#3D4B4E"


def draw(selected, spectra):
    width, height = (180, 132)
    columns = 4
    gap = 7
    left, right, panel_height = (13, 20, 23.5)
    panel_width = (width - left - right - gap * (columns - 1)) / columns
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    axes, panel_ids, row_groups = ([], [], [])
    for row_index, (structure, bottom) in enumerate(zip(STRUCTURES, [93, 52, 11])):
        row = [r for r in selected if r["structure"] == structure]
        if not len(row) == columns:
            raise ValueError("Figure input requirement failed: len(row) == columns")
        heading = structure.title()
        heading += (
            " ("
            + ["one main element", "two main elements", "three main elements"][
                row_index
            ]
            + ")"
        )
        fig.text(
            left / width,
            (bottom + 32.5) / height,
            heading,
            ha="left",
            va="baseline",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
        row_ids = []
        for col_index, c in enumerate(row):
            panel = chr(97 + len(axes))
            row_ids.append(panel)
            panel_ids.append(panel)
            ax = fig.add_axes(
                [
                    (left + col_index * (panel_width + gap)) / width,
                    bottom / height,
                    panel_width / width,
                    panel_height / height,
                ]
            )
            spec = spectra[c["stable_id"]]
            ax.imshow(
                spec["db"],
                extent=spec["extent"],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="Greys",
                vmin=FLOOR_DB,
                vmax=CEILING_DB,
                rasterized=True,
            )
            ax.set_xlim(0, XMAX)
            ax.set_ylim(0, 16)
            ax.set_xticks([0, 0.25, 0.5])
            ax.set_xticklabels(["0", "0.25", "0.5"])
            ax.set_yticks([0, 8, 16])
            ax.tick_params(length=2.2, pad=1.5, labelsize=7)
            if col_index == 0:
                ax.set_ylabel("Frequency (kHz)", fontsize=7.5, labelpad=2)
            if row_index == 2:
                ax.set_xlabel("Time (s)", fontsize=7.5, labelpad=2)
            label = "Family " + c["family"].rsplit("_", 1)[-1]
            ax.text(
                0,
                1.085,
                panel,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                fontweight="bold",
                color=INK,
            )
            ax.text(
                0.14,
                1.085,
                label,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=7.5,
                color=INK,
            )
            axes.append(ax)
        row_groups.append(row_ids)
        cax = fig.add_axes(
            [(width - 16) / width, bottom / height, 2 / width, panel_height / height]
        )
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=Normalize(FLOOR_DB, CEILING_DB), cmap="Greys"
            ),
            cax=cax,
        )
        middle = 5 * round((FLOOR_DB + CEILING_DB) / 10)
        ticks = [FLOOR_DB, middle, CEILING_DB]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{v:g}".replace("-", "−") for v in ticks])
        cb.ax.tick_params(labelsize=6.5, pad=1.5, length=1.8)
        cb.set_label("Relative power (dB)", fontsize=7, labelpad=2)
    fig.canvas.draw()
    return fig
