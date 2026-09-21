"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt

INK, ACCENT, PALE, NEUTRAL, AMBER, GRAY = (
    "#1A1A1A",
    "#34688C",
    "#EAF1F6",
    "#F2F2F2",
    "#FDF0DC",
    "#5B6570",
)


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
        }
    )


def letter(ax, value, *, inside=False):
    if inside:
        ax.text(
            0.006,
            0.985,
            value,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
            color=INK,
        )
    else:
        ax.text(
            -0.005,
            1.03,
            value,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
            color=INK,
        )


def spectrogram_figure(atlas, spectra, panels, size, name):
    width, height = size
    fig = plt.figure(figsize=size)
    n = len(panels)
    right_pad = 0.085 if width >= 6 else 0.135
    left, gap = (0.065, 0.055)
    panel_w = (1 - left - right_pad - (n - 1) * gap) / n
    axes = []
    for i, (cid, label) in enumerate(panels):
        ax = fig.add_axes((left + i * (panel_w + gap), 0.17, panel_w, 0.68))
        spectrum, row = spectra[cid]
        ax.imshow(
            spectrum["db"],
            origin="lower",
            aspect="auto",
            extent=spectrum["extent"],
            interpolation="nearest",
            cmap="Greys",
            vmin=atlas.DB_MIN,
            vmax=0,
            rasterized=True,
        )
        ax.set_xlim(0, 0.7)
        ax.set_ylim(0, 16)
        ax.set_xticks([0, 0.35, 0.7])
        ax.set_xticklabels(["0", "0.35", "0.7"])
        ax.set_yticks([0, 8, 16])
        ax.tick_params(pad=1.5)
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=2)
        if i == 0:
            ax.set_ylabel("Frequency (kHz)", fontsize=7, labelpad=2)
        ax.set_title(label, loc="left", fontsize=6.5, pad=12)
        letter(ax, "abc"[i])
        axes.append(ax)
    cax = fig.add_axes((1 - right_pad + 0.015, 0.17, 0.014, 0.68))
    from matplotlib.colors import Normalize

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=Normalize(atlas.DB_MIN, 0), cmap="Greys"),
        cax=cax,
    )
    cbar.set_ticks([-70, -35, 0])
    cbar.ax.tick_params(labelsize=5.5, length=1.5, pad=1.5)
    cbar.set_label("Relative power (dB)", size=6, labelpad=2)
    return fig
