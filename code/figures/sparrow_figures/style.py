"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
from .resources import ASSETS
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyBboxPatch

SERIF_FILE = ASSETS / "fonts/DatasetCormorantLining-Regular.ttf"
SANS_FILE = ASSETS / "fonts/Lato-Regular.ttf"
INK = "#3D4B4E"
DARK = "#104C59"
BLUE = "#155B67"
PALE = "#FFFFFF"
TINT = "#EDF5F7"
GRAY = "#77878C"
EDGE = "#D8D7CD"
GOLD = "#CE913C"
LINE = "#9BACB3"
WIDTH_MM = 152


def setup():
    for font in [SERIF_FILE, SANS_FILE]:
        fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Lato"],
            "font.size": 6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.7,
        }
    )


def font(serif=False):
    return FontProperties(fname=str(SERIF_FILE if serif else SANS_FILE))


def text(
    ax, x, y, value, size=6, color=INK, serif=False, ha="center", va="center", **kwargs
):
    if not size >= 5:
        raise ValueError("Figure input requirement failed: size >= 5")
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontproperties=font(serif),
        color=color,
        ha=ha,
        va=va,
        linespacing=1.2,
        **kwargs,
    )


def card(
    ax, x, y, width, height, fill=PALE, edge=EDGE, radius=3.5, shadow=True, zorder=3
):
    """Subtle shadow is drawn with vector patches; no text is rasterized."""
    if shadow:
        for dx, dy, alpha, expand in [
            (0, -1.5, 0.012, 1.5),
            (0, -1, 0.02, 0.8),
            (0, -0.5, 0.035, 0.2),
        ]:
            ax.add_patch(
                FancyBboxPatch(
                    (x + dx - expand, y + dy - expand),
                    width + 2 * expand,
                    height + 2 * expand,
                    boxstyle=f"round,pad=0,rounding_size={radius + expand}",
                    facecolor="#647078",
                    edgecolor="none",
                    alpha=alpha,
                    zorder=zorder - 1,
                )
            )
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=0.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel_heading(ax, x, y, letter, title, size=9.2):
    card(ax, x, y - 5.4, 10.8, 10.8, fill=GOLD, edge=GOLD, radius=3, shadow=False)
    text(ax, x + 5.4, y, letter, size=8.5, color="white", serif=True, zorder=4)
    text(ax, x + 15.5, y, title, size=size, serif=True, ha="left", zorder=4)
