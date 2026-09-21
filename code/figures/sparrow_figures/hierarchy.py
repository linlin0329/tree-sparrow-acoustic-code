"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
from . import hierarchy_data as full
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .style import setup, font, card

W, H = (504, 276)


def main(metadata_dir):
    groups, rows = full.validated_data(metadata_dir)
    summaries = []
    for g in groups:
        v = sum(
            (
                r["terminal_role"] == "v1"
                for r in rows
                if r["structure"] == g["structure"]
            )
        )
        summaries.append(
            {
                "structure": g["structure"],
                "elements": g["elements"],
                "syllables": g["syllable_count"],
                "families": g["family_count"],
                "terminal_categories": g["leaf_count"],
                "main_categories": g["family_count"],
                "variant_categories": v,
            }
        )
    example = [r for r in rows if r["family_id"] == "single_group_08"]
    if not (
        len(example) == 2 and [r["terminal_role"] for r in example] == ["Main", "v1"]
    ):
        raise ValueError(
            "Figure input requirement failed: len(example) == 2 and [r['terminal_role'] for r in example] == ['Main', 'v1']"
        )
    setup()
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    geom = []

    def text(x, y, s, size=7.2, color="#293942", weight="normal", ha="center"):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontproperties=font(serif=weight == "bold"),
            color=color,
            ha=ha,
            va="center",
            zorder=4,
        )

    def box(x, y, w, h, face, edge, kind):
        card(ax, x - w / 2, y - h / 2, w, h, fill=face, edge=edge, radius=3.5)
        geom.append(
            {
                "kind": kind,
                "x": x - w / 2,
                "y": y - h / 2,
                "width": w,
                "height": h,
                "face": face,
                "edge": edge,
            }
        )

    def line(points, color="#A1AFB8", lw=0.7):
        xx, yy = zip(*points)
        ax.plot(xx, yy, color=color, lw=lw, zorder=2, solid_capstyle="round")

    xs = (150, 291, 432)
    text(42, 254, "Label\nhierarchy", 9, weight="bold")
    box(291, 254, 175, 25, "#2E4554", "#2E4554", "root")
    text(291, 254, "2,556 reviewed syllables", 9.4, color="white", weight="bold")
    line([(291, 241.5), (291, 228)])
    line([(xs[0], 228), (xs[-1], 228)])
    text(42, 207, "3", 10.5, weight="bold")
    text(42, 191, "structural\ncategories", 7.0, color="#647680")
    text(42, 157, "27", 10.5, weight="bold")
    text(42, 141, "type families", 7.0, color="#647680")
    text(42, 106, "41", 10.5, weight="bold")
    text(42, 89, "terminal\ncategories", 7.0, color="#647680")
    for x, g in zip(xs, summaries):
        c = full.COLORS[g["structure"]]
        t = full.TINTS[g["structure"]]
        bg = full.BANDS[g["structure"]]
        line([(x, 228), (x, 219)], c)
        box(x, 201, 123, 36, bg, c, "structure")
        text(x, 209, g["structure"].capitalize(), 10, weight="bold", color=c)
        text(
            x,
            194,
            f"{g['elements']} main element"
            + ("s" if g["elements"] > 1 else "")
            + f" · n = {g['syllables']:,}",
            6.8,
        )
        line([(x, 183), (x, 165)], c)
        box(x, 151, 123, 28, "white", c, "family_summary")
        text(x, 151, f"{g['families']} type families", 8.7, weight="bold", color=c)
        line([(x, 137), (x, 120.5)], c)
        box(x, 99, 123, 43, t, c, "terminal_summary")
        text(
            x,
            107,
            f"{g['terminal_categories']} terminal categories",
            8.2,
            weight="bold",
            color=c,
        )
        text(
            x,
            91,
            f"{g['main_categories']} main + {g['variant_categories']} variant types",
            6.8,
        )
    ax.add_patch(
        FancyBboxPatch(
            (89, 5),
            405,
            59,
            boxstyle="round,pad=0,rounding_size=4",
            facecolor="#F5F7F8",
            edgecolor="none",
            zorder=0,
        )
    )
    text(106, 43, "Example within Single", 7.1, weight="bold", ha="left")
    text(106, 25, "One family, two terminal types", 6.8, color="#647680", ha="left")
    box(284, 33, 65, 22, full.TINTS["single"], full.COLORS["single"], "example_family")
    text(284, 33, "Family 08", 7.6, weight="bold", color=full.COLORS["single"])
    line([(316.5, 33), (339, 33)], full.COLORS["single"])
    line([(339, 20), (339, 46)], full.COLORS["single"])
    for y, label, tint in [
        (46, "Main form", "white"),
        (20, "Variant (v1)", full.TINTS["single"]),
    ]:
        line([(339, y), (368, y)], full.COLORS["single"])
        box(429, y, 122, 18, tint, full.COLORS["single"], "example_terminal")
        text(429, y, label, 7.3)
    fig.canvas.draw()
    return fig
