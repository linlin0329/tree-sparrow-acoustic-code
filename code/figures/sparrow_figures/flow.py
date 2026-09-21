"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from .style import INK, DARK, GRAY, EDGE, LINE, WIDTH_MM, text, card, panel_heading

GEOMETRY = {"panels": {}, "nodes": [], "arrows": []}
PX = WIDTH_MM / 25.4 * 72 / 1792
HEIGHT_MM = WIDTH_MM * 1400 / 1792
PANEL_BOUNDS = {"a": (50, 375), "b": (435, 945), "c": (1005, 1330)}
CARD_X = (0, 608, 1216)
CARD_WIDTH = 420


def axis(fig, panel):
    top, bottom = PANEL_BOUNDS[panel]
    w, h = fig.get_size_inches() * 72
    rect = [78 * PX, h - bottom * PX, 1636 * PX, (bottom - top) * PX]
    ax = fig.add_axes([rect[0] / w, rect[1] / h, rect[2] / w, rect[3] / h])
    ax.set_xlim(0, rect[2])
    ax.set_ylim(0, rect[3])
    ax.axis("off")
    ax.set_label(panel)
    GEOMETRY["panels"][panel] = dict(
        rect_pt=rect, reference_top_bottom_px=[top, bottom]
    )
    return ax


def xy(ax, x, y):
    return (x * PX, (PANEL_BOUNDS[ax.get_label()][1] - y) * PX)


def label(ax, x, y, value, size=5.8, color=INK, serif=False, ha="center"):
    xx, yy = xy(ax, x, y)
    return text(ax, xx, yy, value, size=size, color=color, serif=serif, ha=ha, zorder=5)


def box(ax, node, column, top, bottom, body, subtitle=None, number=None, output=False):
    x, y = xy(ax, CARD_X[column], bottom)
    width, height = (CARD_WIDTH * PX, (bottom - top) * PX)
    card(
        ax,
        x,
        y,
        width,
        height,
        fill=DARK if output else "white",
        edge=DARK if output else EDGE,
        radius=3.2,
        shadow=True,
    )
    center_x = CARD_X[column] + CARD_WIDTH / 2
    color = "white" if output else INK
    muted = "#C0D3D7" if output else GRAY
    if number is not None:
        label(
            ax,
            center_x,
            top + (bottom - top) * 0.38,
            f"{number:,}",
            size=15.8,
            serif=True,
            color="white" if output else DARK,
        )
        label(ax, center_x, top + (bottom - top) * 0.75, body, size=5.8, color=muted)
    elif subtitle:
        label(ax, center_x, top + (bottom - top) * 0.38, body, size=6.2, color=color)
        label(
            ax, center_x, top + (bottom - top) * 0.68, subtitle, size=5.4, color=muted
        )
    else:
        label(ax, center_x, (top + bottom) / 2, body, size=6.2, color=color)
    GEOMETRY["nodes"].append(
        dict(
            panel=ax.get_label(),
            id=node,
            rect_pt=[x, y, width, height],
            column=column,
            text=body,
            subtitle=subtitle,
            number=number,
            output=output,
        )
    )


def arrow(ax, edge, points, dashed=False):
    points_pt = [xy(ax, *p) for p in points]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    patch = FancyArrowPatch(
        path=MplPath(points_pt, codes),
        arrowstyle="->",
        mutation_scale=6.3,
        linewidth=0.55,
        color="#B7AD96" if dashed else LINE,
        linestyle=(0, (2.4, 1.7)) if dashed else "-",
        zorder=2,
        capstyle="round",
        joinstyle="miter",
    )
    ax.add_patch(patch)
    GEOMETRY["arrows"].append(
        dict(
            panel=ax.get_label(),
            id=edge,
            points_pt=points_pt,
            meaning="training support" if dashed else "record flow",
            dashed=dashed,
        )
    )


def build(counts):
    GEOMETRY.clear()
    GEOMETRY.update({"panels": {}, "nodes": [], "arrows": []})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    w, h = fig.get_size_inches() * 72
    a, b, c = [axis(fig, k) for k in "abc"]
    for ax, y, letter, title in [
        (a, 80, "a", "Raw-recording release"),
        (b, 462, "b", "Song discovery and manual confirmation"),
        (c, 1034, "c", "High-quality excerpts and syllable annotations"),
    ]:
        xx, yy = xy(ax, 0, y)
        panel_heading(ax, xx, yy, letter, title, size=9.2)
    box(
        a,
        "archive",
        0,
        162,
        297,
        "Deduplicated recordings",
        number=counts["logical_archive"],
    )
    box(
        a,
        "candidates",
        1,
        162,
        297,
        "Candidate recordings",
        number=counts["raw_candidates"],
    )
    box(
        a,
        "released_raw",
        2,
        162,
        297,
        "MP3 release set",
        number=counts["raw_admitted"],
        output=True,
    )
    arrow(a, "score_screen", [(434, 230), (598, 230)])
    label(a, 514, 137, "Max. target score ≥ 0.7", size=5.3, color=GRAY)
    arrow(a, "technical_check", [(1042, 230), (1206, 230)])
    label(a, 1124, 137, "Technical checks", size=5.3, color=GRAY)
    label(a, 1124, 255, f"{counts['raw_held']} excluded", size=5.3, color="#9B865D")
    box(b, "initial_listening", 0, 521, 638, "Initial listening", "Reference examples")
    box(b, "initial_songs", 1, 521, 638, "Songs confirmed", "during initial listening")
    arrow(b, "initial_confirmation", [(434, 580), (598, 580)])
    box(b, "retrieval", 0, 751, 868, "Assisted retrieval", "(including songiness)")
    box(b, "manual_confirmation", 1, 751, 868, "Manual confirmation")
    arrow(b, "training_examples", [(210, 645), (210, 742)], dashed=True)
    label(b, 227, 696, "Training examples", size=5.3, color="#988C73", ha="left")
    arrow(b, "review_candidates", [(434, 810), (598, 810)])
    box(
        b,
        "merge_eligibility",
        2,
        641,
        754,
        "Merge and deduplicate",
        "Check release eligibility",
    )
    arrow(
        b,
        "initial_songs_to_merge",
        [(1028, 580), (1122, 580), (1122, 669), (1216, 669)],
    )
    arrow(
        b,
        "reviewed_songs_to_merge",
        [(1028, 810), (1122, 810), (1122, 725), (1216, 725)],
    )
    box(
        b,
        "song_index",
        2,
        801,
        935,
        "Confirmed song recordings",
        number=counts["songs"],
        output=True,
    )
    arrow(b, "release_eligibility_to_index", [(1426, 761), (1426, 796)])
    box(c, "core_raw", 0, 1093, 1227, "Original recordings", number=counts["core_raw"])
    box(
        c,
        "analysis_clips",
        1,
        1093,
        1227,
        "Analysis WAV excerpts",
        number=counts["clips"],
    )
    box(
        c,
        "syllables",
        2,
        1093,
        1227,
        "Labelled syllables",
        number=counts["core_syllables"],
        output=True,
    )
    arrow(c, "source_to_excerpts", [(434, 1159), (598, 1159)])
    arrow(c, "excerpts_to_syllables", [(1042, 1159), (1206, 1159)])
    footer = f"{counts['raven_tables']} matching Raven tables  ·  {counts['families']} families  ·  {counts['leaves']} terminal categories"
    footer_artist = label(c, 818, 1282, footer, size=5.3, color=GRAY)
    fig.canvas.draw()
    footer_width_pt = (
        footer_artist.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    )
    footer_center_pt = 818 * PX
    for xx1, xx2 in [
        (174 * PX, footer_center_pt - footer_width_pt / 2 - 6),
        (footer_center_pt + footer_width_pt / 2 + 6, 1462 * PX),
    ]:
        if not xx2 > xx1:
            raise ValueError("Figure input requirement failed: xx2 > xx1")
        c.plot([xx1, xx2], [(1330 - 1282) * PX] * 2, lw=0.45, color=EDGE, zorder=1)
    decor = fig.add_axes([0, 0, 1, 1], label="decorative_separators", zorder=-1)
    decor.set_xlim(0, w)
    decor.set_ylim(0, h)
    decor.axis("off")
    for y in [410, 982]:
        decor.plot([78 * PX, 1714 * PX], [h - y * PX] * 2, lw=0.45, color=EDGE)
    groups = [
        ["archive", "initial_listening", "retrieval", "core_raw"],
        ["candidates", "initial_songs", "manual_confirmation", "analysis_clips"],
        ["released_raw", "merge_eligibility", "song_index", "syllables"],
    ]
    lookup = {n["id"]: n for n in GEOMETRY["nodes"]}
    if not len(lookup) == 12:
        raise ValueError("Figure input requirement failed: len(lookup) == 12")
    for col, ids in enumerate(groups):
        if not all(
            (
                lookup[i]["rect_pt"][0] == CARD_X[col] * PX
                and lookup[i]["rect_pt"][2] == CARD_WIDTH * PX
                for i in ids
            )
        ):
            raise ValueError(
                "Figure input requirement failed: all((lookup[i]['rect_pt'][0] == CARD_X[col] * PX and lookup[i]['rect_pt'][2] == CARD_WIDTH * PX for i in ids))"
            )
    if not "excluded" not in lookup:
        raise ValueError("Figure input requirement failed: 'excluded' not in lookup")
    if not "quality_exclusion" not in {e["id"] for e in GEOMETRY["arrows"]}:
        raise ValueError(
            "Figure input requirement failed: 'quality_exclusion' not in {e['id'] for e in GEOMETRY['arrows']}"
        )
    GEOMETRY.update(
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
        archetype="schematic-led composite",
        no_between_panel_arrows=True,
        minimum_requested_font_pt=5.3,
        main_text_font_pt=6.2,
        node_number_font_pt=15.8,
        reference_reuse="author-provided layout and colour reference, independently redrawn as vectors",
        raster_image_count=0,
        column_alignment={
            "passed": True,
            "columns": groups,
            "x_positions_pt": [x * PX for x in CARD_X],
            "shared_node_width_pt": CARD_WIDTH * PX,
            "equal_column_gaps_pt": (608 - CARD_WIDTH) * PX,
        },
        exclusion_annotation={
            "text": "33 excluded",
            "point_pt": xy(a, 1124, 255),
            "font_pt": 5.3,
            "parent_step": "technical_check",
            "separate_box_or_arrow": False,
        },
        clock_annotation_displayed=False,
        decorative_separators_excluded_from_comparable_panel_axes=True,
    )
    fig.canvas.draw()
    return fig
