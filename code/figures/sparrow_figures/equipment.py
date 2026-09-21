"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
from .resources import ASSETS
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image
from .style import (
    DARK,
    TINT,
    GRAY,
    EDGE,
    LINE,
    WIDTH_MM,
    setup,
    text,
    card,
    panel_heading,
)

PHOTO = ASSETS / "equipment.jpeg"
HEIGHT_MM = 62
WIDTH_PT = WIDTH_MM / 25.4 * 72
HEIGHT_PT = HEIGHT_MM / 25.4 * 72
CROP = (0, 60, 531, 708)
PHOTO_X, PHOTO_Y, PHOTO_W = (15, 10, 114)
PHOTO_H = PHOTO_W * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0])


def axes_pt(fig, x, width):
    ax = fig.add_axes([x / WIDTH_PT, 0, width / WIDTH_PT, 1])
    ax.set_xlim(x, x + width)
    ax.set_ylim(0, HEIGHT_PT)
    ax.set_axis_off()
    return ax


def number(ax, x, y, label, radius=4.7, size=6.6):
    ax.add_patch(
        Circle(
            (x, y), radius, facecolor="white", edgecolor=DARK, linewidth=0.65, zorder=7
        )
    )
    text(ax, x, y - 0.25, label, size=size, color=DARK, serif=True, zorder=8)


def component(ax, x, y, number_label, title, first, second, fill="white"):
    width, height = (76, 43)
    card(ax, x, y, width, height, fill=fill, radius=3.5)
    cx = x + width / 2
    number(ax, cx, y + 35.6, number_label, radius=4.6, size=6.4)
    text(ax, cx, y + 24.8, title, size=6.25, zorder=5)
    text(ax, cx, y + 16.4, first, size=5.35, color=GRAY, zorder=5)
    text(ax, cx, y + 8.3, second, size=5.35, color=GRAY, zorder=5)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=7,
            linewidth=0.7,
            color=LINE,
            linestyle=(0, (3.3, 2.0)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def main():
    setup()
    original = Image.open(PHOTO)
    if not original.size == (531, 708):
        raise ValueError("Figure input requirement failed: original.size == (531, 708)")
    photograph = original.crop(CROP)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=400)
    photo_ax = axes_pt(fig, 15, 129)
    diagram_ax = axes_pt(fig, 152, 267)
    panel_heading(photo_ax, 15, 162, "a", "Field recorder", size=10)
    panel_heading(diagram_ax, 152, 162, "b", "Main connections", size=10)
    for ax, title in [(photo_ax, "Field recorder"), (diagram_ax, "Main connections")]:
        label = next((t for t in ax.texts if t.get_text() == title))
        label.set_x(label.get_position()[0] + 2.5)
    card(photo_ax, PHOTO_X, PHOTO_Y, PHOTO_W, PHOTO_H, radius=6, zorder=1)
    raster = photo_ax.imshow(
        photograph,
        extent=(PHOTO_X, PHOTO_X + PHOTO_W, PHOTO_Y, PHOTO_Y + PHOTO_H),
        interpolation="none",
        aspect="equal",
        zorder=2,
    )
    clip = FancyBboxPatch(
        (PHOTO_X, PHOTO_Y),
        PHOTO_W,
        PHOTO_H,
        boxstyle="round,pad=0,rounding_size=6",
        facecolor="none",
        edgecolor=EDGE,
        linewidth=0.5,
        zorder=3,
    )
    photo_ax.add_patch(clip)
    raster.set_clip_path(clip)
    callouts = [
        {"number": "1", "pixel_xy_original": [219, 339], "label_xy_pt": [138.5, 84]},
        {"number": "2", "pixel_xy_original": [396, 451], "label_xy_pt": [138.5, 68]},
        {"number": "3", "pixel_xy_original": [329, 548], "label_xy_pt": [138.5, 46]},
        {"number": "4", "pixel_xy_original": [268, 697], "label_xy_pt": [138.5, 23]},
    ]
    for item in callouts:
        px, py = item["pixel_xy_original"]
        point = (PHOTO_X + px / 531 * PHOTO_W, PHOTO_Y + (708 - py) / 531 * PHOTO_W)
        lx, ly = item["label_xy_pt"]
        (line,) = photo_ax.plot(
            [point[0], lx - 4.7],
            [point[1], ly],
            lw=0.65,
            color=DARK,
            solid_capstyle="round",
            zorder=6,
        )
        line.set_path_effects(
            [pe.Stroke(linewidth=1.55, foreground="white"), pe.Normal()]
        )
        photo_ax.add_patch(
            Circle(
                point, 1.15, facecolor=DARK, edgecolor="white", linewidth=0.8, zorder=7
            )
        )
        number(photo_ax, lx, ly, item["number"])
    component(
        diagram_ax, 152, 103, "4", "Microphone", "BOYA BY-M110", "Omnidirectional"
    )
    component(
        diagram_ax, 247.5, 103, "3", "USB interface", "UGREEN CM477", "Audio input"
    )
    component(
        diagram_ax,
        343,
        103,
        "2",
        "Raspberry Pi",
        "4B · 4 GB RAM",
        "BirdNET-Pi",
        fill=TINT,
    )
    for start, end, label in [(228, 247.5, "3.5 mm"), (323.5, 343, "USB")]:
        arrow(diagram_ax, (start + 2.5, 124.5), (end - 2.5, 124.5))
        text(diagram_ax, (start + end) / 2, 133, label, size=5.2, color=GRAY, zorder=5)
    component(diagram_ax, 343, 45, "1", "Battery", "60,000 mAh", "(nominal)")
    arrow(diagram_ax, (381, 89), (381, 102), dashed=True)
    text(diagram_ax, 377, 95.5, "Power", size=5.35, ha="right", color=GRAY, zorder=5)
    card(diagram_ax, 152, 16, 267, 20, fill="#F2F7F7", shadow=False, radius=4)
    diagram_ax.plot([237, 237], [21.5, 30.5], color=EDGE, linewidth=0.55, zorder=4)
    text(
        diagram_ax,
        161,
        26,
        "WAV acquisition",
        size=7.25,
        serif=True,
        ha="left",
        color=DARK,
        zorder=5,
    )
    text(
        diagram_ax,
        247,
        26,
        "48 kHz · 16-bit · Mono · 30 s",
        size=5.8,
        ha="left",
        color=GRAY,
        zorder=5,
    )
    fig.canvas.draw()
    panel_labels = [
        next((t for t in ax.texts if t.get_text() == label))
        for ax, label in [(photo_ax, "a"), (diagram_ax, "b")]
    ]
    label_anchors_pt = [
        fig.dpi_scale_trans.inverted().transform(
            t.get_transform().transform(t.get_position())
        )
        * 72
        for t in panel_labels
    ]
    if not abs(label_anchors_pt[0][1] - label_anchors_pt[1][1]) < 1.5:
        raise ValueError(
            "Figure input requirement failed: abs(label_anchors_pt[0][1] - label_anchors_pt[1][1]) < 1.5"
        )
    if not abs(label_anchors_pt[0][0] - 15 - (label_anchors_pt[1][0] - 152)) < 1.5:
        raise ValueError(
            "Figure input requirement failed: abs(label_anchors_pt[0][0] - 15 - (label_anchors_pt[1][0] - 152)) < 1.5"
        )
    return fig
