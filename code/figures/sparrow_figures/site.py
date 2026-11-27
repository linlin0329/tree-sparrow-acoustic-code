"""Approved manuscript drawing operations with explicit local dependencies."""

from __future__ import annotations
from .resources import ASSETS
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
from pyproj import Transformer
import rasterio
from shapely.geometry import box, shape

HERE = ASSETS / "map"
DATA = HERE / "map_data"
CFG = json.loads((HERE / "map_config.json").read_text())
IMAGERY = json.loads((HERE / "imagery_manifest.json").read_text())
PROJ = ccrs.UTM(48)
GEO = ccrs.PlateCarree()
CRS = CFG["map_crs"]
FORWARD = Transformer.from_crs(4326, CRS, always_xy=True)
INVERSE = Transformer.from_crs(CRS, 4326, always_xy=True)
OFFSETS = {
    "GYY": (5, 5),
    "MQ": (-5, -5),
    "SH": (-5, -5),
    "LZ": (-5, -5),
    "YX": (5, 5),
    "LJX": (5, -5),
}


def plot_geom(frame, ax, bounds, **style):
    clip = frame.clip(box(*bounds))
    if len(clip):
        clip.plot(ax=ax, **style)


def scale_bar(ax, length, bounds):
    l, b, r, t = bounds
    x = l + 0.075 * (r - l)
    y = b + 0.08 * (t - b)
    hh = 0.017 * (t - b)
    ax.plot([x, x + length], [y, y], color="#263A43", lw=1.8, zorder=12)
    ax.plot([x, x], [y - hh / 2, y + hh / 2], color="#263A43", lw=0.65, zorder=12)
    ax.plot(
        [x + length, x + length],
        [y - hh / 2, y + hh / 2],
        color="#263A43",
        lw=0.65,
        zorder=12,
    )
    for px, label in [(x, "0"), (x + length, f"{length / 1000:g} km")]:
        ax.annotate(
            label,
            (px, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#263A43",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            zorder=13,
        )


def north(ax, bounds):
    l, b, r, t = bounds
    x = l + 0.925 * (r - l)
    y = b + 0.805 * (t - b)
    lon, lat = INVERSE.transform(x, y)
    xn, yn = FORWARD.transform(lon, lat + 0.02)
    dx, dy = (xn - x, yn - y)
    length = 0.11 * (t - b)
    factor = length / np.hypot(dx, dy)
    ax.annotate(
        "",
        xy=(x + dx * factor, y + dy * factor),
        xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "color": "#263A43", "lw": 1.0},
        zorder=12,
    )
    ax.annotate(
        "N",
        (x + dx * factor, y + dy * factor),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=13,
    )


def province_locator(fig, regional_ax, regional_bounds, sites):
    """Show provincial context; the small box denotes the regional map extent."""
    cfg = CFG["province_locator"]
    source = json.loads((HERE / cfg["source_manifest"]).read_text())
    geometry_path = HERE / cfg["geometry"]
    payload = geometry_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != source["output"]["sha256"]:
        raise ValueError("Provincial boundary hash does not match its source manifest")
    features = json.loads(payload)["features"]
    if len(features) != 1 or features[0]["properties"]["iso_3166_2"] != "CN-GS":
        raise ValueError("The locator requires the single Gansu province feature")
    province = shape(features[0]["geometry"])
    if province.is_empty or not province.is_valid:
        raise ValueError("Invalid provincial boundary geometry")

    # Fractions refer to the whole figure; keep the inherited map axes unchanged.
    fig.canvas.draw()
    outer = regional_ax.get_position()
    dx, dy = cfg["frame_offset_fraction"]
    width, height = cfg["frame_size_fraction"]
    left, bottom = outer.x0 + dx, outer.y1 - dy - height
    frame = fig.add_axes([left, bottom, width, height], label="province_frame", zorder=30)
    frame.set_facecolor("white")
    frame.set_xticks([])
    frame.set_yticks([])
    for spine in frame.spines.values():
        spine.set_color("#A7AFB2")
        spine.set_linewidth(0.55)
    # Aligned outer edges use the regional-map border; inner dividers remain light.
    for edge, offset in (("left", dx), ("top", dy)):
        if np.isclose(offset, 0.0):
            frame.spines[edge].set_color(regional_ax.spines["geo"].get_edgecolor())
            frame.spines[edge].set_linewidth(regional_ax.spines["geo"].get_linewidth())
    frame.text(0.035, 0.95, "Gansu Province", ha="left", va="top",
               fontsize=8.2, fontweight="bold", color="#263A43")
    area = source["official_area"]["value_km2"]
    frame.text(0.965, 0.94, f"≈{area:,.0f} km²", ha="right", va="top",
               fontsize=7.4, color="#46545C")
    frame.text(0.035, 0.045, "Box: regional map extent", ha="left", va="bottom",
               fontsize=7.0, color="#46545C")

    width_pt, height_pt = fig.get_size_inches() * 72
    projection = cfg["projection"]
    locator_projection = ccrs.LambertConformal(
        central_longitude=projection["central_longitude"],
        central_latitude=projection["central_latitude"],
        standard_parallels=tuple(projection["standard_parallels"]),
    )
    locator = fig.add_axes(
        [left + 4 / width_pt, bottom + 15 / height_pt,
         width - 8 / width_pt, height - 32 / height_pt],
        projection=locator_projection, label="province_map", zorder=31,
    )
    locator.set_facecolor("white")
    locator.spines["geo"].set_visible(False)
    minx, miny, maxx, maxy = province.bounds
    pad = cfg["extent_padding_degrees"]
    locator.set_extent([minx - pad, maxx + pad, miny - pad, maxy + pad], crs=GEO)
    locator.add_geometries([province], crs=GEO, facecolor="#E5E8E7",
                           edgecolor="#56666D", linewidth=0.6, zorder=2)
    locator.add_geometries([box(*regional_bounds)], crs=PROJ,
                           facecolor="#F6D4B6", edgecolor="#A8622C",
                           linewidth=0.85, zorder=3)
    fig.canvas.draw()

    # The context inset may cover background imagery, but never sampling marks.
    frame_display = frame.get_window_extent(fig.canvas.get_renderer())
    for point in sites.geometry:
        x, y = regional_ax.transData.transform((point.x, point.y))
        if frame_display.contains(x, y):
            raise ValueError("The provincial locator obscures a village reference point")
    for title, ax in [("b", fig.axes[1]), ("c", fig.axes[2])]:
        if frame_display.overlaps(ax.get_window_extent(fig.canvas.get_renderer())):
            raise ValueError(f"The provincial locator overlaps panel {title}")
    fig._province_locator_metadata = {
        "feature_id": features[0].get("id"),
        "boundary_sha256": source["output"]["sha256"],
        "official_area_km2": area,
        "official_area_source": source["official_area"]["source_url"],
        "extent_box_role": cfg["box_role"],
        "regional_extent_projected": list(regional_bounds),
        "regional_extent_crs": CRS,
        "frame_bbox_figure": [left, bottom, width, height],
        "projection": projection,
        "sampling_marks_obscured": 0,
    }
    return frame, locator


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )
    approval = json.loads((HERE / "author_site_review.json").read_text())
    if not approval["decision"] == "retain_all_six_reported_reference_locations":
        raise ValueError(
            "Figure input requirement failed: approval['decision'] == 'retain_all_six_reported_reference_locations'"
        )
    gpkg = DATA / "map_layers.gpkg"
    yellow = gpd.read_file(gpkg, layer="osm_yellow_river_aoi").to_crs(CRS)
    reported = gpd.read_file(gpkg, layer="study_site_references").to_crs(CRS)
    fig = plt.figure(figsize=(CFG["figure"]["width_in"], CFG["figure"]["height_in"]))
    axes = []
    bounds_by_panel = {}
    ticks = {
        "a": ([103.2, 103.8, 104.4], [35.5, 36.0, 36.5]),
        "b": ([104.2, 104.3, 104.4], [36.4, 36.45, 36.5]),
        "c": ([103.2, 103.25, 103.3], [35.92, 35.94, 35.96]),
    }
    for panel, cfg in CFG["panels"].items():
        ax = fig.add_axes(CFG["figure"]["axes"][panel], projection=PROJ)
        ax.set_label(panel)
        axes.append(ax)
        bounds = IMAGERY["outputs"][panel]["bounds_projected"]
        bounds_by_panel[panel] = bounds
        l, b, r, t = bounds
        with rasterio.open(DATA / f"imagery_{panel}.tif") as ds:
            rgb = np.moveaxis(ds.read(), 0, -1)
            rb = ds.bounds
        masked = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        masked[:, :, :3] = rgb
        masked[:, :, 3] = np.where(np.any(rgb != 0, axis=2), 255, 0)
        ax.set_facecolor("#F0F0ED")
        ax.imshow(
            masked,
            origin="upper",
            extent=[rb.left, rb.right, rb.bottom, rb.top],
            transform=PROJ,
            zorder=0,
            rasterized=True,
        )
        if panel == "a":
            plot_geom(
                yellow,
                ax,
                bounds,
                color="#8CC6D8",
                linewidth=0.75,
                alpha=0.85,
                zorder=3,
            )
        ax.set_xlim(l, r)
        ax.set_ylim(b, t)
        gl = ax.gridlines(
            crs=GEO,
            draw_labels=True,
            xlocs=ticks[panel][0],
            ylocs=ticks[panel][1],
            linewidth=0.25,
            color="#9AA7AE",
            alpha=0.5,
            linestyle=":",
            x_inline=False,
            y_inline=False,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = False
        gl.ylines = False
        gl.xlabel_style = {"size": 6.3}
        gl.ylabel_style = {"size": 6.3}
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xpadding = 3
        gl.ypadding = 3
        ax.annotate(panel, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="left", va="baseline", fontsize=9, fontweight="bold")
        ax.annotate(cfg["title"], xy=(0, 1), xycoords="axes fraction",
                    xytext=(16, 8), textcoords="offset points",
                    ha="left", va="baseline", fontsize=9, fontweight="bold")
        subset = (
            reported
            if panel == "a"
            else (
                reported[reported.site_code != "LJX"]
                if panel == "b"
                else reported[reported.site_code == "LJX"]
            )
        )
        for _, row in subset.iterrows():
            x, y = (row.geometry.x, row.geometry.y)
            ax.scatter(
                [x],
                [y],
                s=26 if panel == "a" else 42,
                marker="o",
                facecolor="#D78342",
                edgecolor="white",
                linewidth=0.8,
                zorder=9,
            )
            if panel != "a":
                off = OFFSETS[row.site_code]
                ax.annotate(
                    row.site_code,
                    (x, y),
                    xytext=off,
                    textcoords="offset points",
                    fontsize=6.8,
                    fontweight="bold",
                    color="#352516",
                    ha="left" if off[0] >= 0 else "right",
                    va="center",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=11,
                )
        if panel == "a":
            for key in ["b", "c"]:
                bb = IMAGERY["outputs"][key]["bounds_projected"]
                ax.add_patch(
                    Rectangle(
                        (bb[0], bb[1]),
                        bb[2] - bb[0],
                        bb[3] - bb[1],
                        fill=False,
                        edgecolor="white",
                        linewidth=0.85,
                        linestyle="--",
                        zorder=5,
                    )
                )
                ax.text(
                    bb[0] if key == "b" else bb[2] + 0.025 * (r - l),
                    bb[3] + 0.018 * (t - b),
                    key,
                    fontsize=8,
                    weight="bold",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#263A43")],
                    zorder=7,
                )
            ax.text(
                0.43,
                0.37,
                "Yellow River",
                transform=ax.transAxes,
                rotation=15,
                fontsize=7.2,
                color="#275A72",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=6,
            )
        scale_bar(ax, cfg["scale_bar_m"], bounds)
        if panel == "a":
            north(ax, bounds)
    fig.text(
        0.065,
        0.061,
        "Contains modified Copernicus Sentinel data 2023. Image acquired 14 May 2023. Provincial boundary: Natural Earth.",
        fontsize=6.2,
        color="#46545C",
    )
    fig.text(
        0.065,
        0.032,
        "Map data © OpenStreetMap contributors; openstreetmap.org/copyright. Symbols: village-level reference locations.",
        fontsize=6.2,
        color="#46545C",
    )
    province_locator(fig, axes[0], bounds_by_panel["a"], reported)
    fig.canvas.draw()
    return fig
