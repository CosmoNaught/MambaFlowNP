"""Maps of one test day: the observed field, the MFNP prediction from the context stations, and
the error of the prediction at the query stations."""

import pathlib
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.interpolate import griddata

import style
from style import DEG, FIG_W, INK

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
STATES = ASSETS / "us_states_paths.npz"
COUNTIES = ASSETS / "us_county_paths.npz"  # US Census county boundaries, Utah and California

LOWER_48 = (
    "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "District of Columbia", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming")


@dataclass(frozen=True)
class MapRegion:
    """Display settings of one region."""

    states: tuple       # states inside the outline
    counties: bool      # draw county borders inside the states
    pad: float          # margin around the outline, degrees
    grid_step: float    # spacing of the grid the fields are interpolated onto, degrees
    hexagons: tuple     # hexagons across and down the error map
    hex_edge: float     # width of the white lines between hexagons
    dot_size: float     # marker area of the context stations, pt^2
    bin_width: float    # histogram bin width, degrees C


REGIONS = {
    "us": MapRegion(LOWER_48, False, 0.2, 0.1, (46, 16), 0.15, 1.2, 0.25),
    "utah": MapRegion(("Utah",), True, 0.12, 0.01, (9, 7), 0.3, 3.2, 0.5),
    "california": MapRegion(("California",), True, 0.15, 0.02, (13, 9), 0.3, 1.8, 0.5),
}

# Classes of absolute error, shared by the error map and the histogram that serves as its legend.
ERROR_CMAP = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.10, 0.92, 4)))
ERROR_NORM = BoundaryNorm([0.0, 1.0, 2.0, 3.0, np.inf], ERROR_CMAP.N)
ERROR_MAX = 4.0  # the last histogram bin collects every error above this
TEMPERATURE_STEP = 2.0


class Outline:
    """Outline of a set of states and the map extent around it, in degrees."""

    def __init__(self, states, pad, counties=False):
        self.rings = load_rings(STATES, states)
        # Inner borders: between states, and between counties if asked.
        self.borders = self.rings + (load_rings(COUNTIES, states) if counties else [])
        self.path = Path.make_compound_path(*map(Path, self.rings))
        points = np.vstack(self.rings)
        lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
        self.extent = (lo[0], hi[0], lo[1], hi[1])
        # Vertical stretch that gives degrees of longitude and latitude equal length at the centre.
        self.aspect = 1.0 / np.cos(np.deg2rad(0.5 * (lo[1] + hi[1])))
        self.ratio = (hi[0] - lo[0]) / ((hi[1] - lo[1]) * self.aspect)

    def boundary(self):
        """Segments that belong to one ring only, which bound the union of the states."""
        count = Counter(frozenset(edge) for ring in self.rings
                        for edge in zip(map(tuple, ring[:-1]), map(tuple, ring[1:]))
                        if edge[0] != edge[1])
        return [tuple(edge) for edge, n in count.items() if n == 1]


class Day:
    """The stations of one dump that lie inside the outline, in degrees and degrees C."""

    def __init__(self, path, outline):
        with np.load(path) as f:
            ctx = outline.path.contains_points(f["xc_lonlat"])
            qry = outline.path.contains_points(f["xt_lonlat"])
            self.context, self.context_values = f["xc_lonlat"][ctx], f["yc_raw"][ctx]
            self.query, self.truth = f["xt_lonlat"][qry], f["yt_raw"][qry]
            self.prediction = f["pred_mean"][qry]
        self.abs_error = np.abs(self.prediction - self.truth)
        self.rmse = float(np.sqrt(np.mean(self.abs_error ** 2)))

    def temperature_levels(self):
        """Class boundaries every TEMPERATURE_STEP spanning the 1st to 99th percentile."""
        values = np.concatenate([self.context_values, self.truth, self.prediction])
        lo, hi = np.percentile(values, (1, 99)) / TEMPERATURE_STEP
        return np.arange(np.floor(lo), np.ceil(hi) + 1) * TEMPERATURE_STEP


def load_rings(path, states):
    with np.load(path, allow_pickle=True) as f:
        return [np.asarray(p, dtype=np.float64)
                for p, name in zip(f["paths"], f["state_names"]) if name in states]


def setup(ax, outline):
    ax.set_xlim(*outline.extent[:2])
    ax.set_ylim(*outline.extent[2:])
    ax.set_aspect(outline.aspect)
    ax.set_axis_off()


def draw_outline(ax, outline, lw, borders, alpha):
    """Inner borders in the colour ``borders``, clipped to the outline, and the outer outline on
    top."""
    inner = LineCollection(outline.borders, colors=borders, lw=lw / 2, alpha=alpha, zorder=2)
    ax.add_collection(inner)
    inner.set_clip_path(PathPatch(outline.path, transform=ax.transData))
    ax.add_collection(LineCollection(outline.boundary(), colors=INK, lw=lw, capstyle="round",
                                     zorder=2.5))


def draw_field(ax, outline, region, day, values, norm):
    """Linear interpolation on the Delaunay triangulation of the stations, and the nearest
    station outside their convex hull. ``values`` are the values at the query stations."""
    setup(ax, outline)
    x0, x1, y0, y1 = outline.extent
    step = region.grid_step
    grid = tuple(np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)))
    xy = np.vstack([day.context, day.query])
    v = np.concatenate([day.context_values, values])
    z = griddata(xy, v, grid, method="linear")
    z = np.where(np.isnan(z), griddata(xy, v, grid, method="nearest"), z)
    mesh = ax.pcolormesh(*grid, z, cmap="viridis", norm=norm, shading="nearest", zorder=1,
                         rasterized=True)
    mesh.set_clip_path(PathPatch(outline.path, transform=ax.transData))
    draw_outline(ax, outline, lw=0.5, borders="white", alpha=0.8)
    return mesh


def draw_errors(ax, outline, region, day):
    """Mean absolute error of the query stations in each hexagon, on light grey land."""
    setup(ax, outline)
    ax.add_patch(PathPatch(outline.path, facecolor="#efefef", edgecolor="none", zorder=0.2))
    hexes = ax.hexbin(*day.query.T, C=day.abs_error, reduce_C_function=np.mean,
                      gridsize=region.hexagons, extent=outline.extent, cmap=ERROR_CMAP,
                      norm=ERROR_NORM, mincnt=1, linewidths=region.hex_edge, edgecolors="white",
                      zorder=1, rasterized=True)
    hexes.set_clip_path(PathPatch(outline.path, transform=ax.transData))
    draw_outline(ax, outline, lw=0.6, borders="#555555", alpha=0.6)


def draw_histogram(ax, region, day, show_count):
    """Share of query stations by absolute error, in the classes of the error map."""
    width = region.bin_width
    edges = np.arange(0, ERROR_MAX + 1.5 * width, width)
    counts, _ = np.histogram(np.minimum(day.abs_error, ERROR_MAX + width / 2), bins=edges)
    share = 100 * counts / counts.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, share, width=width, color=ERROR_CMAP(ERROR_NORM(centres)),
           edgecolor="#444444", lw=0.25)
    ax.axvline(day.rmse, color="k", lw=0.6, ls=(0, (2, 1.5)))
    if show_count:
        # Two lines of text above the tallest bar.
        label_y = 1.08 * share.max()
        ax.set_ylim(0, 1.12 * label_y)
        ax.text(day.rmse + 0.12, 0.93 * label_y, rf"$n_q={len(day.query)}$", fontsize=7,
                va="top")
    else:
        label_y = 0.97 * share.max()
        ax.set_ylim(0, 1.04 * share.max())
    ax.text(day.rmse + 0.12, label_y, rf"RMSE ${day.rmse:.2f}$ {DEG}", fontsize=7, va="top")
    ax.set_yticks(np.arange(0, label_y, 5 if share.max() < 25 else 10))
    ax.set_xlim(0, edges[-1])
    ax.set_xticks([0, 1, 2, 3, ERROR_MAX + width / 2], ["0", "1", "2", "3", f"{ERROR_MAX:g}+"])
    ax.set_xlabel(rf"$|\hat y_q - y_q|$ ({DEG})", labelpad=1)
    ax.set_ylabel("Stations (%)", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=1.8, width=0.4, pad=1.2)


def figure(ratio):
    """Figure and axes in inches: truth, prediction and colour bar on top; error map and
    histogram below. Wide regions get a wider error map; otherwise the figure is at most 5 in
    high."""
    bar_w, bar_labels, bottom = 0.07, 0.42, 0.02
    if ratio > 1:
        gap, bar_gap, title, row_gap = 0.06, 0.10, 0.16, 0.06
        w = (FIG_W - 0.04 - gap - bar_gap - bar_w - bar_labels) / 2
        h = w / ratio
        left = 0.02
        error_w = FIG_W - 0.04 - 0.40 - 1.10 - 0.06
        error_h = error_w / ratio
        top = bottom + error_h + title + row_gap
        bar = (top + 0.08, h - 0.16)
        hist = (left + error_w + 0.40, bottom + 0.36, 1.10, error_h - 0.44)
    else:
        gap, bar_gap, title, row_gap = 0.10, 0.08, 0.18, 0.08
        w = min((FIG_W - 0.04 - gap - bar_gap - bar_w - bar_labels) / 2,
                (5.0 - 2 * title - row_gap - bottom) / 2 * ratio)
        error_w = w
        h = error_h = w / ratio
        left = (FIG_W - (2 * w + gap + bar_gap + bar_w + bar_labels)) / 2
        top = bottom + h + title + row_gap
        bar = (top + 0.12 * h, 0.76 * h)
        hist_x = left + w + gap + 0.38
        hist_w = left + 2 * w + gap + bar_gap + bar_w + bar_labels - 0.10 - hist_x
        hist = (hist_x, bottom + 0.36 + 0.12 * h, hist_w, 0.60 * h)
    height = top + h + title

    fig = plt.figure(figsize=(FIG_W, height))
    rects = {"truth": (left, top, w, h), "prediction": (left + w + gap, top, w, h),
             "colorbar": (left + 2 * w + gap + bar_gap, bar[0], bar_w, bar[1]),
             "error": (left, bottom, error_w, error_h), "histogram": hist}
    axes = {k: fig.add_axes([x / FIG_W, y / height, rw / FIG_W, rh / height])
            for k, (x, y, rw, rh) in rects.items()}
    return fig, axes


def render_map(name, npz_path, out_prefix):
    """Renders the dump at ``npz_path`` for the region ``name`` ("us", "utah" or "california")
    to ``out_prefix``.pdf and .png. Returns the RMSE over the query stations inside the outline."""
    region = REGIONS[name]
    outline = Outline(region.states, region.pad, region.counties)
    day = Day(npz_path, outline)
    levels = day.temperature_levels()
    norm = BoundaryNorm(levels, 256, extend="both")
    compact = outline.ratio <= 1

    style.apply()
    fig, axes = figure(outline.ratio)
    mesh = draw_field(axes["truth"], outline, region, day, day.truth, norm)
    draw_field(axes["prediction"], outline, region, day, day.prediction, norm)
    axes["prediction"].scatter(*day.context.T, s=region.dot_size, c="#E3120B", lw=0, zorder=3,
                               rasterized=True)
    axes["truth"].set_title("Truth", pad=2)
    axes["prediction"].set_title("MFNP prediction", pad=2)
    bar = fig.colorbar(mesh, cax=axes["colorbar"], ticks=levels[1::2], extendfrac=0.04)
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=7, length=1.8, width=0.4, pad=1.2)
    bar.set_label(f"Temperature ({DEG})", fontsize=7, labelpad=2)

    draw_errors(axes["error"], outline, region, day)
    axes["error"].set_title("MFNP mean absolute error per hexagon", pad=2)
    draw_histogram(axes["histogram"], region, day, show_count=compact)
    axes["histogram"].set_title("Per-station error", pad=4 if compact else 2)
    style.save(fig, pathlib.Path(out_prefix))
    return day.rmse
