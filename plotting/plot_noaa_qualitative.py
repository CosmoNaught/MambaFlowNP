"""Maps of context observations, held-out values and posterior-mean error over a terrain
basemap. Called by figures.py."""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "results" / "qualitative"
STATES_PATH = HERE / "assets" / "us_states_paths.npz"
TERRAIN_DIR = HERE / "assets"
OUT_DIR = HERE.parent / "results"

REGION_BBOX = {
    "us": (-125.0, -66.5, 24.0, 50.0),
    "california": (-124.5, -114.0, 32.0, 42.0),
    "utah": (-114.0, -109.0, 37.0, 42.0),
}
REGION_DISPLAY = {"us": "US", "california": "California", "utah": "Utah"}

# States of each region, used for the clip polygon and the in-region point mask. The US region
# is the contiguous 48 states and the District of Columbia.
REGION_STATES = {
    "california": ("California",),
    "utah": ("Utah",),
    "us": tuple(sorted({
        "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
        "Delaware", "District of Columbia", "Florida", "Georgia", "Idaho", "Illinois",
        "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
        "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
        "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
        "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
        "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
        "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
        "Wisconsin", "Wyoming",
    })),
}


def _load_state_paths() -> tuple[list[np.ndarray], list[str]] | tuple[None, None]:
    """Returns (rings, names). Each ring is an (n, 2) array of longitude and latitude."""
    if not STATES_PATH.is_file():
        return None, None
    blob = np.load(STATES_PATH, allow_pickle=True)
    rings = [np.asarray(p, dtype=np.float32) for p in blob["paths"]]
    names = list(blob.get("state_names", np.array([], dtype=object)))
    if not names or len(names) != len(rings):
        names = [""] * len(rings)
    return rings, names


def _build_region_clip(region: str, rings: list[np.ndarray], names: list[str]) -> MplPath | None:
    """Returns a compound path over the states of the region, used to clip the colour terrain
    and to mask the station points."""
    target = set(REGION_STATES.get(region, ()))
    if not target:
        return None
    chunks = []
    for ring, name in zip(rings, names):
        if name not in target or ring.shape[0] < 3:
            continue
        chunks.append(MplPath(ring))
    if not chunks:
        return None
    return MplPath.make_compound_path(*chunks)


def _mask_inside(points_lonlat: np.ndarray, clip_path: MplPath | None) -> np.ndarray:
    """Returns True where the point is inside the clip path."""
    if clip_path is None or points_lonlat.size == 0:
        return np.ones(points_lonlat.shape[0], dtype=bool)
    return clip_path.contains_points(points_lonlat)


def _load_terrain(region: str):
    img_path = TERRAIN_DIR / f"terrain_{region}.png"
    meta_path = TERRAIN_DIR / f"terrain_{region}.npz"
    if not img_path.is_file() or not meta_path.is_file():
        return None, None
    img = plt.imread(img_path)
    extent = tuple(np.load(meta_path)["extent"].tolist())
    return img, extent


def _draw_terrain(ax, terrain, clip_path: MplPath | None = None) -> None:
    """Draws the terrain. With a clip path, the terrain is drawn dim and grey everywhere and in
    colour inside the path."""
    img, extent = terrain
    if img is None:
        return
    if clip_path is None:
        ax.imshow(img, extent=extent, origin="upper",
                  alpha=0.55, zorder=0.5, interpolation="bilinear")
        return

    rgb = np.asarray(img)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    elif rgb.ndim == 3 and rgb.shape[-1] == 3:
        pass
    else:
        ax.imshow(img, extent=extent, origin="upper",
                  alpha=0.55, zorder=0.5, interpolation="bilinear")
        return

    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    grey = np.stack([lum, lum, lum], axis=-1)
    grey = 0.5 + 0.5 * grey
    ax.imshow(grey, extent=extent, origin="upper",
              alpha=0.45, zorder=0.4, interpolation="bilinear")

    colour_img = ax.imshow(rgb, extent=extent, origin="upper",
                           alpha=0.85, zorder=0.5, interpolation="bilinear")
    transform = ax.transData
    colour_img.set_clip_path(clip_path, transform=transform)


def _draw_outline(ax, region: str, paths: list[np.ndarray] | None) -> None:
    if paths is None:
        return
    lon_min, lon_max, lat_min, lat_max = REGION_BBOX[region]
    lon_pad = 0.5 * (lon_max - lon_min)
    lat_pad = 0.5 * (lat_max - lat_min)
    for path in paths:
        if path.shape[0] < 2:
            continue
        if (path[:, 0].max() < lon_min - lon_pad
                or path[:, 0].min() > lon_max + lon_pad
                or path[:, 1].max() < lat_min - lat_pad
                or path[:, 1].min() > lat_max + lat_pad):
            continue
        ax.plot(path[:, 0], path[:, 1],
                color="#222222", linewidth=1.1, alpha=0.95, zorder=1.6)


def _scatter_panel(
    ax, lonlat, values, *, cmap, vmin, vmax, dot_size,
    context_lonlat=None,
):
    """Draws the context points as red crosses below the value scatter."""
    if context_lonlat is not None and context_lonlat.size:
        ax.scatter(
            context_lonlat[:, 0], context_lonlat[:, 1],
            s=22, marker="x", c="#d62728",
            linewidths=1.0, alpha=0.55, zorder=1.8,
        )
    sc = ax.scatter(
        lonlat[:, 0], lonlat[:, 1],
        c=values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=dot_size, edgecolors="black", linewidths=0.35, zorder=3,
    )
    return sc


def _row_dimensions(region: str, n_panels: int = 3) -> tuple[float, float, float]:
    """Returns (figure width, row height, dot size) for one row of map panels. The row height
    matches the aspect ratio of the region, and two-panel rows keep the panel size of three-panel
    rows."""
    if region == "us":
        if n_panels == 2:
            return 12.0, 2.6, 5.5
        return 17.5, 2.6, 5.5
    if region == "utah":
        if n_panels == 2:
            return 9.5, 3.6, 14.0
        return 13.5, 3.6, 14.0
    if n_panels == 2:
        return 8.5, 3.5, 12.0
    return 12.0, 3.5, 12.0


def render(
    region: str,
    fractions: tuple[float, ...],
    out_suffix: str = "",
    panels: tuple[str, ...] = ("context", "query", "error"),
    data_suffix: str = "",
) -> None:
    if region not in REGION_BBOX:
        raise SystemExit(f"Unsupported region: {region}")

    valid_panels = ("context", "query", "error")
    for p in panels:
        if p not in valid_panels:
            raise SystemExit(f"Unknown panel '{p}'; expected subset of {valid_panels}")
    if not panels:
        raise SystemExit("Need at least one panel.")

    rows = []
    for frac in fractions:
        path = DATA_DIR / f"{region}{data_suffix}_f{int(round(frac * 100)):02d}.npz"
        if not path.is_file():
            raise SystemExit(f"Missing dump for {region} f={frac}: {path}")
        rows.append(np.load(path))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Nimbus Roman", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    rings, names = _load_state_paths()
    paths = rings
    clip_path = _build_region_clip(region, rings, names) if rings is not None else None
    terrain = _load_terrain(region)

    fig_w, row_h, dot_size = _row_dimensions(region, n_panels=len(panels))
    n_rows = len(rows)
    margin = 0.6 if n_rows == 1 else 1.4
    fig_h = max(row_h * n_rows + margin, 3.2)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # One column per map, a shared temperature colour bar after the last temperature map, and an
    # error colour bar after the error map.
    has_temp = any(p in ("context", "query") for p in panels)
    has_err = "error" in panels
    layout: list[tuple[str, float]] = []
    last_temp_idx = -1
    for i, p in enumerate(panels):
        if p in ("context", "query"):
            last_temp_idx = i
    inserted_temp_cbar = False
    for i, p in enumerate(panels):
        layout.append(("map", 1.0))
        if p in ("context", "query") and i == last_temp_idx and not inserted_temp_cbar:
            layout.append(("cbar_temp", 0.06))
            inserted_temp_cbar = True
        if p == "error":
            layout.append(("cbar_err", 0.04))

    width_ratios = [r for _, r in layout]
    gs = fig.add_gridspec(
        n_rows, len(layout),
        width_ratios=width_ratios,
        wspace=0.18, hspace=0.20,
    )

    map_cols: dict[str, int] = {}
    cbar_temp_col = -1
    cbar_err_col = -1
    panel_iter = iter(panels)
    for ci, (kind, _) in enumerate(layout):
        if kind == "map":
            name = next(panel_iter)
            map_cols[name] = ci
        elif kind == "cbar_temp":
            cbar_temp_col = ci
        elif kind == "cbar_err":
            cbar_err_col = ci

    axes_dict: dict[str, np.ndarray] = {p: np.empty(n_rows, dtype=object) for p in panels}
    cax_temp = np.empty(n_rows, dtype=object) if has_temp else None
    cax_sq = np.empty(n_rows, dtype=object) if has_err else None
    for r in range(n_rows):
        share_y = None
        for p in panels:
            ax = fig.add_subplot(gs[r, map_cols[p]], sharey=share_y)
            axes_dict[p][r] = ax
            if share_y is None:
                share_y = ax
        if has_temp:
            cax_temp[r] = fig.add_subplot(gs[r, cbar_temp_col])
        if has_err:
            cax_sq[r] = fig.add_subplot(gs[r, cbar_err_col])

    lon_min, lon_max, lat_min, lat_max = REGION_BBOX[region]

    for r_idx, (frac, blob) in enumerate(zip(fractions, rows)):
        xc = blob["xc_lonlat"]; yc = blob["yc_raw"]
        xt = blob["xt_lonlat"]; yt = blob["yt_raw"]
        sq = blob["sq_err"]
        # Absolute error of the posterior mean in degrees C.
        abs_err = np.sqrt(np.maximum(sq, 0.0))

        # Points outside the region polygon are not drawn.
        ctx_in = _mask_inside(xc, clip_path)
        qry_in = _mask_inside(xt, clip_path)
        xc_v = xc[ctx_in]; yc_v = yc[ctx_in]
        xt_v = xt[qry_in]; yt_v = yt[qry_in]
        sq_v = sq[qry_in]; err_v = abs_err[qry_in]
        n_ctx = xc_v.shape[0]; n_qry = xt_v.shape[0]

        # Colour ranges are set per row.
        all_temps = np.concatenate([yc_v, yt_v]) if (yc_v.size and yt_v.size) else (yc_v if yc_v.size else yt_v)
        if all_temps.size:
            temp_lo = float(all_temps.min())
            temp_hi = float(all_temps.max())
        else:
            temp_lo, temp_hi = 0.0, 1.0
        if err_v.size:
            err_hi = float(np.percentile(err_v, 97.0))
            err_hi = max(err_hi, 1e-6)
        else:
            err_hi = 1.0

        n_c_label = xc.shape[0]
        n_q_label = xt.shape[0]

        sc_temp_ref = None
        sc_err_ref = None
        rmse = float(np.sqrt(np.mean(sq)))

        if "context" in panels:
            ax = axes_dict["context"][r_idx]
            _draw_terrain(ax, terrain, clip_path); _draw_outline(ax, region, paths)
            sc_ctx = _scatter_panel(
                ax, xc_v, yc_v, cmap="viridis",
                vmin=temp_lo, vmax=temp_hi, dot_size=dot_size,
            )
            ax.set_title(f"Context  ($n_c={n_c_label}$)", fontsize=11)
            ax.set_ylabel("Latitude", fontsize=10)
            sc_temp_ref = sc_ctx

        if "query" in panels:
            ax = axes_dict["query"][r_idx]
            _draw_terrain(ax, terrain, clip_path); _draw_outline(ax, region, paths)
            sc_qry = _scatter_panel(
                ax, xt_v, yt_v, cmap="viridis",
                vmin=temp_lo, vmax=temp_hi, dot_size=dot_size,
                context_lonlat=xc_v,
            )
            ax.set_title(f"Query  ($n_q={n_q_label}$)", fontsize=11)
            sc_temp_ref = sc_qry

        if "error" in panels:
            ax = axes_dict["error"][r_idx]
            _draw_terrain(ax, terrain, clip_path); _draw_outline(ax, region, paths)
            sc_sq = _scatter_panel(
                ax, xt_v, err_v, cmap="magma",
                vmin=0.0, vmax=err_hi, dot_size=dot_size,
            )
            ax.set_title(
                f"Per-point error  (RMSE={rmse:.2f}°C)",
                fontsize=11,
            )
            sc_err_ref = sc_sq

        # Temperature colour bar ticks are on the left, away from the error map.
        if cax_temp is not None and sc_temp_ref is not None:
            cb_t = fig.colorbar(sc_temp_ref, cax=cax_temp[r_idx])
            cb_t.ax.yaxis.set_ticks_position("left")
            cb_t.ax.tick_params(labelsize=8)
            cb_t.ax.set_title(r"$^{\circ}$C", fontsize=9, pad=2)
        if cax_sq is not None and sc_err_ref is not None:
            cb_s = fig.colorbar(sc_err_ref, cax=cax_sq[r_idx])
            cb_s.ax.tick_params(labelsize=8)
            cb_s.ax.set_title(r"$^{\circ}$C", fontsize=9, pad=2)

    all_axes = [axes_dict[p][r] for r in range(n_rows) for p in panels]
    for ax in all_axes:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
    for r in range(n_rows):
        for c, p in enumerate(panels):
            if r < n_rows - 1:
                axes_dict[p][r].tick_params(axis="x", labelbottom=False)
            if c >= 1:
                axes_dict[p][r].tick_params(axis="y", labelleft=False)
    for p in panels:
        axes_dict[p][n_rows - 1].set_xlabel("Longitude", fontsize=10)

    # Context fraction of each row, in the left margin of multi-row figures.
    if n_rows > 1:
        fig.canvas.draw()
        first_panel = panels[0]
        for r_idx, frac in enumerate(fractions):
            bbox = axes_dict[first_panel][r_idx].get_position()
            fig.text(
                max(bbox.x0 - 0.045, 0.005),
                bbox.y0 + bbox.height / 2,
                f"{int(round(frac*100))}% context",
                ha="center", va="center", rotation=90,
                fontsize=11, fontweight="bold",
            )

    date_int = int(rows[0]["date_int"])
    date_str = datetime.date(date_int // 10000,
                        (date_int // 100) % 100,
                        date_int % 100).strftime("%B %-d, %Y")
    region_pretty = REGION_DISPLAY[region]
    num_points = int(rows[0]["xc_lonlat"].shape[0] + rows[0]["xt_lonlat"].shape[0])
    fig.suptitle(
        rf"NOAA {region_pretty} daily maximum temperature on {date_str}  ($K{{=}}{num_points}$)",
        fontsize=12.5, y=0.995,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = OUT_DIR / f"noaa_qualitative_{region}{out_suffix}"
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_prefix.with_suffix('.pdf')}")
    print(f"wrote {out_prefix.with_suffix('.png')}")

