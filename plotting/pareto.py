"""Time per episode against RMSE and CRPS for one region, with the Pareto frontier."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

import style
from style import DEG, INK, save

# Sizes on top of style.apply().
RC = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
      "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
      "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5,
      "ytick.minor.width": 0.5, "xtick.major.size": 3, "ytick.major.size": 3,
      "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "axes.spines.top": False,
      "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
      "xtick.color": INK, "ytick.color": INK, "text.color": INK, "hatch.linewidth": 0.5,
      "xtick.major.pad": 3.5, "ytick.major.pad": 3.5}

# Okabe-Ito colours by model class.
CLASSES = {"Factorised": "#7F7F7F", "Joint, quadratic-time": "#0072B2",
           "Joint, linear-time": "#D55E00"}
MODELS = {"CNP": ("o", "Factorised"), "NP": ("s", "Factorised"), "ANP": ("^", "Factorised"),
          "CANP": ("D", "Factorised"), "TNPA": ("P", "Joint, quadratic-time"),
          "NDP": ("X", "Joint, quadratic-time"), "FNP": ("v", "Joint, quadratic-time"),
          "MFNP": ("*", "Joint, linear-time")}
FRONTIER = "#9a9a9a"
DASHES = (0, (4, 2))
HATCH = "#F3C2A6"

X_LIMITS = {"utah": (0.08, 6e4), "california": (0.08, 2e5), "us": (0.15, 1.2e7)}
Y_LIMITS = {"utah": {"rmse": (1.45, 2.6), "crps": (0.65, 1.65)},
            "california": {"rmse": (1.25, 2.2), "crps": (0.6, 1.4)},
            "us": {"rmse": (1.4, 3.45), "crps": (0.6, 2.45)}}

# Label offsets in points and alignment (dx, dy, ha, va). Labels sit right of the marker unless
# listed here.
RIGHT = (5, 0, "left", "center")
ABOVE_RIGHT = (4, 4, "left", "bottom")
SMALL_REGIONS = {"CNP": ABOVE_RIGHT, "ANP": (5, 3, "left", "bottom"),
                 "CANP": (-5, -4, "right", "top"), "TNPA": ABOVE_RIGHT}
LABEL_OFFSETS = {
    "utah": {"rmse": {**SMALL_REGIONS, "FNP": (0, 6, "center", "bottom"),
                      "MFNP": (-7, 5, "right", "bottom")},
             "crps": {**SMALL_REGIONS, "FNP": (0, 6, "center", "bottom"),
                      "MFNP": (-7, 5, "right", "bottom")}},
    "california": {"rmse": {**SMALL_REGIONS, "CANP": (-3, -5, "right", "top")},
                   "crps": {**SMALL_REGIONS, "CANP": (0, -6, "center", "top")}},
    "us": {"rmse": {"CNP": (-6, 0, "right", "center"), "CANP": (4, -5, "left", "top")},
           "crps": {"CNP": (-6, 0, "right", "center"), "CANP": (4, -5, "left", "top")}},
}
OURS_OFFSET = (7, 5, "left", "bottom")


def frontier(points):
    """Names of the points that no other point beats in both coordinates, by increasing x.
    points: {name: (x, y)}."""
    front = [n for n, (x, y) in points.items()
             if not any(x2 <= x and y2 <= y and (x2, y2) != (x, y)
                        for m, (x2, y2) in points.items() if m != n)]
    return sorted(front, key=lambda n: points[n][0])


def panel(ax, results, region, metric):
    """One metric against time. results: {name: {metric: (mean, se), "time_ms": (mean, se)}}."""
    for name, r in results.items():
        marker, cls = MODELS[name]
        color = CLASSES[cls]
        ours = name == "MFNP"
        (x, x_se), (y, y_se) = r["time_ms"], r[metric]
        ax.errorbar(x, y, yerr=y_se, xerr=x_se, fmt="none", ecolor=color, elinewidth=0.8,
                    capsize=0, zorder=3)
        ax.plot(x, y, marker, ms=10 if ours else 5.5, mfc=color, mec=INK if ours else "white",
                mew=0.6 if ours else 0.5, zorder=5 if ours else 4)
        dx, dy, ha, va = LABEL_OFFSETS[region][metric].get(name, OURS_OFFSET if ours else RIGHT)
        # The white box hides the hatch behind the label but not the frontier line.
        ax.annotate(r"$\mathbf{MFNP}$ (ours)" if ours else name, (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=8 if ours else 7.5,
                    color=color if ours else INK, bbox=dict(fc="white", ec="none", pad=0.6),
                    zorder=1.5)

    # Staircase from the top edge down through the frontier and on to the right edge. The region
    # below and left of it, which no model reaches, is hatched.
    x_lo, x_hi = X_LIMITS[region]
    y_lo, y_hi = Y_LIMITS[region][metric]
    front = frontier({n: (r["time_ms"][0], r[metric][0]) for n, r in results.items()})
    xs, ys = [], [y_hi]
    for name in front:
        x, y = results[name]["time_ms"][0], results[name][metric][0]
        xs += [x, x]
        ys += [y, y]
    xs.append(x_hi)
    ax.plot(xs, ys, color=FRONTIER, lw=1.2, ls=DASHES, zorder=2)
    ax.fill([x_lo, *xs, x_hi, x_lo], [y_hi, *ys, y_lo, y_lo], facecolor="none",
            edgecolor=HATCH, hatch="////", lw=0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Time per episode (ms)")
    ax.set_ylabel(f"{metric.upper()} ({DEG})")
    ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)


def plot(results, region, prefix):
    """Writes the figure of one region to ``prefix``.pdf and .png."""
    style.apply()
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(style.FIG_W, 2.35), layout="constrained")
        panel(axes[0], results, region, "rmse")
        panel(axes[1], results, region, "crps")
        handles = [Patch(fc=color, ec="none") for color in CLASSES.values()]
        handles.append((Patch(facecolor="none", edgecolor=HATCH, hatch="////", lw=0),
                        Line2D([], [], color=FRONTIER, lw=1.2, ls=DASHES)))
        fig.legend(handles, [*CLASSES, "Pareto frontier"],
                   handler_map={tuple: HandlerTuple(ndivide=1)}, loc="outside upper center",
                   ncol=4, frameon=False, handletextpad=0.4, columnspacing=1.4,
                   handlelength=1.0, handleheight=0.7)
        save(fig, prefix)
