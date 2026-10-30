"""Per-region latency against RMSE and CRPS, with the Pareto frontier. Called by figures.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DISPLAY_NAMES = {
    "cnp": "CNP", "np": "NP", "canp": "CANP", "anp": "ANP",
    "tnpa": "TNPA", "ndp": "NDP", "fnp": "FNP", "mfnp": "MFNP (ours)",
}
FAMILY_BY_MODEL = {
    "cnp": "CNP + NP", "np": "CNP + NP",
    "canp": "CANP + ANP", "anp": "CANP + ANP",
    "tnpa": "FlowNP + TNPA + NDP", "ndp": "FlowNP + TNPA + NDP", "fnp": "FlowNP + TNPA + NDP",
    "mfnp": "MFNP",
}
FAMILY_COLORS = {
    "CNP + NP": "#4C78A8",
    "CANP + ANP": "#F58518",
    "FlowNP + TNPA + NDP": "#54A24B",
    "MFNP": "#E45756",
}
MODEL_MARKERS = {
    "cnp": "o", "np": "s", "canp": "D", "anp": "^",
    "fnp": "v", "tnpa": "P", "ndp": "X", "mfnp": "*",
}
MODEL_MARKER_SIZE = {
    "cnp": 65, "np": 65, "canp": 65, "anp": 70,
    "fnp": 90, "tnpa": 95, "ndp": 95, "mfnp": 220,
}


def _is_pareto_optimal(points: list[tuple[float, float]]) -> list[bool]:
    """Mask of Pareto-optimal points minimising both x (latency) and y (metric)."""
    n = len(points)
    is_opt = [True] * n
    for i in range(n):
        if not is_opt[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (points[j][0] <= points[i][0] and points[j][1] <= points[i][1]
                    and (points[j][0] < points[i][0] or points[j][1] < points[i][1])):
                is_opt[i] = False
                break
    return is_opt


# Label offsets in points and horizontal alignment, per region and metric.
LABEL_OFFSETS: dict[tuple[str, str], dict[str, tuple[int, int, str]]] = {
    ("utah", "rmse"): {
        "cnp":  (-10, 0, "right"),
        "np":   (0, 12, "center"),
        "canp": (-10, 0, "right"),
        "anp":  (10, 8, "left"),
        "tnpa": (-12, -8, "right"),
        "ndp":  (-12, 8, "right"),
        "fnp":  (0, 12, "center"),
        "mfnp": (12, 6, "left"),
    },
    ("utah", "crps"): {
        "cnp":  (-10, 0, "right"),
        "np":   (0, 12, "center"),
        "canp": (0, -12, "center"),
        "anp":  (10, 8, "left"),
        "tnpa": (-12, -8, "right"),
        "ndp":  (-12, 8, "right"),
        "fnp":  (0, 12, "center"),
        "mfnp": (12, 6, "left"),
    },
    ("california", "rmse"): {
        "cnp":  (-10, 0, "right"),
        "np":   (0, 12, "center"),
        "canp": (0, -12, "center"),
        "anp":  (10, 8, "left"),
        "tnpa": (-12, 0, "right"),
        "ndp":  (-12, 10, "right"),
        "fnp":  (12, -2, "left"),
        "mfnp": (12, 6, "left"),
    },
    ("california", "crps"): {
        "cnp":  (-10, 0, "right"),
        "np":   (0, 12, "center"),
        "canp": (0, -12, "center"),
        "anp":  (10, 8, "left"),
        "tnpa": (-12, 0, "right"),
        "ndp":  (-12, 10, "right"),
        "fnp":  (12, -2, "left"),
        "mfnp": (12, 6, "left"),
    },
    ("us", "rmse"): {
        "cnp":  (-10, -8, "right"),
        "canp": (-10, -8, "right"),
        "anp":  (-10, -10, "right"),
        "np":   (10, 6, "left"),
        "ndp":  (-12, 10, "right"),
        "fnp":  (-12, -10, "right"),
        "mfnp": (12, 6, "left"),
    },
    ("us", "crps"): {
        "cnp":  (-10, -8, "right"),
        "canp": (-10, -8, "right"),
        "anp":  (-10, -10, "right"),
        "np":   (10, 6, "left"),
        "ndp":  (-12, 10, "right"),
        "fnp":  (-12, -10, "right"),
        "mfnp": (12, 6, "left"),
    },
}


def plot_panel(ax, tier_key: str, tier_data: dict, metric_key: str, metric_label: str) -> None:
    pareto_points: list[tuple[float, float]] = []

    for model, vals in tier_data["models"].items():
        x_mean, _ = vals["time_ms"]
        y_mean, _ = vals[metric_key]
        family = FAMILY_BY_MODEL[model]
        color = FAMILY_COLORS[family]
        marker = MODEL_MARKERS[model]
        is_ours = model == "mfnp"
        ms = MODEL_MARKER_SIZE[model]
        zorder = 10 if is_ours else 5
        alpha = 1.0 if is_ours else 0.85
        edgecolor = "black" if is_ours else "white"
        edgewidth = 1.2 if is_ours else 0.6

        ax.scatter(
            x_mean, y_mean,
            s=ms, marker=marker,
            color=color, edgecolors=edgecolor, linewidths=edgewidth,
            alpha=alpha, zorder=zorder,
        )
        ox, oy, ha = LABEL_OFFSETS[(tier_key, metric_key)].get(model, (10, 8, "left"))
        ax.annotate(
            DISPLAY_NAMES[model],
            xy=(x_mean, y_mean),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=9.5 if is_ours else 8.5,
            fontweight="bold" if is_ours else "normal",
            color=color,
            ha=ha, va="center",
            zorder=zorder + 1,
        )
        pareto_points.append((x_mean, y_mean))

    ax.set_xscale("log")
    ax.set_xlabel("Inference time per dataset (ms)", fontsize=10.5)
    ax.set_ylabel(metric_label, fontsize=10.5)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.6, color="#cccccc")
    ax.grid(True, which="minor", alpha=0.07, linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)

    # Pad the axes before drawing the frontier so that it reaches the final limits.
    xlims = ax.get_xlim()
    ax.set_xlim(xlims[0] / 3.5, xlims[1] * 3.5)
    ylims = ax.get_ylim()
    pad = 0.10 * (ylims[1] - ylims[0])
    ax.set_ylim(ylims[0] - pad, ylims[1] + pad)

    # Pareto staircase from the top of the axes, through the optimal points, to the right edge.
    if pareto_points:
        opt_mask = _is_pareto_optimal(pareto_points)
        front_pts = sorted(
            [p for p, o in zip(pareto_points, opt_mask) if o],
            key=lambda p: p[0],
        )
        if front_pts:
            xlo, xhi = ax.get_xlim()
            ylo, yhi = ax.get_ylim()
            front_xs = [front_pts[0][0]]
            front_ys = [yhi]
            for i, (px, py) in enumerate(front_pts):
                front_xs.append(px); front_ys.append(py)
                if i + 1 < len(front_pts):
                    nxt_x = front_pts[i + 1][0]
                    front_xs.append(nxt_x); front_ys.append(py)
            front_xs.append(xhi); front_ys.append(front_pts[-1][1])
            ax.plot(
                front_xs, front_ys,
                color="#666666", linewidth=1.6, linestyle="--",
                alpha=0.9, zorder=2, label="Pareto frontier",
            )


def plot_one_tier(tier_key: str, tier_data: dict, out_prefix: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Nimbus Roman", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    plot_panel(axes[0], tier_key, tier_data, "rmse",
               r"RMSE ($^{\circ}$C)")
    plot_panel(axes[1], tier_key, tier_data, "crps",
               r"CRPS ($^{\circ}$C)")

    fig.suptitle(tier_data["label"], fontsize=13, fontweight="bold", y=1.00)

    legend_models = list(tier_data["models"].keys())
    legend_elements: list[Line2D] = []
    for model in legend_models:
        family = FAMILY_BY_MODEL[model]
        color = FAMILY_COLORS[family]
        is_ours = model == "mfnp"
        legend_elements.append(
            Line2D(
                [0], [0], color="none",
                marker=MODEL_MARKERS[model],
                markersize=11 if is_ours else 7.5,
                markerfacecolor=color,
                markeredgecolor="black" if is_ours else "white",
                markeredgewidth=0.9 if is_ours else 0.4,
                label=DISPLAY_NAMES[model],
            )
        )
    legend_elements.append(
        Line2D(
            [0], [0], color="#888888", linewidth=1.4, linestyle="--",
            label="Pareto frontier",
        )
    )
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=1.2,
        handletextpad=0.4,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_prefix.with_suffix('.pdf')}")
    print(f"wrote {out_prefix.with_suffix('.png')}")

