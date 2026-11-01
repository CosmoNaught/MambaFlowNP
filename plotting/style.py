"""Fonts, sizes and output format shared by the figures, set for a print width of 5.5 in."""

import matplotlib.pyplot as plt

FIG_W = 5.5
INK = "#222222"
DEG = r"$^{\circ}$C"


def apply():
    """Sets the paper's fonts and base sizes."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
    })


def save(fig, prefix):
    """Writes ``prefix``.pdf and ``prefix``.png at 300 dpi and closes the figure."""
    for suffix in (".pdf", ".png"):
        fig.savefig(prefix.with_suffix(suffix), dpi=300)
    plt.close(fig)
