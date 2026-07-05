"""Shared camera-ready figure style (Elsevier / J. Systems Architecture).

Single source of truth for ALL figure styling: rcParams, the unified runtime
palette, figure print sizes, and color helpers. Plot code must pull colors,
labels, sizes, and line styles from here — never hard-code them per plot.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

# --------------------------------------------------------------------------- #
# Unified runtime identity (Okabe–Ito, colorblind-safe)
# --------------------------------------------------------------------------- #
RUNTIME_ORDER = ("plain", "naive", "proposed")
RUNTIME_COLORS = {
    "plain":    "#0072B2",   # blue
    "naive":    "#E69F00",   # orange
    "proposed": "#009E73",   # green
}
RUNTIME_LABELS = {"plain": "Plain", "naive": "Naive", "proposed": "Proposed"}
# linestyle + sparse marker per runtime so series stay readable in grayscale
RUNTIME_STYLES = {
    "plain":    {"linestyle": "-",  "marker": "o"},
    "naive":    {"linestyle": "--", "marker": "s"},
    "proposed": {"linestyle": "-.", "marker": "^"},
}

# latency-decomposition components (Okabe–Ito complements, distinct hues)
COMPONENT_COLORS = {
    "formation_wait":  "#56B4E9",   # sky blue
    "gpu_wait":        "#D55E00",   # vermillion
    "stage1_compute":  "#999999",   # gray
    "seg2_queue_wait": "#CC79A7",   # pink
    "seg2_compute":    "#F0E442",   # yellow
}
COMPONENT_LABELS = {
    "formation_wait":  "Formation wait",
    "gpu_wait":        "GPU wait",
    "stage1_compute":  "Stage-1 compute",
    "seg2_queue_wait": "Stage-2 queue wait",
    "seg2_compute":    "Stage-2 compute",
}

IDLE_COLOR = "#E4E4E4"      # timeline idle segments
STAGE2_TINT = 0.45          # blend-toward-white fraction for stage-2 marks
# neutral swatches for stage legends (dark = stage 1, light = stage 2)
STAGE1_SWATCH = "#555555"
STAGE2_SWATCH = "#BBBBBB"

# final print sizes (inches) — figures must not be rescaled in LaTeX
FIG_SINGLE = (3.5, 2.5)     # single column: plots 1–4, 8
FIG_DOUBLE = (7.2, 2.8)     # double column: plots 5–7


# --------------------------------------------------------------------------- #
def apply_style() -> None:
    """Global rcParams for camera-ready output. Call once at import time."""
    plt.rcParams.update({
        # Times-like serif, embedded TrueType (no Type-3 fonts)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # print-size typography
        "font.size": 8,
        "axes.titlesize": 9,
        "figure.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        # recessive frame: light y-grid only, no top/right spines
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # marks
        "lines.linewidth": 1.3,
        "lines.markersize": 3.2,
        "legend.frameon": False,
        # layout / output
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
    })


def lighten(color, amount: float = STAGE2_TINT):
    """Blend `color` toward white by `amount` (0 = unchanged, 1 = white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def darken(color, amount: float):
    """Blend `color` toward black by `amount`."""
    r, g, b = to_rgb(color)
    f = 1.0 - amount
    return (r * f, g * f, b * f)


def proposed_shades(n: int):
    """Sequential shades of the proposed base color, light -> dark."""
    base = RUNTIME_COLORS["proposed"]
    if n <= 1:
        return [to_rgb(base)]
    fracs = np.linspace(0.55, -0.3, n)
    return [lighten(base, f) if f >= 0 else darken(base, -f) for f in fracs]


def b2_label(B: int) -> str:
    return f"$b_2 = {B}$"
