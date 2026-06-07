"""Shared visual theme for the problem-set figures.

`apply_style()` sets a small, cohesive set of matplotlib rcParams so every
generated figure shares the same modern look as the notebook's markdown
cards (deep-blue titles, slate text, white cards, crisp DPI).  It is applied
automatically when `pset_tools` is imported; call it again yourself if some
other code has clobbered rcParams.
"""

from __future__ import annotations

import matplotlib as mpl

# Palette shared with the notebook's HTML callout cards.
INK        = "#1f2933"   # primary text
SLATE      = "#475569"   # secondary text / labels
MUTED      = "#64748b"   # captions / ticks
TITLE_BLUE = "#1e3a8a"   # figure titles (matches the banner)
ACCENT     = "#2563eb"   # accent blue
CARD_EDGE  = "#cbd5e1"   # light slate borders
CARD_FILL  = "#f8fafc"   # very light card fill

# Semantic colours (kept consistent everywhere: X / Y / Z, snake, lattice).
X_RED   = "#c84e4e"
Y_GREEN = "#5a9c4a"
Z_BLUE  = "#1f497d"


def apply_style() -> None:
    """Apply the shared rcParams theme (idempotent)."""
    mpl.rcParams.update({
        # canvas
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
        "figure.dpi":        110,
        "savefig.dpi":       120,
        # type
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "text.color":        INK,
        # titles
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   TITLE_BLUE,
        "axes.titlepad":     10,
        # axes (most pset figures hide these, but keep them tasteful)
        "axes.edgecolor":    CARD_EDGE,
        "axes.labelcolor":   SLATE,
        "axes.linewidth":    1.0,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "legend.frameon":    False,
    })


# Apply on import so figures look themed without any extra student action.
apply_style()
