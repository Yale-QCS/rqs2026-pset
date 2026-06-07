"""Illustrative figures for the problem set.

`snake_layout_figure()` — pale-yellow 3x3 qubit grid with arrows tracing
the snake JW order  q0 -> q1 -> ... -> q8.  Matches the placeholder text
in the markdown background.

`fermion_lattice_figure()` — pale-green 3x3 lattice of fermionic modes
in raster (row-major) order with the 12 nearest-neighbor hopping edges
drawn as undirected links.  This is the Hamiltonian interaction graph for
Problem 4.

The style matches the FIG. 1 vibe of the companion paper: light-blue/green
filled circles, soft gray edges, large readable labels.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt

# colors picked to mirror the paper's color choices
SNAKE_FILL = "#fff0c2"        # pale yellow
SNAKE_EDGE = "#b88800"
SNAKE_ARROW = "#7a5a00"
LATTICE_FILL = "#d6efd0"      # pale green
LATTICE_EDGE = "#2c7a3a"
LATTICE_LINK = "#7a7a7a"


def _common_axes(ax, span=3, pad=0.55):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-pad, span - 1 + pad)
    ax.set_ylim(-(span - 1 + pad), pad)


def _draw_node(ax, x, y, label, fill, edge, radius=0.34, fontsize=13):
    circ = patches.Circle(
        (x, y), radius, facecolor=fill, edgecolor=edge,
        linewidth=2.0, zorder=3,
    )
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#1a1a1a", zorder=4)


# ---------------------------------------------------------------------------
# Snake JW figure
# ---------------------------------------------------------------------------

# Snake order: q_0..q_8 mapped to (row, col) on a 3x3 grid:
#   q0 q1 q2
#   q5 q4 q3
#   q6 q7 q8
SNAKE_RC = [
    (0, 0), (0, 1), (0, 2),
    (1, 2), (1, 1), (1, 0),
    (2, 0), (2, 1), (2, 2),
]


def _curved_arrow(ax, p, q, color, rad=0.0):
    arr = patches.FancyArrowPatch(
        p, q,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=1.4, zorder=2,
    )
    ax.add_patch(arr)


def snake_layout_figure(figsize=(5.6, 5.6), ax=None,
                        title: Optional[str] = "Snake JW qubit layout"):
    """Draw the 3x3 qubit snake (yellow circles labeled q_0..q_8 connected
    by snake-order arrows)."""
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        new_fig = True
    _common_axes(ax)
    subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    # nodes
    for i, (r, c) in enumerate(SNAKE_RC):
        # invert y so row 0 is at the top
        _draw_node(ax, c, -r, f"q{str(i).translate(subs)}",
                   SNAKE_FILL, SNAKE_EDGE)

    # arrows (snake order)
    for i in range(len(SNAKE_RC) - 1):
        r1, c1 = SNAKE_RC[i]
        r2, c2 = SNAKE_RC[i + 1]
        p = (c1, -r1)
        q = (c2, -r2)
        # offset endpoints so arrows don't overlap the circles
        dx = c2 - c1
        dy = -(r2 - r1)
        # unit vector
        import math as _m
        L = _m.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        p_off = (p[0] + 0.36 * ux, p[1] + 0.36 * uy)
        q_off = (q[0] - 0.36 * ux, q[1] - 0.36 * uy)
        _curved_arrow(ax, p_off, q_off, SNAKE_ARROW, rad=0.0)

    if title:
        ax.set_title(title, fontsize=13, pad=8)
    if new_fig:
        return fig, ax
    return ax


# ---------------------------------------------------------------------------
# Fermion mode lattice (Problem 4)
# ---------------------------------------------------------------------------

# The mode lattice uses the SAME 3x3 grid as the snake JW layout: mode i
# sits at snake cell i.  Drawing it as a clean square grid, the labels read
#       0  1  2
#       5  4  3
#       6  7  8
# Horizontal bonds connect modes that are JW-adjacent (cheap); vertical
# bonds connect modes that are JW-far (expensive).
HORIZONTAL_LATTICE_EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
VERTICAL_LATTICE_EDGES = [(0, 5), (5, 6), (1, 4), (4, 7), (2, 3), (3, 8)]


def fermion_lattice_figure(figsize=(5.6, 5.6), ax=None,
                           title: Optional[str] = "Fermion mode lattice",
                           highlight_vertical: bool = True):
    """Draw the 3x3 fermion-mode lattice (snake labelling) with its 12
    nearest-neighbor hopping bonds.  Vertical bonds are drawn thicker /
    coloured so the student can see which hops are the expensive ones."""
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        new_fig = True
    _common_axes(ax)

    # horizontal bonds (cheap)
    for (i, j) in HORIZONTAL_LATTICE_EDGES:
        r1, c1 = SNAKE_RC[i]
        r2, c2 = SNAKE_RC[j]
        ax.plot([c1, c2], [-r1, -r2], color=LATTICE_LINK,
                linewidth=2.0, zorder=1)
    # vertical bonds (expensive) -- optionally highlighted
    for (i, j) in VERTICAL_LATTICE_EDGES:
        r1, c1 = SNAKE_RC[i]
        r2, c2 = SNAKE_RC[j]
        ax.plot([c1, c2], [-r1, -r2],
                color="#c46a2a" if highlight_vertical else LATTICE_LINK,
                linewidth=3.0 if highlight_vertical else 2.0,
                zorder=2 if highlight_vertical else 1)

    # nodes at snake cells
    for i, (r, c) in enumerate(SNAKE_RC):
        _draw_node(ax, c, -r, str(i), LATTICE_FILL, LATTICE_EDGE)

    if title:
        ax.set_title(title, fontsize=13, pad=8)
    if new_fig:
        return fig, ax
    return ax


# ---------------------------------------------------------------------------
# Side-by-side comparison (used in Problem 4 intro)
# ---------------------------------------------------------------------------

def snake_and_lattice_figure(figsize=(11, 5.4)):
    """Render the two views of the SAME 3x3 grid side by side: (a) the snake
    JW threading order, (b) the mode lattice with its 12 hopping bonds
    (vertical bonds highlighted as the expensive ones)."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    snake_layout_figure(ax=ax_l, title="(a) Snake JW order (threading)")
    fermion_lattice_figure(
        ax=ax_r,
        title="(b) Mode lattice — vertical bonds (orange) are expensive")
    return fig, (ax_l, ax_r)


# ---------------------------------------------------------------------------
# Permutation-on-the-snake figure (Problem 3)
# ---------------------------------------------------------------------------

# Distinct colors for the swap pairs.
_PAIR_COLORS = ["#c84e4e", "#5a9c4a", "#8155b5", "#d08a2a"]


def permutation_on_snake_figure(
    pairs=((1, 5), (2, 6), (3, 7)),
    fixed=(0, 4, 8),
    figsize=(6.4, 6.4),
    ax=None,
    title: Optional[str] = "Problem 3 permutation on the snake",
):
    """Draw the snake JW grid and show a permutation as swap-pair arcs.

    `pairs` are exchanged (each pair gets its own color + double arrow);
    `fixed` points are drawn with a heavier outline so the reader can see
    they stay put.  Mode label == JW position (the snake index), matching
    the layout used throughout the pset.
    """
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        new_fig = True
    _common_axes(ax, pad=0.75)

    fixed_set = set(fixed)

    def pos(jw_index):
        r, c = SNAKE_RC[jw_index]
        return c, -r

    # faint snake-order arrows for context
    for i in range(len(SNAKE_RC) - 1):
        (x1, y1), (x2, y2) = pos(i), pos(i + 1)
        import math as _m
        dx, dy = x2 - x1, y2 - y1
        L = _m.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        ax.plot([x1 + 0.36 * ux, x2 - 0.36 * ux],
                [y1 + 0.36 * uy, y2 - 0.36 * uy],
                color="#d8c98a", linewidth=1.2, zorder=1, alpha=0.8)

    # swap-pair arcs
    for k, (a, b) in enumerate(pairs):
        col = _PAIR_COLORS[k % len(_PAIR_COLORS)]
        (xa, ya), (xb, yb) = pos(a), pos(b)
        arc = patches.FancyArrowPatch(
            (xa, ya), (xb, yb),
            connectionstyle="arc3,rad=0.28",
            arrowstyle="<|-|>", mutation_scale=14,
            color=col, linewidth=2.2, zorder=4,
            shrinkA=16, shrinkB=16,
        )
        ax.add_patch(arc)

    # nodes
    for i, (r, c) in enumerate(SNAKE_RC):
        is_fixed = i in fixed_set
        circ = patches.Circle(
            (c, -r), 0.34,
            facecolor="#e9e9ee" if is_fixed else SNAKE_FILL,
            edgecolor="#555" if is_fixed else SNAKE_EDGE,
            linewidth=2.6 if is_fixed else 2.0, zorder=3,
        )
        ax.add_patch(circ)
        ax.text(c, -r, str(i), ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1a1a1a", zorder=4)

    # legend of pairs
    legend_txt = "swap:  " + "   ".join(
        f"{a}↔{b}" for (a, b) in pairs
    ) + f"     fixed:  {', '.join(map(str, fixed))}"
    ax.text(1.0, 0.62, legend_txt, ha="center", va="bottom",
            fontsize=9.5, color="#444")

    if title:
        ax.set_title(title, fontsize=13, pad=10)
    if new_fig:
        return fig, ax
    return ax
