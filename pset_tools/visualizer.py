"""Majorana-tree visualizer.

Two complementary views of a Jordan-Wigner circuit's effect, both driven by
the same Clifford-conjugation engine (`pset_tools.pauli`):

* **tree view (default)** — `draw_majorana_tree(circuit, step=…)` rebuilds and
  draws the *current* decorated ternary tree.  As gates are applied the tree
  **morphs**: all three components are free to change —

    - **shape** (CZ/CNOT/CY are tree rotations),
    - **qubit labels** on the internal nodes (a physical SWAP swaps two of them),
    - **Majorana labels** γ_p on the leaves (a fermionic SWAP swaps two leaf-pairs).

  The tree is reconstructed from the operators themselves (`pset_tools.tree`),
  so the picture is correct *no matter how a gate was decomposed* — only the
  endpoint Majorana operators matter, and they pin down one unique tree.

* **strings view** — `draw_majorana_tree(..., mode="strings")` keeps the JW tree
  fixed and writes each Majorana's current Pauli string inside its leaf box
  (the original FIG.-6 style).  Useful when a circuit leaves "tree land"
  (a 2-qubit gate applied off a tree edge has no ternary-tree representation);
  the tree view falls back to this automatically.

`interactive_majorana_tree(circuit)` wraps the drawing in an ipywidgets slider;
`animate_majorana_tree(circuit)` redraws frame-by-frame in place; and
`play_majorana_tree(circuit)` returns a self-contained, continuously
**tweened** HTML animation (nodes and leaves glide to their new positions).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from .jw import N_MODES, initial_majorana
from .pauli import PauliString, conjugate_through
from .tree import (
    Layout, NonTreeMapping, layout_tree, reconstruct_tree,
)

# --- color palette (shared with pset_tools.style) ---------------------------
NODE_FILL = "#cfe1f3"      # pale blue
NODE_EDGE = "#3163a3"
SPINE_Z   = "#1f497d"      # deep blue for Z edges
LEAF_X    = "#c84e4e"      # warm red for X edges
LEAF_Y    = "#5a9c4a"      # green for Y edges
BOX_FILL  = "#f8fafc"      # very light slate card (matches markdown cards)
BOX_EDGE  = "#cbd5e1"      # light slate border
GAMMA_COL = "#475569"      # slate for gamma labels
TITLE_COL = "#1e3a8a"      # deep blue title (matches the banner)
MUTED_COL = "#64748b"      # caption / ruler text

LETTER_EDGE_COLOR = {"X": LEAF_X, "Y": LEAF_Y, "Z": SPINE_Z}

_SUBS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

from ._ops import flatten_ops as _as_ops  # lists may be nested (gadgets)


# ---------------------------------------------------------------------------
# Compute Majorana strings at a given gate index
# ---------------------------------------------------------------------------

def majorana_strings_after(circuit, step: Optional[int] = None) -> List[PauliString]:
    """Return the 2N Majorana Pauli strings after applying the first
    `step` ops of `circuit`.  `step=None` means "apply all gates"."""
    ops = _as_ops(circuit)
    if step is None:
        step = len(ops)
    step = max(0, min(step, len(ops)))
    sub = ops[:step]
    return [conjugate_through(initial_majorana(p), sub)
            for p in range(2 * N_MODES)]


def reconstruct_after(circuit, step: Optional[int] = None):
    """Convenience: Majorana strings after `step` gates → reconstructed tree.

    Returns the root `Node`.  Raises `NonTreeMapping` if the mapping is not a
    ternary tree, or `ValueError` if the circuit contains a gate the
    conjugation engine does not support (e.g. a continuous rotation)."""
    return reconstruct_tree(majorana_strings_after(circuit, step))


# ---------------------------------------------------------------------------
# Low-level painters
# ---------------------------------------------------------------------------

def _draw_node(ax, x: float, y: float, label: str, radius: float = 0.34,
               alpha: float = 1.0):
    circ = patches.Circle(
        (x, y), radius, facecolor=NODE_FILL, edgecolor=NODE_EDGE,
        linewidth=1.5, zorder=5, alpha=alpha,
    )
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center", fontsize=10,
            fontweight="bold", color="#1a1a1a", zorder=6, alpha=alpha)


def _trim(x0, y0, x1, y1, r0, r1):
    dx, dy = x1 - x0, y1 - y0
    L = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / L, dy / L
    return x0 + ux * r0, y0 + uy * r0, x1 - ux * r1, y1 - uy * r1


def _paint_tree_frame(ax, nodes: Dict[int, Tuple[float, float, float]],
                      leaves: Dict[int, Tuple[float, float, float]],
                      edges, leaf_sign, node_r: float = 0.30,
                      edge_labels: bool = True, leaf_labels: bool = True):
    """Paint one tree frame.

    nodes:  {qubit -> (x, y, alpha)}
    leaves: {gamma -> (x, y, alpha)}
    edges:  iterable of (parent_ref, child_ref, letter, alpha) where a ref is
            ("node", q) or ("leaf", p).

    Each edge is a straight coloured line (X red / Y green / Z blue) carrying a
    small X/Y/Z letter; edges to subtrees are drawn a touch heavier than edges
    to bare leaves.
    """
    pos: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for q, (x, y, _a) in nodes.items():
        pos[("node", q)] = (x, y)
    for p, (x, y, _a) in leaves.items():
        pos[("leaf", p)] = (x, y)

    # edges first, so nodes overlay their endpoints
    for pref, cref, letter, alpha in edges:
        if pref not in pos or cref not in pos or alpha <= 0.01:
            continue
        x0, y0 = pos[pref]
        x1, y1 = pos[cref]
        col = LETTER_EDGE_COLOR[letter]
        is_node = cref[0] == "node"
        r1 = node_r if is_node else 0.05
        sx0, sy0, sx1, sy1 = _trim(x0, y0, x1, y1, node_r, r1)
        ax.plot([sx0, sx1], [sy0, sy1], color=col,
                linewidth=2.4 if is_node else 1.9, alpha=alpha, zorder=2,
                solid_capstyle="round")
        if edge_labels and alpha > 0.35:
            lx, ly = sx0 + 0.46 * (sx1 - sx0), sy0 + 0.46 * (sy1 - sy0)
            ax.text(lx, ly, letter, color=col, fontsize=8, fontweight="bold",
                    ha="center", va="center", alpha=alpha, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                              edgecolor="none", alpha=0.82 * alpha))

    # leaves (γ labels + sign + a small endpoint dot)
    for p, (x, y, a) in leaves.items():
        if a <= 0.01:
            continue
        s = leaf_sign.get(p, 1 + 0j)
        if abs(s + 1) < 1e-9:
            pre, col = "−", "#b03030"
        elif abs(s - 1j) < 1e-9:
            pre, col = "i", GAMMA_COL
        elif abs(s + 1j) < 1e-9:
            pre, col = "−i", GAMMA_COL
        else:
            pre, col = "", GAMMA_COL
        ax.plot([x], [y], marker="o", ms=3.5, color="#9aa3af", alpha=a, zorder=3)
        if leaf_labels:
            ax.text(x, y - 0.30, f"{pre}γ{str(p).translate(_SUBS)}",
                    ha="center", va="top", fontsize=8.5, color=col, alpha=a,
                    fontstyle="italic", zorder=4)

    # nodes on top
    for q, (x, y, a) in nodes.items():
        if a <= 0.01:
            continue
        _draw_node(ax, x, y, f"q{str(q).translate(_SUBS)}", radius=node_r, alpha=a)


def _frame_from_layout(lo: Layout):
    return dict(
        nodes={q: (x, y, 1.0) for q, (x, y) in lo.node_xy.items()},
        leaves={p: (x, y, 1.0) for p, (x, y) in lo.leaf_xy.items()},
        edges=[(pr, cr, L, 1.0) for pr, cr, L in lo.edges],
        leaf_sign=dict(lo.leaf_sign),
    )


def _frame_interp(A: Layout, B: Layout, u: float):
    """Crossfade layout A → B at parameter u ∈ [0, 1]; shared ids glide."""
    pa, pb = A.positions(), B.positions()
    ida, idb = set(pa), set(pb)
    shared, onlya, onlyb = ida & idb, ida - idb, idb - ida
    nodes: Dict[int, Tuple[float, float, float]] = {}
    leaves: Dict[int, Tuple[float, float, float]] = {}

    def put(ref, x, y, a):
        (nodes if ref[0] == "node" else leaves)[ref[1]] = (x, y, a)

    for r in shared:
        ax_, ay_ = pa[r]
        bx_, by_ = pb[r]
        put(r, ax_ + (bx_ - ax_) * u, ay_ + (by_ - ay_) * u, 1.0)
    for r in onlya:
        put(r, *pa[r], 1.0 - u)
    for r in onlyb:
        put(r, *pb[r], u)

    ea = {(pr, cr, L) for pr, cr, L in A.edges}
    eb = {(pr, cr, L) for pr, cr, L in B.edges}
    edges = [(*e, 1.0) for e in (ea & eb)]
    edges += [(*e, 1.0 - u) for e in (ea - eb)]
    edges += [(*e, u) for e in (eb - ea)]
    leaf_sign = {**A.leaf_sign, **B.leaf_sign}
    return dict(nodes=nodes, leaves=leaves, edges=edges, leaf_sign=leaf_sign)


# ---------------------------------------------------------------------------
# Strings view (original FIG.-6 style: fixed JW tree, Pauli string per leaf)
# ---------------------------------------------------------------------------

LETTER_COLOR = {"X": LEAF_X, "Y": LEAF_Y, "Z": SPINE_Z, "·": "#c4c4cc"}


def _draw_vertical_leaf(ax, x_center, y_top, rh, box_w, ps, n_qubits=N_MODES,
                        fontsize=8.5):
    y_bot = y_top - n_qubits * rh
    rect = patches.FancyBboxPatch(
        (x_center - box_w / 2, y_bot), box_w, y_top - y_bot,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=1.0, zorder=3,
    )
    ax.add_patch(rect)
    for q in range(n_qubits):
        ch = ps.letter_at(q)
        if ch == "I":
            ch = "·"
        y_row = y_top - (q + 0.5) * rh
        ax.text(x_center, y_row, ch, ha="center", va="center",
                fontsize=fontsize, family="DejaVu Sans Mono",
                fontweight="bold" if ch != "·" else "normal",
                color=LETTER_COLOR.get(ch, "#1a1a1a"), zorder=4)
    return y_bot


def _draw_strings_view(ax, strings, step, n_ops, title, frame_border, note=None):
    n_leaves = 2 * N_MODES
    cw, box_w, rh, node_r, lev, bstub = 0.92, 0.66, 0.34, 0.30, 1.00, 1.05
    dotted_min, y_spine_top = 0.32, 0.0

    def leaf_x(col):
        return col * cw

    def q_xy(j):
        return (2 * j + 0.5) * cw, y_spine_top - j * lev

    y_box_top = q_xy(N_MODES - 1)[1] - bstub - dotted_min
    y_box_bot = y_box_top - N_MODES * rh
    y_gamma = y_box_bot - 0.40

    ax.set_aspect("equal")
    ax.axis("off")

    for j in range(N_MODES - 1):
        x0, y0 = q_xy(j)
        x1, y1 = q_xy(j + 1)
        ux, uy = (x1 - x0), (y1 - y0)
        L = (ux * ux + uy * uy) ** 0.5
        ux, uy = ux / L, uy / L
        ax.plot([x0 + node_r * ux, x1 - node_r * ux],
                [y0 + node_r * uy, y1 - node_r * uy],
                color=SPINE_Z, linewidth=2.2, zorder=2)
        ax.text((x0 + x1) / 2 + 0.10, (y0 + y1) / 2 + 0.12, "Z",
                ha="left", va="bottom", fontsize=10, color=SPINE_Z,
                fontweight="bold", zorder=4)

    x_last, y_last = q_xy(N_MODES - 1)
    gx, gy = q_xy(N_MODES)
    ux, uy = (gx - x_last), (gy - y_last)
    L = (ux * ux + uy * uy) ** 0.5
    ux, uy = ux / L, uy / L
    ax.plot([x_last + node_r * ux, gx - node_r * ux],
            [y_last + node_r * uy, gy - node_r * uy],
            color=SPINE_Z, linewidth=1.8, linestyle=":", zorder=2, alpha=0.55)
    ax.add_patch(patches.Circle(
        (gx, gy), node_r, facecolor="none", edgecolor="#aaa",
        linewidth=1.3, linestyle=(0, (3, 2)), zorder=3))
    ax.text(gx, gy - node_r - 0.16, "unused", ha="center", va="top",
            fontsize=8, color="#999", zorder=3)

    for j in range(N_MODES):
        x_q, y_q = q_xy(j)
        cx_X = leaf_x(2 * j)
        cx_Y = leaf_x(2 * j + 1)
        lbl_bbox = dict(boxstyle="round,pad=0.05", facecolor="white",
                        edgecolor="none", alpha=0.9)
        ex_X, ey_X = cx_X, y_q - bstub
        ax.plot([x_q - node_r * 0.45, ex_X], [y_q - node_r * 0.9, ey_X],
                color=LEAF_X, linewidth=1.8, zorder=2)
        ax.plot([cx_X, cx_X], [ey_X, y_box_top],
                color=LEAF_X, linewidth=1.0, linestyle=":", zorder=1, alpha=0.7)
        ax.text(cx_X, ey_X + 0.05, "X", ha="center", va="bottom",
                fontsize=10, color=LEAF_X, fontweight="bold", zorder=5,
                bbox=lbl_bbox)
        ex_Y, ey_Y = cx_Y, y_q - bstub
        ax.plot([x_q + node_r * 0.45, ex_Y], [y_q - node_r * 0.9, ey_Y],
                color=LEAF_Y, linewidth=1.8, zorder=2)
        ax.plot([cx_Y, cx_Y], [ey_Y, y_box_top],
                color=LEAF_Y, linewidth=1.0, linestyle=":", zorder=1, alpha=0.7)
        ax.text(cx_Y, ey_Y + 0.05, "Y", ha="center", va="bottom",
                fontsize=10, color=LEAF_Y, fontweight="bold", zorder=5,
                bbox=lbl_bbox)
        _draw_node(ax, x_q, y_q, f"q{str(j).translate(_SUBS)}", radius=node_r)
        _draw_vertical_leaf(ax, cx_X, y_box_top, rh, box_w, strings[2 * j])
        _draw_vertical_leaf(ax, cx_Y, y_box_top, rh, box_w, strings[2 * j + 1])
        for cx, p_idx in ((cx_X, 2 * j), (cx_Y, 2 * j + 1)):
            ax.annotate("", xy=(cx, y_box_bot - 0.04),
                        xytext=(cx, y_gamma + 0.16),
                        arrowprops=dict(arrowstyle="->", color=GAMMA_COL, lw=0.9))
            ax.text(cx, y_gamma, f"γ{str(p_idx).translate(_SUBS)}",
                    ha="center", va="top", fontsize=9, color=GAMMA_COL,
                    fontstyle="italic", zorder=4)

    ruler_x = -1.05 * cw
    ax.text(ruler_x, y_box_top + 0.18, "qubit", ha="center", va="bottom",
            fontsize=8, color=MUTED_COL)
    for q in range(N_MODES):
        ax.text(ruler_x, y_box_top - (q + 0.5) * rh, str(q),
                ha="center", va="center", fontsize=8, color=MUTED_COL)
    ax.text(ruler_x, y_gamma, "Majorana →", ha="center", va="top",
            fontsize=7.5, color=MUTED_COL)

    if title is None:
        title = f"Majorana strings (after step {step}/{n_ops})"
    y_title = y_spine_top + 1.4
    ax.text(ruler_x, y_title, title, ha="left", va="bottom",
            fontsize=13.5, fontweight="bold", color=TITLE_COL)
    sub = ("Leaves share one row; each shows its Majorana's Pauli string "
           "vertically (qubit 0 top … qubit 8 bottom; · = identity).")
    if note:
        sub = note + "  " + sub
    ax.text(ruler_x, y_title - 0.42, sub, ha="left", va="top",
            fontsize=8.5, color=MUTED_COL)

    ax.set_xlim(ruler_x - 0.9, q_xy(N_MODES)[0] + node_r + 0.7)
    ax.set_ylim(y_gamma - 0.6, y_title + 0.7)
    if frame_border:
        _frame_border(ax)


# ---------------------------------------------------------------------------
# Tree view (reconstructed morphing tree)
# ---------------------------------------------------------------------------

def _layout_extent(lo: Layout, pad=1.0):
    xs = [x for x, _ in lo.node_xy.values()] + [x for x, _ in lo.leaf_xy.values()]
    ys = [y for _, y in lo.node_xy.values()] + [y for _, y in lo.leaf_xy.values()]
    if lo.parity is not None:
        (px, py) = lo.parity[1]
        xs.append(px)
        ys.append(py)
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + 0.9)


def _draw_parity_leaf(ax, lo: Layout, node_r=0.30):
    """Draw the single omitted (parity) leaf as a faint grey dashed circle —
    completing the 2N+1-leaf ternary tree, matching the strings view."""
    if lo.parity is None:
        return
    (fx, fy), (px, py), letter = lo.parity
    sx0, sy0, sx1, sy1 = _trim(fx, fy, px, py, node_r, 0.18)
    ax.plot([sx0, sx1], [sy0, sy1], color="#b8b8b8", linewidth=1.3,
            linestyle=(0, (3, 2)), alpha=0.8, zorder=1)
    ax.add_patch(patches.Circle((px, py), 0.18, facecolor="none",
                                edgecolor="#b8b8b8", linewidth=1.2,
                                linestyle=(0, (3, 2)), zorder=2))
    ax.text(px, py - 0.27, "unused", ha="center", va="top", fontsize=7,
            color="#9aa3af", zorder=2)


_READOUT_RH = 0.30          # row height of a Pauli-readout box
_READOUT_GAP = 1.6          # gap between the tree and the readout strip


def _readout_height():
    return _READOUT_GAP + N_MODES * _READOUT_RH + 0.7  # boxes + γ labels


def _figsize_for(lo: Layout, show_paulis=False, scale=0.8):
    """Pick a figure size that matches the layout's aspect (the geometric tree
    is roughly square / taller-than-wide for a deep caterpillar)."""
    x0, x1, y0, y1 = _layout_extent(lo)
    extra = _readout_height() if show_paulis else 0.0
    w = (x1 - x0) * scale
    h = ((y1 + 0.9) - (y0 - extra)) * scale
    return (min(max(w, 8.5), 19.0), min(max(h, 6.0), 18.5))


def _draw_pauli_readout(ax, lo: Layout, strings, x0, x1):
    """Beneath the tree, an aligned strip of vertical Pauli-string boxes (one
    per Majorana, in left-to-right leaf order) with dotted connectors from each
    leaf — restoring the explicit operator readout of the strings view.

    Each box sits under its own leaf's x (so every connector is vertical and
    none can cross), nudged right only to avoid overlap.  The single omitted
    **parity** leaf is included as its own grey "unused" slot, so it is counted
    into the spacing and always has room.  A caterpillar's leaves + parity lie
    on a uniform grid, so the boxes come out evenly spaced and vertical.

    Returns (y_bottom, x_left, x_right) of the whole strip for the axis limits.
    """
    rh, box_w = _READOUT_RH, 0.62
    edge_of = {c[1]: L for p, c, L in lo.edges if c[0] == "leaf"}

    # slots = the 2N real leaves + the 1 parity leaf, ordered by x
    slots = [(g, xy[0], xy[1], "leaf") for g, xy in lo.leaf_xy.items()]
    if lo.parity is not None:
        (fx, fy), (px, py), _L = lo.parity
        slots.append((None, px, py, "parity"))
    slots.sort(key=lambda s: s[1])
    n = len(slots)

    y_box_top = min(s[2] for s in slots) - _READOUT_GAP
    y_box_bot = y_box_top - N_MODES * rh

    # Each box sits under its OWN slot's x, so every connector is vertical (and
    # vertical connectors can never cross); a slot is only nudged right when it
    # would otherwise overlap its left neighbour.  The parity is one of these
    # slots, so it is counted into the spacing.  For a caterpillar the leaves +
    # parity already lie on a uniform grid, so the boxes come out evenly spaced.
    min_gap = box_w + 0.14
    bxs, prev = [], None
    for _g, sx, _sy, _k in slots:
        bx = sx if prev is None else max(sx, prev + min_gap)
        bxs.append(bx)
        prev = bx

    for i, (g, sx, sy, kind) in enumerate(slots):
        bx = bxs[i]
        if kind == "leaf":
            col = LETTER_EDGE_COLOR.get(edge_of.get(g, "Z"), SPINE_Z)
            ax.plot([sx, bx], [sy - 0.06, y_box_top + 0.02], color=col,
                    linewidth=0.9, linestyle=(0, (1, 1.8)), alpha=0.5, zorder=1)
            _draw_vertical_leaf(ax, bx, y_box_top, rh, box_w, strings[g])
            ax.text(bx, y_box_bot - 0.16, f"γ{str(g).translate(_SUBS)}",
                    ha="center", va="top", fontsize=8.5, color=GAMMA_COL,
                    fontstyle="italic")
        else:  # the parity ("unused") slot — grey, empty, but spaced like the rest
            (fxx, fyy), (pxx, pyy), _Lk = lo.parity
            s0x, s0y, s1x, s1y = _trim(fxx, fyy, pxx, pyy, 0.30, 0.04)
            ax.plot([s0x, s1x], [s0y, s1y], color="#b8b8b8", linewidth=1.2,
                    linestyle=(0, (3, 2)), alpha=0.8, zorder=1)
            ax.add_patch(patches.Circle((pxx, pyy), 0.10, facecolor="none",
                                        edgecolor="#bcbcc4", linewidth=1.0, zorder=2))
            # vertical grey connector to the reserved box (box is under the
            # parity's own x, so it never crosses its neighbours)
            ax.plot([sx, bx], [sy - 0.04, y_box_top + 0.02], color="#c4c4cc",
                    linewidth=0.9, linestyle=(0, (1, 1.8)), alpha=0.55, zorder=1)
            ax.add_patch(patches.FancyBboxPatch(
                (bx - box_w / 2, y_box_bot), box_w, y_box_top - y_box_bot,
                boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="#f6f6f8",
                edgecolor="#ccccd4", linewidth=1.0, linestyle=(0, (3, 2)), zorder=3))
            ax.text(bx, (y_box_top + y_box_bot) / 2, "unused", ha="center",
                    va="center", rotation=90, fontsize=7.5, color="#9aa3af", zorder=4)

    # left qubit-index ruler aligned with the box rows
    rx = bxs[0] - 0.95
    ax.text(rx, y_box_top + 0.16, "qubit", ha="center", va="bottom",
            fontsize=8, color=MUTED_COL)
    for q in range(N_MODES):
        ax.text(rx, y_box_top - (q + 0.5) * rh, str(q), ha="center",
                va="center", fontsize=7.5, color=MUTED_COL)
    ax.text(rx, y_box_bot - 0.16, "Majorana →", ha="center", va="top",
            fontsize=7.5, color=MUTED_COL)
    return (y_box_bot - 0.55, rx - 0.3, bxs[-1] + box_w / 2 + 0.2)


def _draw_tree_view(ax, lo: Layout, strings, step, n_ops, title, frame_border,
                    show_paulis=True):
    ax.set_aspect("equal")
    ax.axis("off")
    f = _frame_from_layout(lo)
    _paint_tree_frame(ax, f["nodes"], f["leaves"], f["edges"], f["leaf_sign"],
                      leaf_labels=not show_paulis)

    x0, x1, y0, y1 = _layout_extent(lo)
    y_bottom = y0
    if show_paulis and strings is not None:
        # the readout reserves a column for the parity leaf (see below)
        y_bottom, bxmin, bxmax = _draw_pauli_readout(ax, lo, strings, x0, x1)
        x0, x1 = min(x0, bxmin - 0.3), max(x1, bxmax + 0.3)
    else:
        _draw_parity_leaf(ax, lo)   # bare tree: tree-level grey "unused" circle
    ax.set_xlim(x0, x1)
    ax.set_ylim(y_bottom, y1 + 0.9)

    if title is None:
        title = f"Majorana tree (after step {step}/{n_ops})"
    ax.text(x0 + 0.1, y1 + 0.78, title, ha="left", va="top",
            fontsize=13.5, fontweight="bold", color=TITLE_COL)
    sub = ("root→leaf spells each Majorana   ·   edge colour: "
           "X red, Y green, Z blue   ·   node = qubit, leaf = γ")
    if show_paulis:
        sub += "   ·   boxes below = each Majorana's current Pauli string"
    ax.text(x0 + 0.1, y1 + 0.30, sub, ha="left", va="top",
            fontsize=8.5, color=MUTED_COL)
    if frame_border:
        _frame_border(ax)


def _frame_border(ax):
    accent = "#d06a1f"
    border = patches.FancyBboxPatch(
        (0.006, 0.006), 0.988, 0.988,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        transform=ax.transAxes, fill=False, edgecolor=accent,
        linewidth=3.0, zorder=20)
    border.set_clip_on(False)
    ax.add_patch(border)
    ax.text(0.5, 0.992, "▶ ANIMATION FRAME  (run the cell to play)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=accent, zorder=21,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=accent, linewidth=1.0))


# ---------------------------------------------------------------------------
# Public: draw a single frame
# ---------------------------------------------------------------------------

def draw_majorana_tree(
    circuit=None,
    step: Optional[int] = None,
    figsize=None,
    title: Optional[str] = None,
    ax=None,
    frame_border: bool = False,
    mode: str = "tree",
    show_paulis: bool = True,
):
    """Draw the Majorana tree after the first `step` gates of `circuit`.

    Args:
        circuit: gate / (possibly nested) list of gates / QuantumTape / qfunc.
                 None means "no gates applied" (initial JW state).
        step:    apply only the first `step` ops.  None means all.
        figsize: matplotlib figure size (auto if None).
        title:   optional title.
        ax:      optional Axes (None creates a new figure).
        frame_border: draw an accent border + banner (used by the animator).
        mode:    "tree" (default) draws the reconstructed, morphing ternary
                 tree (shape + qubit labels + γ labels all free to move);
                 "strings" keeps the JW tree fixed and writes each Majorana's
                 Pauli string in its leaf.  "tree" auto-falls-back to
                 "strings" when the mapping is not a ternary tree.
        show_paulis: in tree mode, also draw the aligned strip of Pauli-string
                 readout boxes beneath the tree (with dotted connectors from
                 each leaf).  Set False for just the bare morphing tree.

    Returns (fig, ax) if a new figure was created, else ax.
    """
    ops = [] if circuit is None else _as_ops(circuit)
    if step is None:
        step = len(ops)
    step = max(0, min(step, len(ops)))

    note = None
    lo = None
    if mode == "tree":
        try:
            strings = majorana_strings_after(ops, step)
            lo = layout_tree(reconstruct_tree(strings))
        except NonTreeMapping:
            note = "⚠ not a ternary tree (a 2-qubit gate landed off a tree edge) — showing Pauli strings."
            mode = "strings"
        except ValueError as e:
            note = f"⚠ {e} — showing Pauli strings."
            mode = "strings"

    if mode != "tree":
        strings = majorana_strings_after(ops, step)

    new_fig = False
    if ax is None:
        if figsize is None:
            figsize = _figsize_for(lo, show_paulis) \
                if (mode == "tree" and lo is not None) else (17.0, 9.0)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        new_fig = True

    if mode == "tree":
        _draw_tree_view(ax, lo, strings, step, len(ops), title, frame_border,
                        show_paulis=show_paulis)
    else:
        _draw_strings_view(ax, strings, step, len(ops), title, frame_border, note)

    if new_fig:
        return fig, ax
    return ax


# ---------------------------------------------------------------------------
# Interactive ipywidgets scrubber
# ---------------------------------------------------------------------------

def _gate_caption(ops, step: int) -> str:
    n = len(ops)
    if step == 0:
        return "step 0 — initial JW state (no gates applied)"
    op = ops[step - 1]
    return f"step {step}/{n} — just applied {op.name}(wires={list(op.wires)})"


def interactive_majorana_tree(circuit, mode: str = "tree"):
    """Return an ipywidgets slider that scrubs through the circuit."""
    try:
        from ipywidgets import IntSlider, interactive
    except ImportError as e:
        raise RuntimeError(
            "ipywidgets is required for interactive_majorana_tree(); "
            "install it with `pip install ipywidgets`."
        ) from e

    ops = _as_ops(circuit)
    n = len(ops)

    def _render(step: int):
        fig, _ = draw_majorana_tree(ops, step=step,
                                    title=_gate_caption(ops, step), mode=mode)
        plt.show()

    slider = IntSlider(min=0, max=n, step=1, value=n,
                       description="step", continuous_update=False)
    return interactive(_render, step=slider)


def animate_majorana_tree(circuit, pause: float = 1.0, figsize=None,
                          mode: str = "tree"):
    """Scrub through `circuit` one gate at a time, redrawing IN PLACE.

    Plays live in a running kernel (clears the cell output between frames).
    A *saved* notebook keeps only the last frame — for a self-contained
    looping player that survives saving, use `play_majorana_tree`.
    """
    import time
    try:
        from IPython.display import clear_output, display
    except ImportError as e:
        raise RuntimeError(
            "animate_majorana_tree() needs IPython (run it in a notebook)."
        ) from e

    ops = _as_ops(circuit)
    n = len(ops)
    if n == 0:
        fig, _ = draw_majorana_tree(
            ops, step=0, figsize=figsize,
            title="(empty circuit — add some gates, then re-run)",
            frame_border=True, mode=mode)
        display(fig)
        plt.close(fig)
        return
    for k in range(n + 1):
        fig, _ = draw_majorana_tree(ops, step=k, figsize=figsize,
                                    title=_gate_caption(ops, k),
                                    frame_border=True, mode=mode)
        clear_output(wait=True)
        display(fig)
        plt.close(fig)
        if k < n:
            time.sleep(pause)


# ---------------------------------------------------------------------------
# Continuous tweened player (self-contained HTML; survives saving)
# ---------------------------------------------------------------------------

def play_majorana_tree(circuit, frames_per_gate: int = 10, hold: int = 4,
                       fps: int = 30, figsize=(13.0, 7.3), dpi: int = 72):
    """Return a continuously-tweened HTML animation of the morphing tree.

    Between consecutive gates the shared qubit-nodes and γ-leaves **glide** to
    their new positions while structural edges cross-fade, so a SWAP slides two
    qubit labels, a fermionic SWAP slides two γ labels, and a CZ/CNOT/CY plays
    as a tree rotation.  Embed the result in a notebook with::

        from IPython.display import HTML
        HTML(play_majorana_tree(my_circuit).to_html5_video())   # or just:
        play_majorana_tree(my_circuit)        # auto-displays via _repr_html_

    For a long circuit, lower `frames_per_gate`/`hold` (or install ffmpeg and
    use `.to_html5_video()`) to keep the embedded animation small.

    Requires a Jupyter front-end (uses `matplotlib.animation` → JS/HTML).
    """
    from matplotlib.animation import FuncAnimation

    ops = _as_ops(circuit)
    n = len(ops)

    # per-step layout (None where the mapping is not a tree / unsupported gate)
    layouts: List[Optional[Layout]] = []
    captions: List[str] = []
    for k in range(n + 1):
        captions.append(_gate_caption(ops, k))
        try:
            layouts.append(layout_tree(reconstruct_tree(
                majorana_strings_after(ops, k))))
        except (NonTreeMapping, ValueError):
            layouts.append(None)

    good = [lo for lo in layouts if lo is not None]
    if not good:
        raise RuntimeError("no step of this circuit is a ternary tree; use "
                           "draw_majorana_tree(..., mode='strings').")

    # fixed axes from the union of all layouts
    xs, ys = [], []
    for lo in good:
        x0, x1, y0, y1 = _layout_extent(lo)
        xs += [x0, x1]
        ys += [y0, y1]
    xlim = (min(xs), max(xs))
    ylim = (min(ys), max(ys) + 0.9)

    # build the flat frame schedule
    schedule = []  # each: (frame_dict, caption)
    for k in range(n + 1):
        lo = layouts[k]
        if lo is None:
            schedule += [(None, captions[k] + "  [not a ternary tree]")] * hold
        else:
            schedule += [(_frame_from_layout(lo), captions[k])] * hold
        if k < n and layouts[k] is not None and layouts[k + 1] is not None:
            A, B = layouts[k], layouts[k + 1]
            for s in range(1, frames_per_gate + 1):
                u = s / (frames_per_gate + 1)
                schedule.append((_frame_interp(A, B, u), captions[k + 1]))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    def update(i):
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        frame, caption = schedule[i]
        ax.text(xlim[0] + 0.1, ylim[1] + 0.05, caption, ha="left", va="top",
                fontsize=12.5, fontweight="bold", color=TITLE_COL)
        ax.text(xlim[0] + 0.1, ylim[1] - 0.45,
                "edge colour: X red · Y green · Z blue   (root→leaf spells the "
                "Majorana)", ha="left", va="top", fontsize=8.5, color=MUTED_COL)
        if frame is None:
            ax.text((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2,
                    "off-tree mapping\n(2-qubit gate off a tree edge)",
                    ha="center", va="center", fontsize=12, color="#b03030")
        else:
            _paint_tree_frame(ax, frame["nodes"], frame["leaves"],
                              frame["edges"], frame["leaf_sign"])

    anim = FuncAnimation(fig, update, frames=len(schedule),
                         interval=1000.0 / fps, blit=False)
    # make a bare `play_majorana_tree(circ)` auto-render in a notebook, and
    # raise the embed limit so typical permutation circuits don't drop frames
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams["animation.embed_limit"] = max(
        plt.rcParams.get("animation.embed_limit", 20.0), 96.0)
    plt.close(fig)
    return anim
