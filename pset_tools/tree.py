"""Canonical ternary-tree reconstruction from a set of Majorana Pauli strings.

A fermion-to-qubit mapping is a *decorated ternary tree*: a full ordered
ternary tree whose internal nodes carry qubit labels and whose leaves carry
Majorana labels γ_p (read off root→leaf as left=X, middle=Y, right=Z).

The key fact we rely on for *correct* animation is **uniqueness**: for a fixed
mapping (a fixed set of 2N anticommuting Pauli strings) the decorated tree is
unique.  The root is the *only* qubit that appears in all 2N strings (the sole
common ancestor of every leaf), so it is forced; partition the strings by that
qubit's Pauli letter into X/Y/Z groups and recurse.

Because we rebuild the tree from the *operators themselves*, the picture is
correct no matter how a gate was decomposed into native gates (e.g. whether a
CZ is applied directly or as H·CNOT·H): only the endpoint operators matter, and
they pin down one tree.

If the operator set is not tree-representable (e.g. a CZ/CNOT applied between
two qubits that are *not* parent-child on the current tree), there is no common
root and we raise `NonTreeMapping`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .pauli import PauliString

EDGE_LETTERS = ("X", "Y", "Z")  # left, middle, right


class NonTreeMapping(Exception):
    """Raised when a set of Majorana strings is not a ternary-tree mapping.

    Carries the offending strings so a caller can fall back to a different
    rendering (e.g. drawing the raw Pauli strings).
    """

    def __init__(self, message: str, strings: Optional[Sequence[PauliString]] = None):
        super().__init__(message)
        self.strings = strings


@dataclass
class Leaf:
    gamma: int            # Majorana index γ_p
    sign: complex = 1+0j  # ±1 gauge (or ±i in intermediate computations)


@dataclass
class Node:
    qubit: int
    # children keyed by edge letter "X"/"Y"/"Z"; None = the omitted (parity) leaf
    children: Dict[str, Optional[Union["Node", "Leaf"]]] = field(default_factory=dict)


TreeT = Union[Node, Leaf]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def _sign(ps: PauliString) -> complex:
    return ps.phase


def _build(work: List[Tuple[int, complex, Dict[int, str]]]) -> TreeT:
    """work = list of (gamma, sign, {qubit: letter}).  Returns Node/Leaf."""
    if len(work) == 1:
        g, s, d = work[0]
        if not d:
            return Leaf(gamma=g, sign=s)
        raise NonTreeMapping(
            f"γ{g} would have to continue past its leaf (remaining support "
            f"{sorted(d)}): not a full ternary tree."
        )

    # Root candidates: qubits present (non-trivially) in *every* string.
    common = set.intersection(*[set(d.keys()) for _, _, d in work])
    if not common:
        raise NonTreeMapping(
            f"no qubit is shared by all {len(work)} Majorana strings — the "
            f"mapping is not a ternary tree (a 2-qubit gate was applied off a "
            f"tree edge)."
        )

    last_err: Optional[NonTreeMapping] = None
    for root in sorted(common):
        groups: Dict[str, List[Tuple[int, complex, Dict[int, str]]]] = {
            "X": [], "Y": [], "Z": []
        }
        for g, s, d in work:
            L = d[root]
            d2 = {q: v for q, v in d.items() if q != root}
            groups[L].append((g, s, d2))
        try:
            children: Dict[str, Optional[TreeT]] = {}
            for L in EDGE_LETTERS:
                children[L] = _build(groups[L]) if groups[L] else None
            return Node(qubit=root, children=children)
        except NonTreeMapping as e:
            last_err = e
            continue
    raise last_err or NonTreeMapping("could not reconstruct a ternary tree.")


def reconstruct_tree(strings: Sequence[PauliString]) -> Node:
    """Rebuild the unique decorated ternary tree from Majorana strings.

    `strings[p]` is the current Pauli operator for Majorana γ_p.  Returns the
    root `Node`.  Raises `NonTreeMapping` if the set is not tree-representable.
    """
    work = [(p, _sign(ps), dict(ps.letters)) for p, ps in enumerate(strings)]
    tree = _build(work)
    if isinstance(tree, Leaf):  # 1-mode degenerate case
        tree = Node(qubit=0, children={"X": tree, "Y": None, "Z": None})
    return tree


# ---------------------------------------------------------------------------
# Traversals / debugging
# ---------------------------------------------------------------------------

def inorder_leaves(tree: TreeT) -> List[Leaf]:
    out: List[Leaf] = []

    def rec(n: TreeT):
        if isinstance(n, Leaf):
            out.append(n)
            return
        for L in EDGE_LETTERS:
            c = n.children.get(L)
            if c is not None:
                rec(c)

    rec(tree)
    return out


def tree_to_text(tree: TreeT, indent: str = "") -> str:
    if isinstance(tree, Leaf):
        sg = "+" if abs(tree.sign - 1) < 1e-9 else (
            "-" if abs(tree.sign + 1) < 1e-9 else f"{tree.sign:+.2g}")
        return f"{indent}→ γ{tree.gamma} ({sg})"
    lines = [f"{indent}q{tree.qubit}"]
    for L in EDGE_LETTERS:
        c = tree.children.get(L)
        if c is None:
            lines.append(f"{indent}  [{L}] ·(parity)")
        elif isinstance(c, Leaf):
            lines.append(tree_to_text(c, indent + f"  [{L}] ").replace("→", "→", 1))
        else:
            lines.append(f"{indent}  [{L}]")
            lines.append(tree_to_text(c, indent + "      "))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout: assign (x, y) to every node and leaf, keyed by *identity*
# (qubit index for nodes, γ index for leaves) so an animator can track the
# same object across frames and interpolate its position.
# ---------------------------------------------------------------------------

@dataclass
class Layout:
    node_xy: Dict[int, Tuple[float, float]]       # qubit -> (x, y)
    leaf_xy: Dict[int, Tuple[float, float]]       # gamma -> (x, y)
    leaf_sign: Dict[int, complex]                 # gamma -> ±1
    # edges as (parent_ref, child_ref, letter) where ref = ("node", q)/("leaf", p)
    edges: List[Tuple[Tuple[str, int], Tuple[str, int], str]]
    y_leaf: float
    max_depth: int
    # the single omitted (parity) leaf: (parent_xy, leaf_xy, letter) or None
    parity: Optional[Tuple[Tuple[float, float], Tuple[float, float], str]] = None

    def positions(self) -> Dict[Tuple[str, int], Tuple[float, float]]:
        pos: Dict[Tuple[str, int], Tuple[float, float]] = {}
        for q, xy in self.node_xy.items():
            pos[("node", q)] = xy
        for p, xy in self.leaf_xy.items():
            pos[("leaf", p)] = xy
        return pos


def layout_tree(tree: Node, leaf_gap: float = 1.0,
                spread_deg: float = 48.0) -> Layout:
    """Fixed-angle ternary-tree layout with leaves on a uniform x-grid.

    Every leaf (and the single omitted *parity* slot) is placed at a uniform
    in-order x-position.  A node sits directly above its **middle (Y)**
    descendant, and the **left (X)** / **right (Z)** edges leave every node at
    the *same* symmetric angles ``∓spread_deg`` from vertical — only their
    *lengths* vary, chosen so each child lands exactly on the grid.  So:

      * the drawing keeps the canonical symmetric-angle look (constant-slope
        X/Z edges, vertical Y, a long constant-slope Z-spine for JW); yet
      * every leaf is evenly spaced on a grid, so a Pauli box placed directly
        under it hangs from a perfectly vertical, evenly-spaced, non-overlapping
        connector — for *every* tree shape; and
      * it is planar (in-order leaves ⇒ contiguous subtree bands ⇒ no crossings).

    For a caterpillar this reproduces ``spine_len == 2·leaf_len`` exactly.
    Positions are keyed by identity (qubit/γ) so an animator can glide objects.
    """
    import math
    cot = math.cos(math.radians(spread_deg)) / math.sin(math.radians(spread_deg))
    drop_y = leaf_gap * cot          # vertical drop of a one-grid-unit X/Z edge
    Ly = drop_y                       # Y edge (vertical) drops the same amount

    # --- pass 1: uniform in-order x for every leaf and the parity slot.
    leaf_x: Dict[int, float] = {}
    counter = [0]
    parity_ref: List = [None]   # (parent_node, letter, x)

    def inorder(node: TreeT):
        if isinstance(node, Leaf):
            leaf_x[node.gamma] = counter[0] * leaf_gap
            counter[0] += 1
            return
        for L in EDGE_LETTERS:
            c = node.children.get(L)
            if c is None:
                if parity_ref[0] is None:
                    parity_ref[0] = (node, L, counter[0] * leaf_gap)
                    counter[0] += 1
            else:
                inorder(c)

    inorder(tree)

    # --- pass 2: a node's x is its Y-descendant's x (so the Y edge is vertical).
    node_gx: Dict[int, float] = {}

    def comp_x(node: Node) -> float:
        yc = node.children.get("Y")
        if yc is None:                       # this node's Y slot is the parity
            nx = parity_ref[0][2]
        elif isinstance(yc, Leaf):
            nx = leaf_x[yc.gamma]
        else:
            nx = comp_x(yc)
        node_gx[node.qubit] = nx
        for L in ("X", "Z"):
            c = node.children.get(L)
            if isinstance(c, Node):
                comp_x(c)
        return nx

    comp_x(tree)

    # --- pass 3: place; X/Z drop = |Δx|·cot(spread) keeps the angle fixed.
    node_xy: Dict[int, Tuple[float, float]] = {}
    leaf_xy: Dict[int, Tuple[float, float]] = {}
    leaf_sign: Dict[int, complex] = {}
    edges: List[Tuple[Tuple[str, int], Tuple[str, int], str]] = []
    depth_seen = [0]
    parity_box: List = [None]

    def place(node: Node, ny: float, level: int):
        nx = node_gx[node.qubit]
        node_xy[node.qubit] = (nx, ny)
        depth_seen[0] = max(depth_seen[0], level)
        for L in EDGE_LETTERS:
            c = node.children.get(L)
            if c is None:
                if parity_ref[0] and parity_ref[0][0] is node and parity_ref[0][1] == L:
                    px = parity_ref[0][2]
                    py = ny - Ly if L == "Y" else ny - abs(nx - px) * cot
                    parity_box[0] = ((nx, ny), (px, py), L)
                continue
            cx = leaf_x[c.gamma] if isinstance(c, Leaf) else node_gx[c.qubit]
            cy = ny - Ly if L == "Y" else ny - abs(nx - cx) * cot
            child_ref = ("leaf", c.gamma) if isinstance(c, Leaf) else ("node", c.qubit)
            edges.append((("node", node.qubit), child_ref, L))
            if isinstance(c, Leaf):
                leaf_xy[c.gamma] = (cx, cy)
                leaf_sign[c.gamma] = c.sign
            else:
                place(c, cy, level + 1)

    place(tree, 0.0, 0)

    # A fixed angle turns horizontal span into vertical drop, so a wide/balanced
    # tree can get very tall.  Caterpillar-like trees (the usual case) are well
    # within bounds; only rarely do we uniformly compress y to cap the aspect
    # ratio — which just makes the (still uniform, still symmetric) X/Z angle a
    # touch flatter for that one tree.  Leaves keep their x, so the readout stays
    # vertical / even.
    xs = [x for x, _ in node_xy.values()] + [x for x, _ in leaf_xy.values()]
    ys = [y for _, y in node_xy.values()] + [y for _, y in leaf_xy.values()]
    width = (max(xs) - min(xs)) if xs else 1.0
    height = -min(ys) if ys else 0.0
    max_aspect = 1.3
    if height > max_aspect * max(width, 1.0):
        f = max_aspect * max(width, 1.0) / height
        node_xy = {q: (x, y * f) for q, (x, y) in node_xy.items()}
        leaf_xy = {p: (x, y * f) for p, (x, y) in leaf_xy.items()}
        if parity_box[0] is not None:
            (fx, fy), (px, py), L = parity_box[0]
            parity_box[0] = ((fx, fy * f), (px, py * f), L)
        ys = [y * f for y in ys]

    return Layout(node_xy=node_xy, leaf_xy=leaf_xy, leaf_sign=leaf_sign,
                  edges=edges, y_leaf=min(ys) if ys else 0.0,
                  max_depth=depth_seen[0], parity=parity_box[0])
