"""Tools provided to students for the fermionic-permutation problem set.

Public API:

    verify_fp(circuit, target_perm)        -- Problems 1-3 checker
    verify_trotter_step(circuit, theta=…)  -- Problem 4-5 checker
    depth_2q(circuit, connectivity=None)   -- native+topology check, then 2q depth
                                              ("2dnn" requires grid-adjacency).
                                              Accepts any gate that decomposes to
                                              native gates (SWAP, qml.PauliRot, …).

    (You build the fermionic swap yourself in Problem 1 — it is not provided.)

    draw_majorana_tree(circuit, step=…)    -- the Majorana-tree visualizer
    interactive_majorana_tree(circuit)     -- ipywidgets slider scrubber

    HORIZONTAL_EDGES, VERTICAL_EDGES, HOPPING_EDGES
    hopping_pauli_strings(i, j)

    snake_layout_figure()                  -- yellow snake JW figure
    fermion_lattice_figure()               -- green fermion mode lattice figure
"""

from .checker import (
    HOPPING_EDGES,
    HORIZONTAL_EDGES,
    VERTICAL_EDGES,
    hopping_pauli_strings,
    reference_trotter_step_ops,
    verify_fp,
    verify_trotter_step,
)
from .depth import depth_2q
from .style import apply_style  # applies the shared figure theme on import
from .figures import (
    fermion_lattice_figure,
    permutation_on_snake_figure,
    snake_and_lattice_figure,
    snake_layout_figure,
)
from .jw import N_MODES, initial_majorana, is_grid_adjacent, rc_to_snake, snake_to_rc
from .pauli import PauliString, conjugate, conjugate_through
from .tree import (
    Layout,
    Leaf,
    Node,
    NonTreeMapping,
    layout_tree,
    reconstruct_tree,
    tree_to_text,
)
from .visualizer import (
    animate_majorana_tree,
    draw_majorana_tree,
    interactive_majorana_tree,
    majorana_strings_after,
    play_majorana_tree,
    reconstruct_after,
)

__all__ = [
    "verify_fp",
    "verify_trotter_step",
    "depth_2q",
    "is_grid_adjacent",
    "draw_majorana_tree",
    "interactive_majorana_tree",
    "animate_majorana_tree",
    "play_majorana_tree",
    "majorana_strings_after",
    "reconstruct_after",
    "reconstruct_tree",
    "layout_tree",
    "tree_to_text",
    "NonTreeMapping",
    "Node",
    "Leaf",
    "Layout",
    "snake_layout_figure",
    "fermion_lattice_figure",
    "snake_and_lattice_figure",
    "permutation_on_snake_figure",
    "apply_style",
    "PauliString",
    "conjugate",
    "conjugate_through",
    "initial_majorana",
    "snake_to_rc",
    "rc_to_snake",
    "N_MODES",
    "HORIZONTAL_EDGES",
    "VERTICAL_EDGES",
    "HOPPING_EDGES",
    "hopping_pauli_strings",
    "reference_trotter_step_ops",
]
