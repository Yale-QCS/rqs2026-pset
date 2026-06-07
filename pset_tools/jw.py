"""Jordan-Wigner geometry on the 3x3 snake.

This module defines the constants and helpers tied to the layout used by
every problem in this pset:

  * `N_MODES = 9` fermionic modes on a 3x3 grid in row-major (raster) order,
  * `snake_to_rc(j)` maps JW position j -> (row, col) on the qubit grid,
  * `initial_majorana(p)` returns the JW Pauli string for the p-th Majorana.

We treat snake JW position == qubit index throughout, so "qubit j" and
"JW position j" refer to the same wire.
"""

from __future__ import annotations

from typing import Tuple

from .pauli import PauliString

N_MODES = 9
GRID_L = 3


def snake_to_rc(j: int) -> Tuple[int, int]:
    """Map JW position j ∈ {0, ..., 8} to (row, col) on the 3x3 snake grid.

      0  1  2
      5  4  3
      6  7  8
    """
    r = j // GRID_L
    pos = j % GRID_L
    c = pos if r % 2 == 0 else GRID_L - 1 - pos
    return r, c


def rc_to_snake(r: int, c: int) -> int:
    """Inverse of `snake_to_rc`."""
    return r * GRID_L + (c if r % 2 == 0 else GRID_L - 1 - c)


def is_grid_adjacent(a: int, b: int) -> bool:
    """True iff qubits a, b are physically adjacent on the 3x3 snake grid."""
    ra, ca = snake_to_rc(a)
    rb, cb = snake_to_rc(b)
    return abs(ra - rb) + abs(ca - cb) == 1


def initial_majorana(p: int) -> PauliString:
    """Return the JW Pauli string for the p-th Majorana operator gamma_p.

    Convention (matches the problem set):
        gamma_{2j}   = Z_0 Z_1 ... Z_{j-1} X_j
        gamma_{2j+1} = Z_0 Z_1 ... Z_{j-1} Y_j
    """
    j = p // 2
    alpha = p % 2
    letters = {k: "Z" for k in range(j)}
    letters[j] = "X" if alpha == 0 else "Y"
    return PauliString(phase=1+0j, letters=letters)


def all_initial_majoranas():
    """List of 2 * N_MODES initial Majorana Pauli strings."""
    return [initial_majorana(p) for p in range(2 * N_MODES)]
