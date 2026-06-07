"""Checkers for the pset.

`verify_fp(circuit, target_perm)` — Boolean checker for Problems 1-3.

`verify_trotter_step(circuit, theta)` — unitary-equality checker for
Problems 4-5, using a small explicit reference circuit built from the
twelve hopping terms in a fixed order.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import pennylane as qml

from ._ops import flatten_ops
from .jw import N_MODES, all_initial_majoranas, initial_majorana
from .pauli import PauliString, conjugate_through

PermSpec = Union[Sequence[int], Tuple[int, int], Sequence[Tuple[int, int]]]


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _as_perm(target: PermSpec, n: int = N_MODES) -> List[int]:
    """Normalize `target` to a full permutation list of length n.

    Accepts:
      * a length-2 tuple (i, j)  -> single transposition
      * a list of length-2 tuples [(i1,j1), (i2,j2), ...]  -> product of
        disjoint transpositions
      * a length-n permutation list [pi(0), pi(1), ..., pi(n-1)]
    """
    if isinstance(target, tuple) and len(target) == 2 \
            and all(isinstance(x, (int, np.integer)) for x in target):
        i, j = target
        perm = list(range(n))
        perm[i], perm[j] = perm[j], perm[i]
        return perm
    if isinstance(target, (list, tuple)) and len(target) > 0 \
            and isinstance(target[0], tuple):
        perm = list(range(n))
        for i, j in target:
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    if isinstance(target, (list, tuple)) and len(target) == n:
        perm = list(target)
        if sorted(perm) != list(range(n)):
            raise ValueError(f"Not a permutation of 0..{n-1}: {target}")
        return perm
    raise ValueError(
        f"Could not interpret target_perm={target!r} for n={n}. "
        "Pass a (i, j) tuple, a list of (i, j) tuples, or a full "
        "permutation list of length n."
    )


# ---------------------------------------------------------------------------
# Circuit normalization: accept tape / list-of-ops / callable
# ---------------------------------------------------------------------------

def _as_ops(circuit) -> List[qml.operation.Operator]:
    """Normalize the input to a flat list of PennyLane operations.

    Lists may be nested (group sub-circuits as gadgets); see ``_ops.py``.
    """
    return flatten_ops(circuit)


# ---------------------------------------------------------------------------
# verify_fp
# ---------------------------------------------------------------------------

def verify_fp(circuit, target_perm: PermSpec, verbose: bool = False) -> bool:
    """Return True iff `circuit` realizes the fermionic permutation `target_perm`.

    The check is: for every j ∈ {0,...,N-1} and α ∈ {0,1},

        U @ gamma_{2j+α} @ U.adjoint()  ==  gamma_{2 pi(j) + α}.

    `circuit` may be a list of PennyLane operations, a QuantumTape, or a
    no-argument function that builds one in its enclosing tape context.
    """
    ops = _as_ops(circuit)
    perm = _as_perm(target_perm)

    ok = True
    for j in range(N_MODES):
        for alpha in (0, 1):
            p = 2 * j + alpha
            target = initial_majorana(2 * perm[j] + alpha)
            actual = conjugate_through(initial_majorana(p), ops)
            if not actual.equals(target):
                ok = False
                if verbose:
                    print(
                        f"  γ_{p}: expected {target.to_label()}, "
                        f"got {actual.to_label()}"
                    )
    return ok


# ---------------------------------------------------------------------------
# Trotter-step reference (Problems 4 and 5)
# ---------------------------------------------------------------------------

# Mode lattice = the 3x3 grid laid out in SNAKE (JW) order, so mode i sits
# at JW position i:
#       0  1  2          (row 0, left to right)
#       5  4  3          (row 1, right to left)
#       6  7  8          (row 2, left to right)
# Horizontal bonds run along the snake, so their two modes are JW-adjacent
# (cheap).  Vertical bonds jump between rows where the snake has folded, so
# their two modes are JW-far (expensive) -- JW distances 1, 3, or 5.
HORIZONTAL_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
]
VERTICAL_EDGES: List[Tuple[int, int]] = [
    (0, 5), (5, 6), (1, 4), (4, 7), (2, 3), (3, 8),
]
HOPPING_EDGES: List[Tuple[int, int]] = HORIZONTAL_EDGES + VERTICAL_EDGES


def hopping_pauli_strings(i: int, j: int) -> Tuple[PauliString, PauliString]:
    """Return the two commuting Pauli operators that sum to (a_i† a_j + h.c.)
    under Jordan-Wigner.

    For JW positions p = min(i, j), q = max(i, j),

        a_i† a_j + a_j† a_i
            = (1/2) (X_p X_q + Y_p Y_q) * Z_{p+1} ... Z_{q-1}.

    We drop the global 1/2 by absorbing it into the rotation angle in the
    Trotter step: exp(i θ (X X + Y Y) Z…Z / 2)  = ∏ exp(i θ/2 ...).
    """
    p, q = (i, j) if i < j else (j, i)
    z_letters = {k: "Z" for k in range(p + 1, q)}
    xx_letters = {p: "X", q: "X", **z_letters}
    yy_letters = {p: "Y", q: "Y", **z_letters}
    return (
        PauliString(phase=1+0j, letters=xx_letters),
        PauliString(phase=1+0j, letters=yy_letters),
    )


def _pauli_rot_ops(theta: float, ps: PauliString) -> List[qml.operation.Operator]:
    """Return a (small) PennyLane circuit implementing exp(i theta ps).

    We use qml.PauliRot, which is a built-in multi-qubit Pauli rotation,
    so we don't have to worry about ordering of CNOT staircases here.
    """
    qubits = sorted(ps.letters.keys())
    label = "".join(ps.letters[q] for q in qubits)
    # qml.PauliRot rotates by exp(-i theta/2 P), so to get exp(+i theta P)
    # we pass an angle of -2*theta.
    return [qml.PauliRot(-2.0 * theta, label, wires=qubits)]


def reference_trotter_step_ops(theta: float) -> List[qml.operation.Operator]:
    """Build the reference U(θ) circuit for one Trotter step of the 3x3
    spinless Fermi-Hubbard hopping Hamiltonian, in the fixed factor order
    (the 6 horizontal edges then the 6 vertical edges, in the order listed
    in HORIZONTAL_EDGES / VERTICAL_EDGES)."""
    ops: List[qml.operation.Operator] = []
    for (i, j) in HOPPING_EDGES:
        xx, yy = hopping_pauli_strings(i, j)
        # exp(i θ (a_i† a_j + h.c.)) = exp(i θ/2 (XX + YY) Z…Z)
        # = exp(i θ/2 XX Z…Z) * exp(i θ/2 YY Z…Z)   [they commute]
        ops += _pauli_rot_ops(theta / 2.0, xx)
        ops += _pauli_rot_ops(theta / 2.0, yy)
    return ops


def _circuit_matrix(ops: List[qml.operation.Operator], n_wires: int) -> np.ndarray:
    """Compute the unitary matrix of `ops` on the given number of wires."""
    wire_order = list(range(n_wires))
    if not ops:
        return np.eye(2 ** n_wires, dtype=complex)
    # Use qml.matrix on a tape so we don't run a device.
    with qml.tape.QuantumTape() as tape:
        for op in ops:
            qml.apply(op)
    return qml.matrix(tape, wire_order=wire_order)


def verify_trotter_step(
    circuit,
    theta: float = 0.137,
    atol: float = 1e-7,
) -> bool:
    """Return True iff `circuit` implements one Trotter step at angle `theta`.

    The comparison is exact: both the student circuit and the reference
    circuit are expanded to full unitaries on 9 qubits and compared up to
    global phase.  Default `theta` is a generic incommensurate angle so
    that a wrong circuit cannot accidentally pass.
    """
    student_ops = _as_ops(circuit)
    ref_ops = reference_trotter_step_ops(theta)
    U_student = _circuit_matrix(student_ops, N_MODES)
    U_ref = _circuit_matrix(ref_ops, N_MODES)
    # Compare up to a global phase: <U_ref, U_student> should be a scalar.
    overlap = np.vdot(U_ref.flatten(), U_student.flatten()) / (2 ** N_MODES)
    return abs(abs(overlap) - 1.0) < atol
