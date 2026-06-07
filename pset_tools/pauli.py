"""Symbolic Pauli-string tracking under Clifford conjugation.

A `PauliString` is a sparse mapping `{qubit -> "X"/"Y"/"Z"}` together with a
complex `phase` (typically +/-1 for Hermitian Pauli strings, but we allow
+/- i during intermediate computations).

The `conjugate(ps, op)` helper computes  ``U @ ps @ U.adjoint()``  for a
Clifford `op` (one of qml.SWAP, qml.CZ, qml.CNOT, qml.CY, qml.Hadamard,
qml.S, qml.PauliX/Y/Z) and returns a new PauliString.

This module is deliberately small and self-contained so students can read
it.  It is the engine behind `verify_fp` and the Majorana-tree visualizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import pennylane as qml

PauliLetter = str  # one of "X", "Y", "Z" — "I" is represented by absence

# ---------------------------------------------------------------------------
# Single-qubit Pauli multiplication table:  L1 * L2 == phase * L_out
# ---------------------------------------------------------------------------

_PAULI_MUL: Dict[Tuple[PauliLetter, PauliLetter], Tuple[PauliLetter, complex]] = {
    ("I", "I"): ("I",  1+0j), ("I", "X"): ("X",  1+0j),
    ("I", "Y"): ("Y",  1+0j), ("I", "Z"): ("Z",  1+0j),
    ("X", "I"): ("X",  1+0j), ("X", "X"): ("I",  1+0j),
    ("X", "Y"): ("Z",  1j),   ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y",  1+0j), ("Y", "X"): ("Z", -1j),
    ("Y", "Y"): ("I",  1+0j), ("Y", "Z"): ("X",  1j),
    ("Z", "I"): ("Z",  1+0j), ("Z", "X"): ("Y",  1j),
    ("Z", "Y"): ("X", -1j),   ("Z", "Z"): ("I",  1+0j),
}


def _mul(a: PauliLetter, b: PauliLetter) -> Tuple[PauliLetter, complex]:
    return _PAULI_MUL[(a, b)]


# ---------------------------------------------------------------------------
# PauliString class
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class PauliString:
    """A signed Pauli tensor product on a finite number of qubits.

    `letters[q]` ∈ {"X", "Y", "Z"} for the qubits the string acts on
    non-trivially; identity factors are implicit (missing keys).
    """
    phase: complex = 1+0j
    letters: Dict[int, PauliLetter] = field(default_factory=dict)

    # --- constructors ------------------------------------------------------

    @classmethod
    def identity(cls) -> "PauliString":
        return cls(phase=1+0j, letters={})

    @classmethod
    def from_string(cls, spec: str, phase: complex = 1) -> "PauliString":
        """Parse a string like 'Z0 Z1 Z2 X3' into a PauliString."""
        letters: Dict[int, str] = {}
        for tok in spec.split():
            L, q = tok[0], int(tok[1:])
            if L != "I":
                letters[q] = L
        return cls(phase=phase, letters=letters)

    # --- inspection --------------------------------------------------------

    def letter_at(self, q: int) -> PauliLetter:
        return self.letters.get(q, "I")

    def weight(self) -> int:
        return len(self.letters)

    def support(self) -> Iterable[int]:
        return self.letters.keys()

    # --- equality (signed) -------------------------------------------------

    def equals(self, other: "PauliString", atol: float = 1e-9) -> bool:
        if self.letters != other.letters:
            return False
        return abs(self.phase - other.phase) < atol

    # --- pretty printing ---------------------------------------------------

    def to_label(
        self,
        n_qubits: Optional[int] = None,
        style: str = "subscript",
        identity_char: str = "·",
    ) -> str:
        """Render to a short label.

        style="subscript": "Z₀Z₁Z₂X₃" (skips identities, subscript indices).
        style="dense":     "ZZZX·····"  (one char per qubit, identity_char for I).
        """
        if style == "dense":
            assert n_qubits is not None, "dense style requires n_qubits"
            return "".join(self.letters.get(q, identity_char) for q in range(n_qubits))
        # subscript
        subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        parts = [f"{L}{str(q).translate(subs)}"
                 for q, L in sorted(self.letters.items())]
        s = "".join(parts) if parts else "I"
        if abs(self.phase - 1) < 1e-9:
            return s
        if abs(self.phase + 1) < 1e-9:
            return "-" + s
        if abs(self.phase - 1j) < 1e-9:
            return "i" + s
        if abs(self.phase + 1j) < 1e-9:
            return "-i" + s
        return f"({self.phase})" + s

    def __repr__(self) -> str:
        return f"PauliString({self.to_label()})"


# ---------------------------------------------------------------------------
# Multiplying single-qubit factors into a PauliString
# ---------------------------------------------------------------------------

def _apply_letter(ps_letters: Dict[int, str], q: int, new_factor: str) -> complex:
    """Multiply `new_factor` into the existing letter at qubit `q`.

    Returns the phase picked up (1, ±i, or -1).  Convention: we apply
    `new_factor` AFTER the existing letter, i.e. the result is
    ``letters[q] * new_factor``.
    """
    old = ps_letters.get(q, "I")
    res, phase = _mul(old, new_factor)
    if res == "I":
        ps_letters.pop(q, None)
    else:
        ps_letters[q] = res
    return phase


# ---------------------------------------------------------------------------
# Single-qubit Clifford gates: U L U^†  (action on the Pauli letter at q)
# ---------------------------------------------------------------------------

_SINGLE_Q_RULES = {
    "Hadamard": {"X": ("Z", 1+0j), "Y": ("Y", -1+0j), "Z": ("X", 1+0j)},
    "S":        {"X": ("Y", 1+0j), "Y": ("X", -1+0j), "Z": ("Z", 1+0j)},
    "PauliX":   {"X": ("X", 1+0j), "Y": ("Y", -1+0j), "Z": ("Z", -1+0j)},
    "PauliY":   {"X": ("X", -1+0j),"Y": ("Y", 1+0j),  "Z": ("Z", -1+0j)},
    "PauliZ":   {"X": ("X", -1+0j),"Y": ("Y", -1+0j), "Z": ("Z", 1+0j)},
}


def _conjugate_single_qubit(ps: PauliString, gate_name: str, q: int) -> PauliString:
    L = ps.letter_at(q)
    new_letters = dict(ps.letters)
    new_phase = ps.phase
    if L != "I":
        new_L, phase = _SINGLE_Q_RULES[gate_name][L]
        new_phase *= phase
        if new_L == "I":
            new_letters.pop(q, None)
        else:
            new_letters[q] = new_L
    return PauliString(new_phase, new_letters)


# ---------------------------------------------------------------------------
# Two-qubit Clifford gates: precomputed tables for (L_a, L_b) -> (L_a', L_b', phase)
#
# We derive these by:
#   1. computing U L_a U^† and U L_b U^† individually (each yields a 2-qubit Pauli),
#   2. multiplying the two 2-qubit Paulis to get the conjugate of L_a L_b.
# ---------------------------------------------------------------------------

# CNOT(c, t) action on single Paulis (control=c, target=t):
#   X_c -> X_c X_t,  Y_c -> Y_c X_t,  Z_c -> Z_c
#   X_t -> X_t,      Y_t -> Z_c Y_t,  Z_t -> Z_c Z_t
#
# CZ(a, b) action:
#   X_a -> X_a Z_b,  Y_a -> Y_a Z_b,  Z_a -> Z_a
#   X_b -> Z_a X_b,  Y_b -> Z_a Y_b,  Z_b -> Z_b
#
# CY(c, t) action (control=c, target=t);  CY = (I⊗S) CNOT (I⊗S†):
#   X_c -> X_c Y_t,  Y_c -> Y_c Y_t,  Z_c -> Z_c
#   X_t -> Z_c X_t,  Y_t -> Y_t,      Z_t -> Z_c Z_t
#
# SWAP(a, b) action: swap the letters at a and b.


def _conjugate_cnot(ps: PauliString, c: int, t: int) -> PauliString:
    la = ps.letter_at(c)
    lt = ps.letter_at(t)
    new_letters = dict(ps.letters)
    new_phase = ps.phase
    # First strip old factors at c, t:
    new_letters.pop(c, None)
    new_letters.pop(t, None)

    # Effect of CNOT on L_a alone (acting at c):
    # L_a = I: contributes I_c I_t.
    # L_a = X: contributes X_c X_t.
    # L_a = Y: contributes Y_c X_t.
    # L_a = Z: contributes Z_c I_t.
    a_effect_c = {"I":"I", "X":"X", "Y":"Y", "Z":"Z"}[la]
    a_effect_t = {"I":"I", "X":"X", "Y":"X", "Z":"I"}[la]

    # Effect of CNOT on L_t alone (acting at t):
    # L_t = I: I_c I_t.
    # L_t = X: I_c X_t.
    # L_t = Y: Z_c Y_t.
    # L_t = Z: Z_c Z_t.
    t_effect_c = {"I":"I", "X":"I", "Y":"Z", "Z":"Z"}[lt]
    t_effect_t = {"I":"I", "X":"X", "Y":"Y", "Z":"Z"}[lt]

    # Multiply the two contributions (on each qubit separately):
    new_c, ph_c = _mul(a_effect_c, t_effect_c)
    new_t, ph_t = _mul(a_effect_t, t_effect_t)
    new_phase *= ph_c * ph_t
    if new_c != "I":
        new_letters[c] = new_c
    if new_t != "I":
        new_letters[t] = new_t
    return PauliString(new_phase, new_letters)


def _conjugate_cz(ps: PauliString, a: int, b: int) -> PauliString:
    la = ps.letter_at(a)
    lb = ps.letter_at(b)
    new_letters = dict(ps.letters)
    new_phase = ps.phase
    new_letters.pop(a, None)
    new_letters.pop(b, None)

    # Effect of CZ on L_a alone:
    # I -> I_a I_b, X -> X_a Z_b, Y -> Y_a Z_b, Z -> Z_a I_b.
    a_effect_a = {"I":"I", "X":"X", "Y":"Y", "Z":"Z"}[la]
    a_effect_b = {"I":"I", "X":"Z", "Y":"Z", "Z":"I"}[la]

    # Effect of CZ on L_b alone (symmetric):
    b_effect_a = {"I":"I", "X":"Z", "Y":"Z", "Z":"I"}[lb]
    b_effect_b = {"I":"I", "X":"X", "Y":"Y", "Z":"Z"}[lb]

    new_a, ph_a = _mul(a_effect_a, b_effect_a)
    new_b, ph_b = _mul(a_effect_b, b_effect_b)
    new_phase *= ph_a * ph_b
    if new_a != "I":
        new_letters[a] = new_a
    if new_b != "I":
        new_letters[b] = new_b
    return PauliString(new_phase, new_letters)


def _conjugate_cy(ps: PauliString, c: int, t: int) -> PauliString:
    la = ps.letter_at(c)
    lt = ps.letter_at(t)
    new_letters = dict(ps.letters)
    new_phase = ps.phase
    new_letters.pop(c, None)
    new_letters.pop(t, None)

    # Effect of CY on the control letter alone (acting at c):
    #   I -> I_c I_t,  X -> X_c Y_t,  Y -> Y_c Y_t,  Z -> Z_c I_t.
    a_effect_c = {"I":"I", "X":"X", "Y":"Y", "Z":"Z"}[la]
    a_effect_t = {"I":"I", "X":"Y", "Y":"Y", "Z":"I"}[la]

    # Effect of CY on the target letter alone (acting at t):
    #   I -> I_c I_t,  X -> Z_c X_t,  Y -> I_c Y_t,  Z -> Z_c Z_t.
    t_effect_c = {"I":"I", "X":"Z", "Y":"I", "Z":"Z"}[lt]
    t_effect_t = {"I":"I", "X":"X", "Y":"Y", "Z":"Z"}[lt]

    new_c, ph_c = _mul(a_effect_c, t_effect_c)
    new_t, ph_t = _mul(a_effect_t, t_effect_t)
    new_phase *= ph_c * ph_t
    if new_c != "I":
        new_letters[c] = new_c
    if new_t != "I":
        new_letters[t] = new_t
    return PauliString(new_phase, new_letters)


def _conjugate_swap(ps: PauliString, a: int, b: int) -> PauliString:
    new_letters = dict(ps.letters)
    la = new_letters.pop(a, None)
    lb = new_letters.pop(b, None)
    if lb is not None:
        new_letters[a] = lb
    if la is not None:
        new_letters[b] = la
    return PauliString(ps.phase, new_letters)


# ---------------------------------------------------------------------------
# Public entry point: conjugate a PauliString by a PennyLane operator.
# ---------------------------------------------------------------------------

def conjugate(ps: PauliString, op: qml.operation.Operator) -> PauliString:
    """Return ``U ps U†`` for a Clifford `op` (PennyLane gate)."""
    name = op.name
    wires = list(op.wires)
    if name == "SWAP":
        return _conjugate_swap(ps, wires[0], wires[1])
    if name == "CZ":
        return _conjugate_cz(ps, wires[0], wires[1])
    if name == "CNOT":
        return _conjugate_cnot(ps, wires[0], wires[1])
    if name == "CY":
        return _conjugate_cy(ps, wires[0], wires[1])
    if name in ("Hadamard", "H"):
        return _conjugate_single_qubit(ps, "Hadamard", wires[0])
    if name == "S":
        return _conjugate_single_qubit(ps, "S", wires[0])
    if name in ("PauliX", "X"):
        return _conjugate_single_qubit(ps, "PauliX", wires[0])
    if name in ("PauliY", "Y"):
        return _conjugate_single_qubit(ps, "PauliY", wires[0])
    if name in ("PauliZ", "Z"):
        return _conjugate_single_qubit(ps, "PauliZ", wires[0])
    if name in ("FSWAP", "FermionicSWAP"):
        # Conjugate through the gate's own decomposition (H CNOT CNOT H),
        # so the visualizer always matches the real hardware gate.
        return conjugate_through(ps, op.decomposition())
    raise ValueError(
        f"conjugate() only supports Clifford gates "
        f"(SWAP, CZ, CNOT, CY, Hadamard, S, PauliX/Y/Z, FSWAP); "
        f"got {name!r}."
    )


def conjugate_through(
    ps: PauliString, ops: Iterable[qml.operation.Operator]
) -> PauliString:
    """Return ``U ps U†`` where U = product of `ops` (last op applied first)."""
    cur = ps
    for op in ops:
        cur = conjugate(cur, op)
    return cur
