"""Two-qubit depth report (with a native-gate + topology check).

There is no compiler in this pset.  You build your circuit out of native
gates; `depth_2q` then (optionally) validates the gate set and the hardware
topology, and reports the **two-qubit (CNOT-level) depth** — the length of the
critical path counting only two-qubit gates.  This is computed exactly as
qiskit's `circuit.depth(lambda g: g.operation.num_qubits == 2)`: every gate is
walked in order; single-qubit gates are threaded through (they add no depth)
rather than dropped.

    depth_2q(circuit)              # native-gate check only, then 2q-depth
    depth_2q(circuit, "ata")       # same (all-to-all has no topology limit)
    depth_2q(circuit, "2dnn")      # also require every 2q gate grid-adjacent

Native gate set: single-qubit (anything) + two-qubit {CNOT (=CX), CY, CZ}.  Any
other gate is accepted *iff* it recursively `.decomposition()`s down to those —
so `SWAP` counts as its three CNOTs, your fermionic swap as whatever primitives
you built it from, and even `qml.PauliRot` lowers to its CNOT staircase.  A
multi-qubit gate with no decomposition is rejected.

CY is native because CY = (I (x) S) . CNOT . (I (x) S^dag) — one CNOT plus
single-qubit gates — so a CY costs exactly one two-qubit layer.  This is why the
Givens / XY rotation `qml.IsingXY(2*theta, [p, q])` (= H . CY . RY . RX . CY . H)
counts as depth 2: it is the natural two-CNOT form of exp(i*theta/2 (XX+YY)).
(We whitelist CY directly rather than lower it, because PennyLane's *default* CY
decomposition routes through CRY = 2 CNOTs, which would over-count it.)
"""

from __future__ import annotations

from typing import List, Optional

import pennylane as qml

from ._ops import flatten_ops as _as_ops
from .jw import is_grid_adjacent

# Two-qubit gates we count as one native layer each: CNOT (=CX), CY, CZ.
# CY is included on purpose -- it is one CNOT up to single-qubit S gates, so it
# costs one layer; lowering it via PennyLane's default (CY -> CRY -> 2 CNOTs)
# would wrongly double it.  This makes qml.IsingXY (the Givens hop) cost depth 2.
_NATIVE_2Q = {"CNOT", "CZ", "CY"}
# Guard against a pathological gate whose decomposition never bottoms out.
_MAX_DECOMP_DEPTH = 64


def _native_gates(ops, _depth: int = 0) -> List[qml.operation.Operator]:
    """Lower a flat op list to native gates, KEEPING single-qubit gates.

    Every returned op is either single-qubit (kept untouched) or a native
    two-qubit gate (CNOT/CZ/CY).  Any other multi-qubit gate is recursively
    replaced by its `.decomposition()`; the recursion bottoms out when
    everything is native.  A multi-qubit gate with no decomposition raises.

    We deliberately *keep* the single-qubit gates rather than dropping them, so
    the depth pass below can walk the circuit exactly as written -- the same way
    qiskit's `QuantumCircuit.depth` visits every instruction.
    """
    if _depth > _MAX_DECOMP_DEPTH:
        raise ValueError("gate decomposition did not terminate in native gates")
    out: List[qml.operation.Operator] = []
    for op in ops:
        n = len(op.wires)
        if n <= 1 or op.name in _NATIVE_2Q:
            out.append(op)
            continue
        # Not natively two-qubit -> lower it.  Accepts SWAP (3 CNOTs), a
        # student-built fermionic swap (whatever primitives they chose),
        # qml.PauliRot (its CNOT staircase), etc.
        try:
            decomp = op.decomposition()
        except Exception as exc:  # qml.DecompositionUndefinedError and friends
            raise ValueError(
                f"gate {op.name!r} on wires {list(op.wires)} is not native and "
                f"has no decomposition into native gates: build it yourself out "
                f"of single-qubit gates + CNOT/CZ."
            ) from exc
        out.extend(_native_gates(decomp, _depth + 1))
    return out


def depth_2q(circuit, connectivity: Optional[str] = None) -> int:
    """Validate (gate set, and topology if given) and return two-qubit depth.

    Args:
        circuit: a gate, a (possibly nested) list of gates, a QuantumTape, or a
            quantum function.
        connectivity: None or "ata" — native-gate check only.
                      "2dnn" — additionally require every two-qubit gate to act
                      on a physically-adjacent pair of the 3x3 snake grid.

    The two-qubit depth is the length of the critical path counting only
    two-qubit gates.  We compute it exactly as qiskit's
    ``circuit.depth(lambda g: g.operation.num_qubits == 2)`` does: walk EVERY
    gate in order keeping a per-qubit level, and for each gate set all the
    qubits it touches to ``max(level over its qubits) + (1 if it is a two-qubit
    gate else 0)``.  The depth is the max level over all qubits.  Single-qubit
    gates are *threaded through* (they neither merge wires nor add depth) rather
    than dropped, so the schedule never diverges from the circuit as written.

    Depth follows the primitives you used: a fermionic swap built as
    H·CNOT·CNOT·H costs 2, the same gate built as SWAP·CZ costs 4.
    """
    if connectivity not in (None, "ata", "2dnn"):
        raise ValueError("connectivity must be None, 'ata', or '2dnn'")

    gates = _native_gates(_as_ops(circuit))

    # qiskit-style critical-path layering over EVERY gate (1q threaded, 2q counted).
    level: dict = {}
    for g in gates:
        w = [int(x) for x in g.wires]
        is_2q = len(w) == 2           # after lowering, every 2-wire gate is native
        if is_2q and connectivity == "2dnn" and not is_grid_adjacent(w[0], w[1]):
            raise ValueError(
                f"2D-NN violation: {g.name} on ({w[0]}, {w[1]}) is not "
                f"nearest-neighbor on the 3x3 snake grid.  Route it with adjacent "
                f"fermionic swaps, or gather parity along a contiguous chain "
                f"(not a long-range tree)."
            )
        base = max((level.get(q, 0) for q in w), default=0)
        lvl = base + 1 if is_2q else base
        for q in w:
            level[q] = lvl
    return max(level.values(), default=0)
