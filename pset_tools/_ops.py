"""Normalize a ``circuit`` argument to a flat list of PennyLane Operators.

Every tool that takes a "circuit" (``verify_fp``, ``verify_trotter_step``,
``depth_2q``, the Majorana visualizer) routes its input through
:func:`flatten_ops`, so they all accept the same shapes.

A circuit may be given as:

* a single gate (an :class:`~pennylane.operation.Operator`);
* a list/tuple of gates -- and the list **may be nested**, so you can group
  sub-circuits as "gadgets", e.g. ``[fswap(2), fswap(3)]`` or
  ``[basis_change, gadget_a(), gadget_b()]``.  Nested lists are flattened
  depth-first, left-to-right -- exactly the order you wrote them, which is the
  order they are applied;
* a :class:`~pennylane.tape.QuantumTape` / ``QuantumScript``;
* a quantum function (called once to capture its gates).
"""

from __future__ import annotations

from typing import List

import pennylane as qml


def flatten_ops(circuit) -> List[qml.operation.Operator]:
    """Return ``circuit`` as a flat list of Operators (see module docstring)."""
    return _flatten(circuit, top=True)


def _flatten(circuit, top: bool) -> List[qml.operation.Operator]:
    if isinstance(circuit, qml.tape.QuantumScript):
        return list(circuit.operations)
    if isinstance(circuit, qml.operation.Operator):
        return [circuit]
    if isinstance(circuit, (list, tuple)):
        ops: List[qml.operation.Operator] = []
        for item in circuit:
            ops.extend(_flatten(item, top=False))
        return ops
    if top and callable(circuit):
        with qml.tape.QuantumTape() as tape:
            circuit()
        return list(tape.operations)
    # --- not a recognizable circuit element: give a targeted hint ---
    if callable(circuit):
        raise TypeError(
            f"found a function "
            f"({getattr(circuit, '__name__', 'callable')!r}) inside the "
            f"circuit instead of gates -- did you forget to CALL your gadget, "
            f"e.g. `fswap(2)` rather than `fswap`?"
        )
    raise TypeError(
        f"expected a gate, a (possibly nested) list of gates, a QuantumTape, "
        f"or a quantum function; got {type(circuit).__name__}"
    )
