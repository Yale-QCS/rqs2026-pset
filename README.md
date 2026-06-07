# Fermionic Permutation — Problem Set (RQS Summer School 2026)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yale-QCS/rqs2026-pset/blob/main/problem_set.ipynb)

Hands-on coding exercise for the lecture **"Architecture and Compiler Design
for Hamiltonian Simulations"**: five problems on fermionic permutations under
the Jordan–Wigner mapping, from a single fermionic swap to a full Trotter step
of a 2-D Fermi–Hubbard model on two hardware topologies.

## Quick start

**Google Colab (recommended for the lecture):** click the badge above, then
run the *Setup* cells — they install PennyLane and fetch the `pset_tools`
toolkit automatically (~1 minute).

**Local Jupyter:**

```bash
git clone https://github.com/Yale-QCS/rqs2026-pset
cd rqs2026-pset
pip install -r requirements.txt
jupyter notebook problem_set.ipynb
```

## Files

| Path                | Purpose                                          |
|---------------------|--------------------------------------------------|
| `problem_set.ipynb` | The problem set (read & fill in)                 |
| `pset_tools/`       | The toolkit the notebook imports — short and readable; peek inside! |

## Toolkit cheatsheet

```python
from pset_tools import (
    verify_fp, verify_trotter_step,     # the two checkers
    depth_2q,                           # native+topology check, then 2q depth
    draw_majorana_tree,                 # the Majorana tree visualizer (static)
    interactive_majorana_tree,          # slider scrubber (ipywidgets)
    animate_majorana_tree,              # in-place gate-by-gate replay
    play_majorana_tree,                 # self-contained tweened animation
    snake_layout_figure,                # yellow snake JW figure
    fermion_lattice_figure,             # green fermion lattice figure
    snake_and_lattice_figure,           # the two side by side
    HORIZONTAL_EDGES, VERTICAL_EDGES, HOPPING_EDGES,
    hopping_pauli_strings,              # JW Pauli form of a hopping term
)
```

Solutions will be posted here after the session.
