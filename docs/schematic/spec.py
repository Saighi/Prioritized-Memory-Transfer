"""Single source of truth for the two-population predictive-coding schematic.

Pure Python, **no torch** — this declares the *figure*: which value/error neurons
exist, the recurrent self-loops, and the labelled couplings between them. The TikZ
backend (``tikz/render.py``) consumes ``build_spec()``, so the diagram is parametrized
from this one place.

Labels are LaTeX math bodies *without* the surrounding ``$`` — the template adds them.

`build_spec` is the knob:
  - The teacher's interface drive uses the signed gain ``kappa``. ``phase="sleep"``
    (default) shows ``kappa>0`` for deficit ascent; ``phase="wake"`` shows ``kappa<0``
    for ordinary discrepancy descent. The sign is the phase; there is no separate gate.
  - ``values`` optionally folds numeric gains into the labels. The paper-minimal default
    leaves them symbolic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---- geometry: screen convention, +x right, +y DOWN, in grid units -----------
# Student on top (small y), teacher at the bottom (large y); the interface error is
# the central spine. render.py flips y and scales to cm.
@dataclass
class Node:
    id: str
    x: float
    y: float
    kind: str          # "value" | "error"  -> matches a TikZ style name
    tex: str
    pop: str           # "T" | "S" | "interface"  (decides self-loop side)
    loop: Optional[str] = None      # recurrent self-loop label, or None


@dataclass
class Edge:
    src: str
    dst: str
    tex: str
    flow: str          # "pred" (top-down) | "err" (bottom-up / drive)  -> TikZ style name
    label_pos: float = 0.5


def build_spec(phase: str = "sleep", values: Optional[object] = None) -> Dict:
    """Return {nodes, edges, meta}. `phase` in {"sleep","wake"}; `values` optional."""
    if phase not in ("sleep", "wake"):
        raise ValueError("phase must be 'sleep' or 'wake'")

    def g(name: str, default_tex: str) -> str:
        """Label for a gain: append `=value` if `values` carries a numeric attribute."""
        if values is None or not hasattr(values, name):
            return default_tex
        v = getattr(values, name)
        if isinstance(v, str):           # unresolved symbolic value
            return default_tex
        return f"{default_tex}={v:g}"

    piT = g("pi_T", r"\pi_T")
    piS = g("pi_S", r"\pi_S")
    piTS = g("pi_TS", r"\pi_{TS}")
    kappa = g("kappa", r"\kappa")

    # Positive teacher-side gain ascends the deficit during replay; negative gain gives
    # ordinary discrepancy descent during wake. The phase is the sign.
    numeric_kappa = (values is not None and hasattr(values, "kappa")
                     and not isinstance(getattr(values, "kappa"), str))
    if numeric_kappa:
        drive = kappa
    else:
        drive = kappa + (">0" if phase == "sleep" else "<0")

    nodes: List[Node] = [
        Node("x_S",  4.0, 0.0, "value", r"x_S", "S", loop=r"W_S"),
        Node("e_S",  1.4, 0.0, "error", r"\varepsilon_S", "S"),
        Node("e_TS", 4.0, 2.0, "error", r"\varepsilon_{TS}", "interface"),
        Node("x_T",  4.0, 4.0, "value", r"x_T", "T", loop=r"W_T"),
        Node("e_T",  6.6, 4.0, "error", r"\varepsilon_T", "T"),
    ]

    edges: List[Edge] = [
        # student self-error pair
        Edge("x_S", "e_S", r"M_S", "err"),
        Edge("e_S", "x_S", piS, "pred"),
        # interface spine (each pair separates under the uniform bend in the template)
        Edge("x_S", "e_TS", r"-x_S", "pred"),   # descending prediction enters negatively
        Edge("e_TS", "x_S", piTS, "err"),        # ascending student correction
        Edge("x_T", "e_TS", r"+x_T", "err", 0.36),  # source state enters positively
        Edge("e_TS", "x_T", drive, "pred", 0.36),    # teacher search drive
        # teacher self-error pair
        Edge("x_T", "e_T", r"M_T", "err"),
        Edge("e_T", "x_T", piT, "pred"),
    ]

    meta = {"phase": phase, "note_tex": r"\varepsilon_{TS}=x_T-x_S"}
    return {"nodes": nodes, "edges": edges, "meta": meta}


# Colours, as bare HTML hex (no '#') for xcolor's HTML model. Mid-tones so the figure
# reads on both white paper and dark slides.
COLORS = {"pred": "4C8DFF", "err": "E0772B",
          "value_stroke": "222222", "error_stroke": "888888"}


if __name__ == "__main__":
    sp = build_spec()
    print(f"{len(sp['nodes'])} nodes, {len(sp['edges'])} edges, phase={sp['meta']['phase']}")
    for n in sp["nodes"]:
        print(f"  node {n.id:5s} ({n.kind:5s}) at ({n.x},{n.y})  {n.tex}")
    for e in sp["edges"]:
        print(f"  edge {e.src:5s} -> {e.dst:5s}  {e.flow:4s}  {e.tex}")
