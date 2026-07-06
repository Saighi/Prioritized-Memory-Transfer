"""Single source of truth for the two-population predictive-coding schematic.

Pure Python, **no torch** — this just declares the *figure*: which value/error
neurons exist, the recurrent self-loops, and the labelled couplings between them.
All four backend generators (schemdraw / TikZ / D2 / Penrose) import `build_spec`
so the diagram stays identical no matter which tool draws it, and so the whole
thing is parametrized from one place.

Two label strings are carried per item:
  - ``txt`` : Unicode (for D2 / Penrose / plain renderers)
  - ``tex`` : LaTeX math body, no ``$`` (for TikZ; also fed to schemdraw, whose
              matplotlib backend understands a mathtext subset incl. \\varepsilon)

`build_spec` is the knob:
  - The teacher's interface drive uses the **signed** precision ``π_ST``. ``phase="sleep"``
    (default) shows it as a **reversed (negative) precision** ``π_ST<0`` (drive-to-disagree);
    ``phase="wake"`` shows the ordinary ``π_ST>0`` (recall/inference). The sign *is* the phase —
    there is no separate ±1 gate.
  - ``values`` (optional ModelConfig-like) folds numeric gains into the labels,
    e.g. ``π_ST=-0.21``. The paper-minimal default leaves them symbolic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---- geometry: screen convention, +x right, +y DOWN, in grid units -----------
# Teacher sits on top (small y), student on the bottom (large y); the interface
# error is the central spine. Backends with their own layout engine (D2, Penrose)
# may ignore these and just use the edges.
@dataclass
class Node:
    id: str
    x: float
    y: float
    kind: str          # "value" | "error"
    txt: str
    tex: str
    pop: str           # "T" | "S" | "interface"
    loop: Optional[str] = None      # recurrent self-loop label (tex), or None


@dataclass
class Edge:
    src: str
    dst: str
    txt: str
    tex: str
    flow: str          # "pred" (top-down) | "err" (bottom-up / drive)
    bend: float = 0.0  # signed lateral offset for parallel spine arrows


def build_spec(phase: str = "sleep", values: Optional[object] = None) -> Dict:
    """Return {nodes, edges, meta}. `phase` in {"sleep","wake"}; `values` optional."""
    if phase not in ("sleep", "wake"):
        raise ValueError("phase must be 'sleep' or 'wake'")

    # optional numeric folding ------------------------------------------------
    def g(sym: str, name: str, default_tex: str) -> tuple[str, str]:
        """Return (txt, tex); append =value if `values` carries the attribute."""
        if values is None or not hasattr(values, name):
            return sym, default_tex
        v = getattr(values, name)
        if isinstance(v, str):           # e.g. pi_ST="auto", unresolved
            return sym, default_tex
        return f"{sym}={v:g}", f"{default_tex}={v:g}"

    piT_t, piT_x = g("π_T", "pi_T", r"\pi_T")
    piS_t, piS_x = g("π_S", "pi_S", r"\pi_S")
    piTS_t, piTS_x = g("π_TS", "pi_TS", r"\pi_{TS}")
    piST_t, piST_x = g("π_ST", "pi_ST", r"\pi_{ST}")

    # The teacher's interface drive is the SIGNED precision π_ST: reversed (negative) in
    # sleep, ordinary (positive) in wake. The phase IS the sign — no separate ±1 factor.
    numeric_piST = (values is not None and hasattr(values, "pi_ST")
                    and not isinstance(getattr(values, "pi_ST"), str))
    if numeric_piST:
        drive_txt, drive_tex = piST_t, piST_x        # the folded value already carries the sign
    else:
        cond = "<0" if phase == "sleep" else ">0"    # symbolic: show the sign as the phase
        drive_txt = f"{piST_t}{cond}"
        drive_tex = f"{piST_x}{cond}"

    nodes: List[Node] = [
        Node("x_T",  4.0, 0.0, "value", "x_T",  r"x_T", "T", loop=r"W_T"),
        Node("e_T",  1.4, 0.0, "error", "ε_T",  r"\varepsilon_T", "T"),
        Node("e_TS", 4.0, 2.0, "error", "ε_TS", r"\varepsilon_{TS}", "interface"),
        Node("x_S",  4.0, 4.0, "value", "x_S",  r"x_S", "S", loop=r"W_S"),
        Node("e_S",  6.6, 4.0, "error", "ε_S",  r"\varepsilon_S", "S"),
    ]

    edges: List[Edge] = [
        # teacher self-error pair
        Edge("x_T", "e_T", "M_T", r"M_T", "err"),
        Edge("e_T", "x_T", piT_t, piT_x, "pred"),
        # interface spine (two parallel arrows each segment)
        Edge("x_T", "e_TS", "x_T", r"x_T", "pred", bend=-0.32),
        Edge("e_TS", "x_T", drive_txt, drive_tex, "err", bend=+0.32),
        Edge("x_S", "e_TS", "x_S", r"x_S", "err", bend=+0.32),
        Edge("e_TS", "x_S", piTS_t, piTS_x, "pred", bend=-0.32),
        # student self-error pair
        Edge("x_S", "e_S", "M_S", r"M_S", "err"),
        Edge("e_S", "x_S", piS_t, piS_x, "pred"),
    ]

    meta = {
        "phase": phase,
        "title_txt": f"Two-population PC network — {phase}",
        "title_tex": rf"Two-population PC network --- \emph{{{phase}}}",
        "legend": [("pred", "top-down prediction"), ("err", "bottom-up / drive")],
        # interface error definition, shown as a caption note
        "note_txt": "ε_TS = x_S − x_T",
        "note_tex": r"\varepsilon_{TS}=x_S-x_T",
    }
    return {"nodes": nodes, "edges": edges, "meta": meta}


# Two named colours used consistently across backends (mid-tones: read on light & dark)
COLORS = {"pred": "#4C8DFF", "err": "#E0772B",
          "value_stroke": "#222222", "error_stroke": "#888888",
          "text": "#222222"}


if __name__ == "__main__":
    sp = build_spec()
    print(f"{len(sp['nodes'])} nodes, {len(sp['edges'])} edges, phase={sp['meta']['phase']}")
    for n in sp["nodes"]:
        print(f"  node {n.id:5s} ({n.kind:5s}) at ({n.x},{n.y})  {n.txt}")
    for e in sp["edges"]:
        print(f"  edge {e.src:5s} -> {e.dst:5s}  {e.flow:4s}  {e.txt}")
