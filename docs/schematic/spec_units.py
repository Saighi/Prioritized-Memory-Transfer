"""Unit-level spec: the two-population network unrolled to **2 units per population**.

Companion to ``spec.py`` (the population-level figure). Same idea — pure Python, no
torch, declares the *figure* — but at the granularity where the wiring becomes visible:

  - the **one-to-one interface**: each interface error unit connects exactly
    ``x_{S,i} <-> eps_{TS,i} <-> x_{T,i}`` (no crossing at the interface),
  - the **lateral cross-communication inside each population**: the recurrent weights
    have zero diagonal, so within T (and S) unit i talks to unit j only through the
    crossing pairs ``eps_i <-> x_j`` carrying ``W_{12}`` / ``W_{21}``.

Sign convention (Tang et al. 2023, Fig. 1): excitatory = arrowhead, inhibitory = dot.
It follows from the maths, per unit i (j the other unit):

  eps_{T,i} = x_{T,i} - W_{T,ij} x_{T,j}      -> x_i->eps_i excitatory, x_j->eps_i inhibitory
  dx_{T,i} propto -pi_T (eps_{T,i} - W_{T,ji} eps_{T,j}) + kappa eps_{TS,i}
                                              -> eps_i->x_i inhibitory, eps_j->x_i excitatory

The two arrows in a recurrent pair share the base weight W_ij without writing a
transpose on the individual return edge. The return path includes the population
precision, so its full coefficient is pi W_ij.
  eps_{TS,i} = x_{T,i} - x_{S,i}              -> x_T excitatory, x_S inhibitory into it
  dx_{S,i} gets +pi_TS eps_{TS,i}             -> excitatory
  x_T gets kappa eps_{TS,i}                   -> the terminal follows the phase:
      sleep excitatory, wake inhibitory; the label remains kappa in both cases.

Colours follow the paper legend: blue arrowhead = excitatory, orange dot = inhibitory
(``COLORS`` is shared from ``spec.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---- geometry: screen convention, +x right, +y DOWN, grid units (render flips y) ----
# Five rows, top to bottom: eps_S / x_S / eps_TS / x_T / eps_T; one column per unit.
@dataclass
class Node:
    id: str
    x: float
    y: float
    kind: str          # "value" | "error"  -> TikZ style name
    tex: str


@dataclass
class Edge:
    src: str
    dst: str
    tex: str           # label ('' = unlabelled, e.g. unit-gain identity lines)
    flow: str          # "pred" (blue) | "err" (orange); names retained for template compatibility
    sign: str          # "+" excitatory (arrowhead) | "-" inhibitory (dot)
    lpos: float = 0.5  # label position along the path
    bend: float = 16.0


XCOL = {1: 0.0, 2: 2.6}                       # unit columns
YROW = {"eS": 0.0, "xS": 1.5, "eTS": 3.0, "xT": 4.5, "eT": 6.0}


def build_spec(phase: str = "sleep", values: Optional[object] = None) -> Dict:
    """Return {nodes, edges, meta}. Same knobs as spec.build_spec."""
    if phase not in ("sleep", "wake"):
        raise ValueError("phase must be 'sleep' or 'wake'")

    def g(name: str, default_tex: str) -> str:
        if values is None or not hasattr(values, name):
            return default_tex
        v = getattr(values, name)
        if isinstance(v, str):
            return default_tex
        return f"{default_tex}={v:g}"

    piT = g("pi_T", r"\pi_T")
    piS = g("pi_S", r"\pi_S")
    piTS = g("pi_TS", r"\pi_{TS}")
    numeric_kappa = (values is not None and hasattr(values, "kappa")
                     and not isinstance(getattr(values, "kappa"), str))
    if numeric_kappa:
        drive = f"{abs(float(getattr(values, 'kappa'))):g}"
    else:
        drive = r"\kappa"
    drive_sign = "+" if phase == "sleep" else "-"   # the terminal IS the phase

    nodes: List[Node] = []
    for i in (1, 2):
        x = XCOL[i]
        nodes += [
            Node(f"eS{i}",  x, YROW["eS"],  "error", rf"\varepsilon_{{S,{i}}}"),
            Node(f"xS{i}",  x, YROW["xS"],  "value", rf"x_{{S,{i}}}"),
            Node(f"eTS{i}", x, YROW["eTS"], "error", rf"\varepsilon_{{TS,{i}}}"),
            Node(f"xT{i}",  x, YROW["xT"],  "value", rf"x_{{T,{i}}}"),
            Node(f"eT{i}",  x, YROW["eT"],  "error", rf"\varepsilon_{{T,{i}}}"),
        ]

    edges: List[Edge] = []
    for i in (1, 2):
        edges += [
            # student unit-i vertical pair
            Edge(f"xS{i}", f"eS{i}", "", "pred", "+"),
            Edge(f"eS{i}", f"xS{i}", piS, "err", "-"),
            # interface column i — the one-to-one mapping, no crossing here
            Edge(f"xS{i}", f"eTS{i}", "", "err", "-"),          # prediction (enters -)
            Edge(f"eTS{i}", f"xS{i}", piTS, "pred", "+"),        # student correction
            Edge(f"xT{i}", f"eTS{i}", "", "pred", "+"),         # source state (enters +)
            Edge(f"eTS{i}", f"xT{i}", drive, "pred", drive_sign), # teacher search drive
            # teacher unit-i vertical pair (eps_T,i = x_T,i - ...)
            Edge(f"xT{i}", f"eT{i}", "", "pred", "+"),
            Edge(f"eT{i}", f"xT{i}", piT, "err", "-"),
        ]
    # lateral cross-communication: W has zero diagonal, so within a population the
    # units talk ONLY through these crossing reciprocal pairs (cf. Tang Fig. 1B).
    for i, j in ((1, 2), (2, 1)):
        edges += [
            Edge(f"xT{j}", f"eT{i}", rf"W_{{T,{i}{j}}}", "err", "-", lpos=0.18, bend=10),
            Edge(f"eT{i}", f"xT{j}", rf"\pi_T W_{{T,{i}{j}}}", "pred", "+", lpos=0.18, bend=10),
            Edge(f"xS{j}", f"eS{i}", rf"W_{{S,{i}{j}}}", "err", "-", lpos=0.18, bend=10),
            Edge(f"eS{i}", f"xS{j}", rf"\pi_S W_{{S,{i}{j}}}", "pred", "+", lpos=0.18, bend=10),
        ]

    meta = {"phase": phase,
            "note_tex": r"\varepsilon_{TS,i}=x_{T,i}-x_{S,i}"}
    return {"nodes": nodes, "edges": edges, "meta": meta}


if __name__ == "__main__":
    sp = build_spec()
    print(f"{len(sp['nodes'])} nodes, {len(sp['edges'])} edges, phase={sp['meta']['phase']}")
    for e in sp["edges"]:
        lab = e.tex or "-"
        print(f"  {e.src:5s} -> {e.dst:5s}  {e.flow:4s} {e.sign}  {lab}")
