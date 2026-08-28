"""Single source of truth for the training-protocol figure (two panels, A|B).

A *state* figure, not a wiring one: the same recurrent-network glyph shown under two
conditions, so time reads left (acquisition) to right (consolidation). The detailed
mechanism lives in ``spec_units.py``; this figure only says **when each component is
active and plastic**.

Only the genuinely invariant / repeated parts live here — the thing the figure is most
insistent about, *same node positions and same spatial shade pattern across panels*, is
therefore mechanical rather than eyeballed. All the bespoke chrome (clamp, locks,
captions, legend) is hard-coded in ``tikz_protocol/protocol.tex.j2``.

To tweak the figure you almost always touch just this file:
  - ``RING``        node ring positions (shared by both panels)
  - ``CHORDS``      the few representative recurrent connections drawn on the ring
  - ``ACTIVITY``    per-panel shade of each node, 0..1 (0 = silent, 1 = full)
  - ``build_panel`` the per-population state flags: 'plastic' | 'frozen' | 'off'
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


# ---- ring geometry: 8 nodes, unit circle, +y DOWN (render.py flips y) --------------
N_RING = 8
R_RING = 1.15                                    # ring radius, cm before SX/SY
# Recurrent connectivity is all-to-all with zero diagonal. Draw the COMPLETE graph among
# the ring nodes as a thin, semi-transparent mesh (a *texture*, not wires to be traced),
# so the reader reads "fully connected" rather than "a few representative links".
CHORDS = [(i, j) for i in range(N_RING) for j in range(i + 1, N_RING)]


def _ring(cx: float, cy: float) -> List[Dict]:
    """Eight node centres on a ring around (cx, cy). Index 0 at top, clockwise."""
    out = []
    for i in range(N_RING):
        a = -math.pi / 2 + 2 * math.pi * i / N_RING
        out.append({"i": i, "x": cx + R_RING * math.cos(a), "y": cy + R_RING * math.sin(a)})
    return out


# ---- per-panel node activity (shade), 0..1 — the SAME spatial pattern in both -------
# One representative pattern lit on the ring. Panel A: teacher lit, student silent.
# Panel B: same pattern in both, student a notch weaker so x_S != x_T and the
# plasticity signal eps_TS is still visibly alive (the completed state would be equal
# shades, eps_TS = 0, and is deliberately NOT what we draw).
_PATTERN = [0.90, 0.15, 0.55, 0.15, 0.80, 0.20, 0.60, 0.15]
_STUDENT_DIM = 0.72                              # student shades = pattern * this, in B

ACTIVITY = {
    "A": {"T": _PATTERN, "S": [0.0] * N_RING},
    "B": {"T": _PATTERN, "S": [v * _STUDENT_DIM for v in _PATTERN]},
}


@dataclass
class Panel:
    key: str                                     # "A" | "B"
    title: str
    tnodes: List[Dict]                           # teacher ring nodes (with shade)
    snodes: List[Dict]                           # student ring nodes (with shade)
    chords: List                                 # shared recurrent chords
    t_state: str                                 # 'plastic' | 'frozen' | 'off'
    s_state: str
    iface: str                                   # 'on' | 'off'  (teacher<->student coupling)
    clamp: bool                                  # external x_T <- m^(p) clamp on teacher
    meta: Dict = field(default_factory=dict)


# panel layout in cm (pre-scale): student ring on top, teacher ring below, per column
CY_S, CY_T = 0.0, 3.7                            # ring centres (y DOWN); gap = CY_T - 2*R_RING


def build_panel(key: str, cx: float) -> Panel:
    if key not in ("A", "B"):
        raise ValueError("panel key must be 'A' or 'B'")
    act = ACTIVITY[key]
    tnodes = [dict(n, shade=act["T"][n["i"]]) for n in _ring(cx, CY_T)]
    snodes = [dict(n, shade=act["S"][n["i"]]) for n in _ring(cx, CY_S)]

    if key == "A":                               # acquisition
        return Panel("A", "Acquisition", tnodes, snodes, CHORDS,
                     t_state="plastic", s_state="off", iface="off", clamp=True,
                     meta={"cx": cx})
    return Panel("B", "Consolidation", tnodes, snodes, CHORDS,   # consolidation
                 t_state="frozen", s_state="plastic", iface="on", clamp=False,
                 meta={"cx": cx})


PANEL_DX = 6.5                                    # centre-to-centre spacing of the panels


def build_scene() -> Dict:
    panels = [build_panel("A", 0.0), build_panel("B", PANEL_DX)]
    return {"panels": panels,
            "meta": {"note_tex": r"\varepsilon_{TS}=x_T-x_S"}}


# Colours (bare HTML hex for xcolor). Blue/orange coupling reuse spec.COLORS in render.
COLORS = {
    "plastic": "4C8DFF",     # plastic recurrent weights, blue (Wdot != 0)
    "frozen":  "9AA0A6",     # frozen weights, gray (Wdot = 0, locked)
    "pred":    "4C8DFF",     # teacher -> student, blue arrowhead (excitatory)
    "err":     "E0772B",     # student -> teacher, orange dot (inhibitory)
    "node_stroke": "222222",
    "shadecol": "3C4043",    # base colour that node fill is a %-tint of
    "clamp":   "222222",
    "title":   "222222",
    "muted":   "6B7075",
}
OPACITY_OFF = 0.2            # inactive components dimmed to this


if __name__ == "__main__":
    sc = build_scene()
    for p in sc["panels"]:
        print(f"panel {p.key} ({p.title}): T={p.t_state} S={p.s_state} "
              f"iface={p.iface} clamp={p.clamp}")
