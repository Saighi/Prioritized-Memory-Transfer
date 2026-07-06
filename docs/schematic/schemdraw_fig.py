"""schemdraw backend for the two-population PC schematic — pure Python, no LaTeX.

    python schemdraw_fig.py [--phase sleep|wake] [--show]

Consumes the shared spec.py (same nodes/edges/labels as every other backend) and
renders schemdraw_<phase>.svg + .png. Labels use matplotlib mathtext, so the LaTeX
bodies in spec.py (\\varepsilon, \\pi_{ST}, ...) render directly.

This is the one that drops cleanly into the package as pmt/viz_schematic.py: it needs
no external toolchain and shows inline in the VS Code #%% interactive window.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec                                       # noqa: E402

import schemdraw                                  # noqa: E402
from schemdraw import flow                        # noqa: E402

SCALE = 1.6


def _anchors(a, b):
    """Pick boundary anchors for an arrow from node `a` to node `b` by geometry."""
    dx, dy = b.x - a.x, b.y - a.y                 # spec coords: +y is DOWN
    if abs(dy) >= abs(dx):
        return ("S", "N") if dy > 0 else ("N", "S")
    return ("E", "W") if dx > 0 else ("W", "E")


def build(phase: str) -> schemdraw.Drawing:
    sp = spec.build_spec(phase=phase)
    C = spec.COLORS
    d = schemdraw.Drawing()
    d.config(fontsize=13)

    # --- place nodes; keep refs + spec coords for anchor geometry -----------
    placed = {}
    for n in sp["nodes"]:
        xy = (n.x * SCALE, -n.y * SCALE)          # flip y: teacher on top
        if n.kind == "value":
            el = flow.Circle(r=0.55).at(xy).label(f"${n.tex}$",
                                                  color=C["value_stroke"])
            el.color(C["value_stroke"]).linewidth(1.6)
        else:
            el = flow.RoundBox(w=1.25, h=0.85).at(xy).label(f"${n.tex}$",
                                                            color=C["text"])
            el.color(C["error_stroke"]).linestyle("--").linewidth(1.0)
        d += el
        el.x, el.y = n.x, n.y                      # stash spec coords on the element
        placed[n.id] = el

    # --- recurrent self-loops (W_T above x_T, W_S below x_S) ----------------
    for n in sp["nodes"]:
        if not n.loop:
            continue
        el = placed[n.id]
        if n.pop == "T":
            d += flow.Arc2(arrow="<-", k=0.8, color="gray").at(el.NW).to(el.NE) \
                 .label(f"${n.loop}$", loc="top", color="gray", fontsize=11)
        else:
            d += flow.Arc2(arrow="->", k=-0.8, color="gray").at(el.SW).to(el.SE) \
                 .label(f"${n.loop}$", loc="bottom", color="gray", fontsize=11)

    # --- couplings ----------------------------------------------------------
    for e in sp["edges"]:
        a, b = placed[e.src], placed[e.dst]
        sa, sb = _anchors(a, b)
        col = C[e.flow]
        d += (flow.Arc2(arrow="->", k=0.28, color=col)
              .at(getattr(a, sa)).to(getattr(b, sb))
              .label(f"${e.tex}$", color=col, fontsize=11))

    # --- captions -----------------------------------------------------------
    d += flow.Box(w=0.01, h=0.01).at((-0.2 * SCALE, 0.9)) \
         .label("TEACHER T", loc="left", color="gray", fontsize=10).color("white")
    d += flow.Box(w=0.01, h=0.01).at((-0.2 * SCALE, -4 * SCALE - 0.9)) \
         .label("STUDENT S", loc="left", color="gray", fontsize=10).color("white")
    d += flow.Box(w=0.01, h=0.01).at((6.9 * SCALE, -2 * SCALE)) \
         .label(f"${sp['meta']['note_tex']}$", loc="right",
                color="gray", fontsize=10).color("white")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sleep", choices=["sleep", "wake"])
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    d = build(args.phase)
    for ext in ("svg", "png"):
        out = HERE / f"schemdraw_{args.phase}.{ext}"
        d.save(str(out), dpi=200)
        print(f"wrote {out.name}")
    if args.show:
        d.draw()


if __name__ == "__main__":
    main()
