"""Single source of truth for the PFC -> hippocampus schematic (biological framing).

Pure Python, **no torch**. Declares four *propositions* for the same idea, so the
paper can pick one:

  ``a`` / ``minimal``    two regions, two arrows: the asymmetry and nothing else.
  ``b`` / ``pathways``   the same, with the two descending routes drawn explicitly.
  ``c`` / ``regimes``    coordinated vs independent PFC ripples: the sign flip.
  ``d`` / ``bridge``     biology on the left, the model interface on the right.

Everything is declared as a flat scene (boxes / links / texts / rules / legend);
``tikz_bio/pfc_hpc.tex.j2`` renders it without knowing which variant it is drawing.

Conventions
-----------
* Coordinates are **centimetres**, screen convention: +x right, **+y DOWN**.
  ``render.py`` flips y and scales by ``SX``/``SY`` (both 1.0: this figure is
  hand-placed, not grid-placed like ``spec.py``).
* ``Link.path`` is a raw TikZ ``to[...]`` option string, so curvature stays here
  rather than being re-derived in the template.
* Colour semantics are **geometric**: descending (cortex -> hippocampus) is blue,
  ascending (hippocampus -> cortex) is orange. With the receiver (student, cortex)
  drawn on top, that coincides with the prediction/error colours of ``spec.py``.
* Terminals follow the ``tikz_units`` convention borrowed from Tang et al. 2023:
  arrowhead = excitatory, filled dot = inhibitory.

Biology (see ``tikz_bio/README.md`` for the citations)
-----------------------------------------------------
Ascending CA1 -> PFC is direct and excitatory. Descending PFC -> CA1 is not: it is
(i) a monosynaptic long-range GABAergic projection and (ii) a disynaptic relay through
nucleus reuniens that recruits local feedforward inhibition. Both end as suppression
of CA1 principal cells, which is what the model's receiver-to-source suppression expresses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

try:                                   # same directory as spec.py; keep one palette
    from spec import COLORS as _BASE
except ImportError:                    # importable standalone too
    _BASE = {"pred": "4C8DFF", "err": "E0772B",
             "value_stroke": "222222", "error_stroke": "888888"}

# Bare HTML hex (no '#') for xcolor's HTML model.
COLORS = {
    "pred": _BASE["pred"],             # descending / top-down (cortex -> hippocampus)
    "err": _BASE["err"],               # ascending / bottom-up (hippocampus -> cortex)
    "region_stroke": _BASE["value_stroke"],
    "relay_stroke": _BASE["error_stroke"],
    "inter_fill": "6E6E6E",            # local inhibitory interneuron (filled disc)
    "rule": "C9C9C9",                  # hairlines, correspondence dots
    "title": "222222",
}

VARIANTS = ("a", "b", "c", "d")
TITLES = {"a": "minimal", "b": "pathways", "c": "regimes", "d": "bridge"}


# ---- scene elements ---------------------------------------------------------
@dataclass
class Box:
    id: str
    x: float
    y: float
    kind: str = "region"               # region | relay | inter | value | err | ghost
    title: str = ""
    sub: str = ""                      # small grey second line
    w: float = 0.0                     # cm, 0 -> style default
    h: float = 0.0
    font: str = r"\footnotesize\bfseries"
    loop: str = ""                     # recurrent self-loop label (value nodes)
    loop_side: str = "below"


@dataclass
class Link:
    src: str
    dst: str
    flow: str = "down"                 # down | up | local | corr  -> TikZ style name
    term: str = "excit"                # excit | inhib | "" (no terminal)
    path: str = "bend left=15"         # raw TikZ `to[...]` options
    label: str = ""
    lpos: float = 0.5
    lopt: str = ""                     # extra options on the label node
    opts: str = ""                     # extra options on the draw (dashed, opacity, ...)


@dataclass
class Text:
    x: float
    y: float
    tex: str
    style: str = "note"                # note | panel | ttl | lgd
    anchor: str = "center"


@dataclass
class Rule:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Legend:
    x: float
    y: float
    term: str
    flow: str
    text: str
    length: float = 0.55


@dataclass
class Scene:
    boxes: List[Box] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    texts: List[Text] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    legend: List[Legend] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


# Reused label fragments -------------------------------------------------------
PFC_T, PFC_S = "PREFRONTAL CORTEX", r"$x_S$ (receiver)"
HPC_T, HPC_S = "HIPPOCAMPUS (CA1)", r"$x_T$ (source)"


def up_path(loose: float = 1.0) -> str:
    """Ascending arc, bulging right of the column."""
    return f"out=58, in=-58, looseness={loose:g}"


def down_path(loose: float = 1.0) -> str:
    """Descending arc, bulging left of the column."""
    return f"out=-122, in=122, looseness={loose:g}"


def _terminal_legend(x: float, y: float) -> List[Legend]:
    return [Legend(x, y, "excit", "local", "excitatory"),
            Legend(x + 2.55, y, "inhib", "local", "inhibitory")]


# ---- a. minimal --------------------------------------------------------------
def _variant_a() -> Scene:
    """Two regions, one arrow each way. The asymmetry, and nothing else."""
    s = Scene()
    s.boxes = [
        Box("pfc", 0.0, 0.0, "region", PFC_T, PFC_S, w=4.8, h=1.35),
        Box("hpc", 0.0, 4.2, "region", HPC_T, HPC_S, w=4.8, h=1.35),
    ]
    s.links = [
        Link("hpc", "pfc", "up", "excit", up_path(1.15),
             label=r"reactivation\\[1pt]{\tiny SWR replay}",
             lopt="anchor=west, xshift=3pt"),
        Link("pfc", "hpc", "down", "inhib", down_path(1.15),
             label=r"top-down\\suppression\\[1pt]{\tiny PFC ripples}",
             lopt="anchor=east, xshift=-3pt"),
    ]
    s.texts = [Text(0.0, 6.35,
                    r"the receiver suppresses the source it also learns from:"
                    r"\\[1pt]in the model, the teacher ascends the interface"
                    r" discrepancy ($\kappa>0$)")]
    s.legend = _terminal_legend(-2.35, 5.55)
    s.meta = {"name": "minimal"}
    return s


# ---- b. pathways -------------------------------------------------------------
def _variant_b() -> Scene:
    """Same story, with the two descending routes drawn: direct GABAergic, and
    the thalamic relay that recruits local feedforward inhibition."""
    s = Scene()
    s.boxes = [
        Box("pfc", 0.0, 0.0, "region", PFC_T, PFC_S, w=5.2, h=1.45),
        Box("hpc", 0.0, 6.0, "region", HPC_T, HPC_S, w=5.2, h=1.45),
        Box("re", -5.3, 3.0, "relay", "nucleus reuniens", "thalamus",
            w=2.5, h=1.0, font=r"\scriptsize"),
        Box("inn", -2.3, 4.2, "inter"),
    ]
    s.links = [
        # descending route 1: monosynaptic long-range GABAergic (straight = direct)
        Link("pfc", "hpc", "down", "inhib", "out=-90, in=90",
             label=r"long-range\\GABAergic\\[1pt]{\tiny monosynaptic}",
             lpos=0.36, lopt="anchor=east, xshift=-3pt"),
        # descending route 2: via nucleus reuniens -> local interneuron
        Link("pfc", "re", "down", "excit", "out=200, in=95"),
        Link("re", "inn", "down", "excit", "out=-30, in=180"),
        # NB: a size command (\tiny) before a `\\` breaks the node's align=center
        # cell, so line-broken labels set the size through `font=` instead.
        # `dst` may be any TikZ coordinate: drop straight onto CA1 below the disc
        # rather than aiming at the box centre.
        Link("inn", "hpc.north -| inn", "local", "inhib", "out=-90, in=90",
             label=r"feedforward\\inhibition",
             lopt=r"anchor=west, xshift=3pt, font=\tiny"),
        # ascending route: direct and excitatory
        Link("hpc", "pfc", "up", "excit", up_path(1.12),
             label=r"reactivation\\[1pt]{\tiny SWR replay, CA1 $\to$ PFC}",
             lopt="anchor=west, xshift=3pt"),
    ]
    s.texts = [
        Text(-2.3, 3.7, r"{\tiny local interneuron}", "note"),
        Text(0.0, 8.05,
             r"two descending routes, one net effect: suppression of CA1 principal"
             r" cells.\\[1pt]the ascending route is direct and excitatory."
             r" the model writes the teacher's search gain as $\kappa>0$."),
    ]
    s.legend = _terminal_legend(-2.3, 7.35)
    s.meta = {"name": "pathways"}
    return s


# ---- c. regimes --------------------------------------------------------------
def _variant_c() -> Scene:
    """Coordinated vs independent PFC ripples: the same circuit, opposite sign."""
    s = Scene()
    dx = 5.8
    for k, (sfx, ox) in enumerate((("1", 0.0), ("2", dx))):
        s.boxes += [
            Box("pfc" + sfx, ox, 0.0, "region", "PFC", w=2.6, h=1.0),
            Box("hpc" + sfx, ox, 3.0, "region", "CA1", w=2.6, h=1.0),
        ]
    s.links = [
        # coordinated: cooperative, ordinary predictive coding
        Link("hpc1", "pfc1", "up", "excit", up_path(1.05), label="replay",
             lopt="anchor=west, xshift=2pt"),
        Link("pfc1", "hpc1", "down", "excit", down_path(1.05), label="predict",
             lopt="anchor=east, xshift=-2pt"),
        # independent PFC ripples: CA1 reactivation is suppressed
        Link("hpc2", "pfc2", "up", "excit", up_path(1.05),
             label=r"{\color{rulecol}replay}",
             lopt="anchor=west, xshift=4pt, fill=none",   # no halo: keep the dashes
             opts="dashed, opacity=0.45"),
        Link("pfc2", "hpc2", "down", "inhib", down_path(1.05), label="suppress",
             lopt="anchor=east, xshift=-2pt"),
    ]
    s.rules = [Rule(2.9, -1.75, 2.9, 5.35)]
    s.texts = [
        Text(0.0, -1.35, r"COORDINATED CA1--PFC RIPPLES", "panel"),
        Text(dx, -1.35, r"INDEPENDENT PFC RIPPLES", "panel"),
        Text(0.0, 4.35, r"{\small$\kappa<0$}\\[2pt]reconstruct (transfer)"),
        Text(dx, 4.35, r"{\small$\kappa>0$}\\[2pt]cancel (prioritise)"),
        Text(2.9, 6.55, r"one interface, one sign: the regime is the sign of the drive"),
    ]
    s.legend = _terminal_legend(0.55, 5.85)
    s.meta = {"name": "regimes"}
    return s


# ---- d. bridge ---------------------------------------------------------------
def _variant_d() -> Scene:
    """The correspondence itself: circuit on the left, interface on the right."""
    s = Scene()
    mx, gap = 8.4, 3.8                         # model column, cortex-hippocampus gap
    s.boxes = [
        Box("pfc", 0.0, 0.0, "region", PFC_T, w=4.8, h=1.2),
        Box("hpc", 0.0, gap, "region", HPC_T, w=4.8, h=1.2),
        Box("xs", mx, 0.0, "value", r"$x_S$", w=1.25, h=1.25, font=""),
        Box("ets", mx, gap / 2, "err", r"$\varepsilon_{TS}$", w=1.35, h=0.85, font=""),
        Box("xt", mx, gap, "value", r"$x_T$", w=1.25, h=1.25, font="",
            loop=r"W_T", loop_side="below"),
    ]
    # A reciprocal pair drawn with `bend left` separates by direction of travel:
    # the descending edge bulges right (label anchored west), the ascending one left.
    s.links = [
        # biology
        Link("hpc", "pfc", "up", "excit", up_path(1.12), label=r"{\tiny replay}",
             lopt="anchor=west, xshift=2pt"),
        Link("pfc", "hpc", "down", "inhib", down_path(1.12),
             label=r"{\tiny suppression}", lopt="anchor=east, xshift=-2pt"),
        # correspondence
        Link("pfc", "xs", "corr", "", "out=0, in=180"),
        Link("hpc", "xt", "corr", "", "out=0, in=180"),
        # model interface
        Link("xs", "ets", "down", "inhib", "bend left=22", label=r"$x_S$",
             lopt="anchor=west, xshift=1pt"),
        Link("ets", "xs", "up", "excit", "bend left=22", label=r"$\pi_{TS}$",
             lopt="anchor=east, xshift=-1pt"),
        Link("xt", "ets", "up", "excit", "bend left=22", label=r"$x_T$",
             lopt="anchor=east, xshift=-1pt"),
        Link("ets", "xt", "down", "excit", "bend left=22", label=r"$\kappa>0$",
             lopt="anchor=west, xshift=1pt"),
    ]
    s.rules = [Rule(4.75, -1.45, 4.75, 5.25)]
    s.texts = [
        Text(0.0, -1.2, "CIRCUIT", "panel"),
        Text(mx, -1.2, "INTERFACE", "panel"),
        Text(4.2, 6.15,
             r"the receiver's reconstruction enters the interface negatively:"
             r"\\[1pt]shared content cancels, the residual survives"
             r" and is replayed"),
    ]
    s.legend = _terminal_legend(1.0, 5.35)
    s.meta = {"name": "bridge"}
    return s


_BUILDERS = {"a": _variant_a, "b": _variant_b, "c": _variant_c, "d": _variant_d}


def build_scene(variant: str = "a") -> Scene:
    """Return the Scene for `variant` in {'a','b','c','d'} (or its long name)."""
    key = variant.strip().lower()
    if key not in _BUILDERS:                       # accept the long names too
        key = next((k for k, v in TITLES.items() if v == key), key)
    if key not in _BUILDERS:
        raise ValueError(f"variant must be one of {VARIANTS} or {tuple(TITLES.values())}")
    scene = _BUILDERS[key]()
    scene.meta["variant"] = key
    return scene


if __name__ == "__main__":
    for v in VARIANTS:
        sc = build_scene(v)
        print(f"{v} ({TITLES[v]}): {len(sc.boxes)} boxes, {len(sc.links)} links, "
              f"{len(sc.texts)} texts, {len(sc.rules)} rules, {len(sc.legend)} legend")
