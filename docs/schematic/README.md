# Network schematic (TikZ)

A "paper-minimal" wiring diagram of the two-population predictive-coding network
(teacher **T** above student **S**; value neurons, the three error populations, the
recurrent weights, and the labelled prediction/drive couplings).

Blue = top-down prediction, orange = bottom-up / drive. `ε_TS = x_S − x_T`.

Two granularities, same pipeline:

- **`tikz/`** — population level: one node per population/error (the compact paper figure).
- **`tikz_units/`** — unit level, **2 units per population** (Tang et al. 2023 Fig. 1 style):
  shows the strictly **one-to-one interface** (`x_T,i ↔ ε_TS,i ↔ x_S,i`, no crossing) and
  the **lateral cross-communication inside each population** (zero-diagonal recurrent
  weights ⇒ units talk only through the crossing `W_{12}`/`W_{21}` reciprocal pairs).
  Adds Tang's sign convention: arrowhead = excitatory, dot = inhibitory — so the signed
  interface drive is visible: sleep `π_ST<0` terminates in a dot, wake `π_ST>0` in an arrow.

## Render

```bash
python tikz/render.py                 # -> tikz/network.pdf + tikz/network.svg
python tikz/render.py --phase wake    # the recall/inference control
python tikz/render.py --no-svg        # skip the dvisvgm step

python tikz_units/render.py           # -> tikz_units/network_units.pdf + .svg (same flags)
```

Runs in the `pytorch` conda env (needs only `jinja2`). Requires a LaTeX toolchain with
`tikz` + `standalone` (MiKTeX/TeX Live); `dvisvgm` — which ships with both — does the
PDF→SVG step.

## Layout

| File | Role |
|---|---|
| [`spec.py`](spec.py) | **The figure, declared once**: nodes, coordinates, edges, labels, colours |
| [`tikz/network.tex.j2`](tikz/network.tex.j2) | Jinja2 → TikZ template (styles, node/edge drawing) |
| [`tikz/render.py`](tikz/render.py) | Fills the template, compiles, converts to SVG |
| `tikz/network.tex` / `.pdf` / `.svg` | Generated — **don't hand-edit**, re-run `render.py` |
| [`spec_units.py`](spec_units.py) | The **unit-level** figure (2 units/population): per-unit nodes, signed edges |
| [`tikz_units/network_units.tex.j2`](tikz_units/network_units.tex.j2) | Unit-level template: adds excitatory/inhibitory terminals + legend |
| [`tikz_units/render.py`](tikz_units/render.py) | Same render pipeline for the unit-level figure |
| `tikz_units/network_units.tex` / `.pdf` / `.svg` | Generated — **don't hand-edit** |

## Parametrization

Everything is driven from `spec.py`:

- **Phase / drive sign.** The teacher's interface drive is the **signed** precision `π_ST`.
  `build_spec(phase="sleep")` shows it as a *reversed* (negative) precision `π_ST<0`
  (drive-to-disagree); `phase="wake"` shows the ordinary `π_ST>0` (recall). The sign *is*
  the phase — no separate gate.
- **Numeric gains.** Pass a `ModelConfig`-like object as `values=` to fold values into the
  labels (e.g. `π_ST=-0.21`); the paper-minimal default keeps them symbolic. A resolved
  `π_ST` comes from `model.build_system(cfg)`, which turns `"auto"` into a number.
- **Colours** live in `spec.COLORS` (bare HTML hex) and are injected into the template's
  `\definecolor` lines, so there's one place to change them.
- **Geometry.** Node coordinates are in `spec.py` in grid units (+y is *down*);
  `render.py` flips y and scales via `SX`/`SY`, and `BEND` sets the arrow curvature that
  separates each reciprocal pair.

## Why TikZ

The topology here is **fixed** (always T, S, three error populations) and the layout is
already known, so explicit placement beats auto-layout. Tools like D2, Graphviz, Mermaid
and Penrose deliberately offer no "put node X at (x, y)" — you describe structure and an
engine picks the geometry, which for a small *cyclic* circuit (every coupling is a
reciprocal pair, plus two self-loops) means the engine breaks cycles by reversing edges
and routing them the long way round. TikZ gives coordinate-level control, full LaTeX math
in labels, and is how PC papers (Bogacz, Friston, Tang) draw these figures.
