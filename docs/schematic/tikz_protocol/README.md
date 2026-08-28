# Training-protocol schematic (TikZ)

A **state** figure, not a wiring one: the same recurrent-network glyph under two
conditions, so time reads left → right. It answers *when each component is active and
plastic*; the mechanism itself lives in [`../tikz_units/`](../tikz_units/).

| Panel | Shows |
|---|---|
| **(A) Acquisition** | Teacher clamped to interleaved patterns $x_T\leftarrow m^{(p)}$, $W_T$ plastic. Student and interface dimmed to 20% and labelled *off*. |
| **(B) Consolidation** | Clamp gone, $W_T$ frozen (lock, $\dot W_T=0$). Interface on; teacher and student co-evolve with $W_S$ plastic. Same spatial shade pattern, student a notch weaker so $x_S\neq x_T$ and $\varepsilon_{TS}$ is still alive. |

## Render

```bash
python render.py            # -> protocol_ab.pdf + protocol_ab.svg
python render.py --no-svg
```

Runs in the `pytorch` conda env (needs only `jinja2`); LaTeX with `tikz`+`standalone`,
`dvisvgm` for the SVG. Same pipeline as the sibling folders.

## Knobs — you almost always touch only [`../spec_protocol.py`](../spec_protocol.py)

| What | Where |
|---|---|
| Ring node positions (shared by both panels) | `_ring`, `R_RING`, `N_RING` |
| Which recurrent chords are drawn | `CHORDS` |
| Per-panel node shade (the activity pattern) | `_PATTERN`, `_STUDENT_DIM`, `ACTIVITY` |
| Which population is plastic / frozen / off, clamp on/off | `build_panel` |
| Panel spacing, ring centres | `PANEL_DX`, `CY_S`, `CY_T` |
| Colours, off-opacity | `COLORS`, `OPACITY_OFF` |

The template [`protocol.tex.j2`](protocol.tex.j2) hard-codes the bespoke chrome (clamp,
padlock, captions, legend); it does not know which panel it draws.

## Conventions (shared with the other schematics)

- **Colour = state**: blue plastic weights ($\dot W\neq0$), gray frozen ($\dot W=0$, lock).
- **Terminal = sign** on the coupling: teacher $\to$ student is excitatory with a blue
  arrowhead; student $\to$ teacher is inhibitory with an orange dot.
- **Hierarchy**: student ($x_S$) on top, teacher ($x_T$) below, as in `../spec.py`.
- Geometry is in cm with **+y down**; `render.py` flips y.
- Generated `protocol_ab.{tex,pdf,svg}` — **don't hand-edit**, re-run `render.py`.
