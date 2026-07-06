# Penrose backend (starter)

Penrose (https://penrose.cmu.edu) is a *constraint-based* diagramming system: you
declare a vocabulary (`pc.domain`), the specific objects (`pc.substance`), and how to
draw them (`pc.style`), and an optimizer lays it out. It makes beautiful math figures
but is the highest-effort option here, and — unlike the other three — it could **not** be
auto-rendered in this environment (it needs a Node toolchain). Treat these three files as
a verified-in-editor starting point.

## Render it

**Web editor (no install):** open https://penrose.cmu.edu/try and paste the three files
into the `.domain` / `.substance` / `.style` panes.

**Local CLI (Node ≥ 18):**
```bash
npm install -g @penrose/roger
roger trio pc.substance pc.style pc.domain --out network.svg
```

## What's done vs. what's left

- **Done:** value neurons (circles) and the three error populations (dashed boxes),
  blue prediction / orange drive arrows, recurrent-loop markers, and a top→middle→bottom
  layout driven by the `Above` / `LeftOf` relations.
- **Left (the manual effort Penrose costs you):**
  1. **Edge gain-labels** (`π_T`, `x_T`, `π_ST<0`, `π_TS`, `M_S`, …). Penrose has no
     per-edge label out of the box; you attach an `Equation` at each arrow's midpoint,
     which means carrying the gain string on the edge (e.g. encode it via extra
     predicates `PredictsL(a, b, lbl)` or a label map generated from `spec.py`).
  2. **Parallel-pair separation:** the two arrows of each pair (e.g. `x_T↔ε_TS`) currently
     overlap as straight lines; swap `Line` for a `Path`/Bézier with opposing curvature,
     as the schemdraw/TikZ backends already do.

## Parametrizing from `spec.py`

`pc.substance` is the file to generate. A ~30-line emitter that walks
`spec.build_spec()["edges"]` and prints `Predicts(...)` / `Errors(...)` / `Label` lines
gives you the same single-source-of-truth parametrization the other backends have.
