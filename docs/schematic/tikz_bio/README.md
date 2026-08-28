# PFC → hippocampus schematic (TikZ)

The biological counterpart of the interface: the **receiver suppresses the source it
also learns from**. Four propositions of the same figure, from bare to explicit, so the
paper can pick the granularity it needs. All four are minimal by construction: no brain
outlines, no layers, no cell morphology. They are a *computational* figure that sets up
the analogy, not an anatomy plate.

| Variant | File stem | What it shows | Use it for |
|---|---|---|---|
| **a** | `pfc_hpc_a_minimal` | Two regions, one arrow each way. Ascending excitatory, descending inhibitory. | An inset next to the model figure; the discussion opener. |
| **b** | `pfc_hpc_b_pathways` | The same, with the **two descending routes** drawn: direct long-range GABAergic, and the thalamic relay that recruits local feedforward inhibition. | Wherever the reader may object "there is no inhibitory projection from PFC to CA1". |
| **c** | `pfc_hpc_c_regimes` | Coordinated CA1–PFC ripples vs independent PFC ripples: the same circuit with the teacher-side drive flipped, `κ<0` vs `κ>0`. | The descent-versus-ascent claim. Pairs with the sleep/wake panels of the results. |
| **d** | `pfc_hpc_d_bridge` | Circuit on the left, model interface on the right, dotted correspondences between them. | The discussion figure that states the mapping outright. |

## Render

```bash
python render.py                 # -> pfc_hpc_{a,b,c,d}_*.pdf + .svg
```

```bash
python render.py --variant b --no-svg
```

Runs in the `pytorch` conda env (needs only `jinja2`). Requires a LaTeX toolchain with
`tikz` + `standalone`; `dvisvgm` does the PDF→SVG step. Same pipeline as `../tikz/`.

## Layout

| File | Role |
|---|---|
| [`../spec_bio.py`](../spec_bio.py) | **The four figures, declared once**: boxes, links, captions, legend, colours |
| [`pfc_hpc.tex.j2`](pfc_hpc.tex.j2) | Jinja2 → TikZ template (styles only; it does not know which variant it draws) |
| [`render.py`](render.py) | Fills the template, compiles, converts to SVG |
| `pfc_hpc_*.tex` / `.pdf` / `.svg` | Generated — **don't hand-edit**, re-run `render.py` |

## Conventions

- **Terminal = sign**: arrowhead excitatory, filled dot inhibitory (as in `../tikz_units/`,
  after Tang et al. 2023).
- **Colour = direction**: blue descending (cortex → hippocampus), orange ascending
  (hippocampus → cortex). Since the receiver is drawn on top, this coincides with the
  prediction/error colours of `../spec.py`, and `spec_bio.COLORS` imports that palette.
- **Hierarchy**: cortex (student, `x_S`, receiver) on top, hippocampus (teacher, `x_T`,
  source) below. All model and biological diagrams use this order.
- Geometry is in centimetres in `spec_bio.py` with **+y down**; `render.py` flips y.
  `Link.path` is a raw TikZ `to[...]` string, and `Link.dst` may be any TikZ coordinate
  expression (e.g. `hpc.north -| inn`), not just a node name.
- Gotcha: a size command (`\tiny`) placed *before* a `\\` breaks a node's `align=center`
  cell. Line-broken labels set their size through `font=` in `lopt` instead.

## What the biology is, and what is deliberately left out

Drawn:

- **Ascending, direct, excitatory.** CA1 → PFC is monosynaptic (ventral CA1) and drives
  PFC reactivation; SWR-coordinated replay is the transfer channel, and cortical engram
  maturation is its consequence (Kitamura et al. 2017).
- **Descending, indirect, net inhibitory.** Two routes:
  1. **Monosynaptic long-range GABAergic** mPFC → hippocampus. These projections
     preferentially inhibit VIP interneurons (themselves disinhibitory), which *increases*
     hippocampal feedforward inhibition and reduces hippocampal activity in vivo
     (Malik, Li, Schamiloglu & Sohal 2022, *Cell* 185:1602). The figure draws the net
     sign; the double negative is one level of detail below what it is for.
  2. **Via nucleus reuniens** (midline thalamus): mPFC → RE → CA1 *stratum
     lacunosum-moleculare*, glutamatergic, recruiting neurogliaform and
     interneuron-selective cells, so CA1 principal cells see monosynaptic excitation
     plus **polysynaptic inhibition** (Dolleman-van der Weel et al.; Frontiers Cell.
     Neurosci. 2021, 15:660897).
- **The phenomenon itself.** During NREM, independent PFC ripples suppress CA1 activity
  broadly (pyramidal cells *and* interneurons) and suppress reactivation of recent
  experience; suppression peaks before CA1 SWRs (Shin & Jadhav 2024, *Curr. Biol.*
  34:2801, in `litterature/04_empirical_constraints/`).

Left out on purpose: entorhinal cortex (a genuine descending route, but excitatory, so it
would misread next to inhibitory terminals), lateral septum, medial septum, dorsal/ventral
CA1 distinctions, laminar targets, cell types beyond a single interneuron disc, and the
slow-oscillation / spindle scaffold.

## Why this matters to the model

The suppression lands on CA1 **principal** cells, not on error units. The interface error is
`ε_TS = x_T - x_S`: the source enters positively and the receiver's reconstruction enters
negatively. During replay, the positive gain `κ > 0` drives the teacher along this residual.
With recurrent weights in the source population, that residual steers replay toward the memories
the receiver cannot yet reconstruct.
