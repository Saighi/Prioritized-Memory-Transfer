# `viz_embedding/` — SMACOF-reconstructed VFE landscapes

A standalone visualization folder. It borrows a trick from a separate Hopfield project —
**SMACOF-embed a cloud of states into a reconstructed (non-isometric, intuitive) 2-D layout, then
drape a scalar height over it** — and applies it to this project's *variational free energy*. The
network logic is reused from `pmt`; nothing here is imported by `pmt`.

```
embedding.py                  pure SMACOF embed-and-drape toolkit (numpy/scipy/sklearn/plotly only;
                              NO torch, NO pmt). pairwise_dist · smacof_layout · drape_surface · ...
figures.py                    the six figure builders; calls pmt (model/info/hist) for S_T, N_S,
                              and the W_S weight snapshots, then hands heights to embedding.py
nb_embedding_landscapes.py    the one notebook: builds + runs the network via pmt, renders + exports
html/                         self-contained .html per figure (written at runtime)
```

## The scalars (fast-S limit, quadratic forms in x_T on the sphere ‖x_T‖=r0)

| field | meaning |
|---|---|
| `F_T  = ½ π_T xᵀ S_T x`   | teacher self-energy — **zero on the memory manifold**, walls off it |
| `F_S* = ½ π_TS xᵀ N_S x`  | novelty surface — 0 on learned directions, >0 on unlearned ones |
| `Φ    = F_T − F_S*`       | the saddle — flat memory floor, **valleys at novel directions**, off-manifold walls |

Heights depend on `S_T, N_S, π_T, π_TS` — **not** on the sign of `π_ST`. Only the simulated `W_S`
trajectory differs between sleep and wake; that is exactly what figure 5 contrasts.

## The figures

1. **Teacher plateau** — `F_T`, static. "T is flat."
2. **Student carving** — `F_S^self`, animated. Uniform on the sphere → valleys carve along learned dirs.
3. **Saddle cross-sections** — `Φ` along a line: bowl (predictable) vs hill (novel; flattens with learning).
4. **Deflating novelty** — `Φ`, animated. Novel-direction valleys fill in as S learns (the primary result).
5. **Sleep vs wake** — final `Φ` for π_ST<0 (transfer) vs π_ST>0 (no transfer), side by side.
6. **Flow field** — steepest-descent quiver on `Φ` = the teacher's replay drive.

**Honest caveat (in every caption):** the linear model's VFE is a smooth quadratic form with *no
local basins on the manifold* — a faithful render is one flat plateau, not Hopfield-style per-memory
valleys. The interest is the morphing over training and the sleep/wake sign flip, not static basins.

## Run

Prereq (one-time): `conda run -n pytorch pip install scikit-learn scipy`.

- **Interactive:** open `nb_embedding_landscapes.py` in VS Code, pick the `pytorch` kernel, run the
  `# %%` cells. Figures display inline; each is also written to `html/`.
- **Headless smoke test** (builds + exports all six, skips `.show()`):

  ```bash
  PYTHONIOENCODING=utf-8 PMT_NO_SHOW=1 MPLBACKEND=Agg PYTHONPATH=. \
    C:/Users/pauls/anaconda3/envs/pytorch/python.exe viz_embedding/nb_embedding_landscapes.py
  ```

  Pass = exit 0, six `NN_*.html` files under `html/`.

## Notes

- **Fixed-layout rule:** the notebook computes ONE SMACOF layout and reuses it for every figure and
  every animation frame, so the embedding never jitters — only the draped height changes.
- Animations use a 100² drape grid (≈13 MB HTML); statics use 160². Raise `n` (sample count) and
  `n_weight_snapshots` for smoother results.
- To watch the manifold *collapse* instead of a fixed layout, pass `metric=N_S` to
  `figures.make_layout` (re-embeds in the student's own novelty metric).
