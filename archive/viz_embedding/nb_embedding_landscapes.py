# %% [markdown]
# # SMACOF-reconstructed VFE landscapes for prioritized memory transfer
#
# This notebook borrows a trick from a Hopfield energy-landscape project — **SMACOF-embed a cloud
# of states into a reconstructed (non-isometric, intuitive) 2-D layout, then drape a scalar height
# over it** — and applies it to *this* project's variational free energy. The network logic lives in
# `src`; everything visual lives in the standalone `viz_embedding/` folder (`embedding.py` = pure
# SMACOF/drape, `figures.py` = the six builders). Run cell-by-cell in VS Code with the **`pytorch`**
# kernel, or headless as a script.
#
# **Honest caveat (in every caption):** the linear model's VFE is a *smooth quadratic form with no
# local basins on the manifold* — the manifold is exactly flat. A faithful render is **one flat
# plateau**, not Hopfield-style per-memory valleys. The interest is the surface **morphing over
# training** (novel-direction valleys filling in) and the **sleep-vs-wake sign flip** — not static
# basins. That is the correct picture of "T is flat / transfers a *subspace*."
#
# Prereq (one-time): `conda run -n pytorch pip install scikit-learn scipy`.

# %% imports & path setup
import os
import pathlib

import numpy as np
import torch

# make `import src` and `import viz_embedding` work whatever the cwd / nesting depth

from src import ModelConfig, SimConfig, build_system, simulate
from src import diagnostics as dg
from viz_embedding import embedding as emb
from viz_embedding import figures as F

SHOW = os.environ.get("PMT_NO_SHOW") != "1"           # set PMT_NO_SHOW=1 to run headless
EXPORT = True                                          # write a self-contained .html per figure
HTML_DIR = pathlib.Path(__file__).resolve().parent / "html" if "__file__" in globals() \
    else _root / "viz_embedding" / "html"
HTML_DIR.mkdir(exist_ok=True)
torch.manual_seed(0)
print("torch", torch.__version__, "| html ->", HTML_DIR)


def emit(fig, name, n_msg=""):
    """Inline-show (unless headless) and export a standalone HTML."""
    nf = f", {len(fig.frames)} frames" if getattr(fig, "frames", None) else ""
    print(f"  {name}: {len(fig.data)} traces{nf}  {n_msg}")
    if EXPORT:
        fig.write_html(str(HTML_DIR / f"{name}.html"), include_plotlyjs="cdn", auto_play=False)
    if SHOW:
        fig.show()


# %% configure & build the SLEEP system (π_ST < 0, the transfer regime)
# exact_saddle=True sets π_ST = −π_TS, so Φ = F_T − F_S is an exact saddle potential (the caption
# in figures 3–5 is then literally true, not just structural).
cfg_sleep = ModelConfig(
    d=48, P=6,
    pi_TS=1.0, pi_S=0.5,
    exact_saddle=True,                 # π_ST = −π_TS
    tau_T=10.0, tau_S=1.0, eta=0.02,
    sigma_xi=0.05, r0=1.0,
    pattern_kind="orthonormal", W_T_kind="covpcn",
    seed=0, device="cpu",
)
model, info = build_system(cfg_sleep)
print(f"spectral gap σ²_min    : {info['sigma2_min']:.4f}")
print(f"manifold dim (eff rank): {info['manifold_dim']}   (memory subspace)")
print(f"π_ST                   : {info['pi_ST']:.4f}  (reversed / <0 ⇒ sleep-replay)")
print(f"guard |π_ST|<σ²_min?   : {abs(info['pi_ST']) < info['guard_safe']}")
e_xS, e_WS = dg.check_gradients(model)
print(f"VFE gradient check     : err_xS={e_xS:.1e}  err_WS={e_WS:.1e}")

# %% run the SLEEP transfer (adiabatic fast-S; many weight snapshots for smooth animations)
n_steps = 20000
sim_sleep = SimConfig(n_steps=n_steps, dt=0.5, mode="adiabatic",
                      record_every=max(1, n_steps // 200),
                      n_weight_snapshots=30, progress=True)
hist = simulate(model, sim_sleep, info)
Hn = hist.to_numpy()
print(f"novelty top eig: {Hn['novelty_spec'][0].max():.3f} → {Hn['novelty_spec'][-1].max():.3f}"
      "   (→0 ⇒ subspace transferred)")

# %% build + run a WAKE control (π_ST > 0): same teacher/seed, no transfer drive
cfg_wake = ModelConfig(**{**cfg_sleep.__dict__, "exact_saddle": False, "pi_ST": +cfg_sleep.pi_TS})
model_wake, info_wake = build_system(cfg_wake)
hist_wake = simulate(model_wake, sim_sleep, info_wake)
print(f"wake π_ST = {info_wake['pi_ST']:+.3f}  |  "
      f"wake novelty top eig: {hist_wake.to_numpy()['novelty_spec'][-1].max():.3f}  (stays high ⇒ no transfer)")

# %% shared state cloud + ONE SMACOF layout (the fixed-layout rule: reuse across all figures)
X, layout = F.make_layout(model, info, n=800, seed=0)
print(f"sampled {X.shape[0]} states (d={X.shape[1]}); SMACOF layout {layout.shape}")

# %% Fig 1 — teacher flat plateau (F_T)
emit(F.teacher_plateau(model, info, layout=layout, X=X), "01_teacher_plateau")

# %% Fig 2 — student carving its manifold (F_S^self over training)
emit(F.student_carving_animated(hist, model, info, layout=layout, X=X), "02_student_carving")

# %% Fig 3 — saddle cross-sections (bowl vs hill; hill flattens with learning)
emit(F.saddle_cross_sections(hist, model, info), "03_saddle_cross_sections")

# %% Fig 4 — deflating novelty (Φ over training): the primary result as a landscape
emit(F.deflating_novelty_animated(hist, model, info, layout=layout, X=X), "04_deflating_novelty")

# %% Fig 5 — sleep vs wake (the reversed-precision sign flip)
emit(F.sleep_vs_wake(hist, hist_wake, model, info, layout=layout, X=X), "05_sleep_vs_wake")

# %% Fig 6 — flow field: T's sleep flow = steepest descent on Φ (S empty)
emit(F.flow_field(model, info, layout=layout, X=X), "06_flow_field")

# %% [markdown]
# ## What to read off these
# - **Fig 1** — the memory manifold is a flat floor; off-manifold rises into walls. "T is flat."
# - **Fig 2 / Fig 4** — complementary carvings over training: F_S^self deepens valleys along the
#   directions S *has learned*; Φ shows the *novel*-direction valleys (where T is driven in replay)
#   **filling in** as transfer completes — the novelty-spectrum staircase rendered as a landscape.
# - **Fig 3 / Fig 6** — why the sleep flow is a saddle: a bowl along predictable directions, a hill
#   along novel ones; streamlines run downhill into the novel valleys (the replay drive).
# - **Fig 5** — flip the sign of π_ST and the same teacher either transfers (sleep, valleys fill) or
#   does not (wake, valleys persist). The reversed precision *is* the mechanism.
#
# Next: raise `n` (sample count) and `n_weight_snapshots` for smoother figures; swap the fixed
# Euclidean layout for the model-native `metric=N_S` in `make_layout` to watch the manifold collapse.
