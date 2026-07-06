# %% [markdown]
# # Interleaved merge — correlation sweep & multi-memory teachers
#
# Run cell-by-cell (**`pytorch`** kernel). Companion to `01_interleaved_single_run.py`. How the
# crosstalk sawtooth and the convergence depend on how correlated the two memories are, and a check
# that interleaving also works for multi-memory teachers.

# %% imports, path setup, run helper
import os
import sys
import pathlib

import numpy as np
import torch

_root = pathlib.Path.cwd()
while not (_root / "pmt").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from pmt import AdditiveSynthesisConfig, SimConfig, build_interleaved_synthesis, simulate_interleaved

SHOW = os.environ.get("PMT_NO_SHOW") != "1"
torch.manual_seed(0)


def run(n_steps=90000, bout_steps=3000, **cfg_kw):
    cfg = AdditiveSynthesisConfig(**{**dict(d=32, pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9,
                                            eta=0.05, sigma_xi1=0.1, sigma_xi2=0.1, seed=0), **cfg_kw})
    macro, info = build_interleaved_synthesis(cfg)
    H = simulate_interleaved(macro, SimConfig(n_steps=n_steps, dt=0.5, mode="adiabatic",
                                              bout_steps=bout_steps, progress=False), info).to_numpy()
    return H, info


def crosstalk_bump(H):
    """Size of the inactive teacher's residual over the SECOND HALF of the run — after both memories
    are learned, how much rehearsing one disturbs the other (the crosstalk)."""
    r1, r2, act = H["resid1"], H["resid2"], H["active"]
    inactive = np.where(act == 0, r2, r1)      # when T1 active, the disturbed one is r2, and vice versa
    return float(inactive[len(inactive) // 2:].mean())


# %% [markdown]
# ## Correlation sweep: orthogonal → oblique → near-parallel
# Two single memories at increasing correlation. Orthogonal → no crosstalk (flat). Oblique →
# a decaying sawtooth. Near-parallel → strong, slow-to-cancel crosstalk (ill-conditioned).

# %%
for name, kw in [("orthogonal", dict(geometry="orthogonal", rank1=1, rank2=1)),
                 ("oblique θ=0.5", dict(geometry="oblique", rank1=1, rank2=1, principal_angle=0.5)),
                 ("near-parallel θ=0.2", dict(geometry="oblique", rank1=1, rank2=1, principal_angle=0.2))]:
    H, info = run(**kw)
    print(f"{name:20s}  r_Σ={info['r_Sigma']}  union deficit -> {H['union_deficit'][-1]:.4f}  "
          f"crosstalk bump ~ {crosstalk_bump(H):.3f}")
print("→ more correlation = larger, slower-to-cancel crosstalk, but interleaving still builds 𝒰_Σ")

# %% [markdown]
# ## Multi-memory teachers
# When each teacher has ≥2 memories the additive *sum* also works — but so does interleaving. It is
# the general mechanism; the single-memory case is just where it is *necessary*.

# %%
Hm, infom = run(geometry="shared", rank1=2, rank2=2, overlap=1, n_steps=140000, bout_steps=3500)
print(f"multi-memory (2+2, overlap 1): r_Σ={infom['r_Sigma']}  "
      f"‖M_S U1‖ -> {Hm['resid1'][-1]:.3f}  ‖M_S U2‖ -> {Hm['resid2'][-1]:.3f}  "
      f"union deficit -> {Hm['union_deficit'][-1]:.4f}")
assert Hm["union_deficit"][-1] < 0.3, "interleaving should merge multi-memory teachers too"

# %% [markdown]
# ## Summary
# Interleaving builds the combined subspace across the whole correlation range — flat for orthogonal
# memories, a pronounced decaying sawtooth for correlated ones (the back-and-forth cancelling
# crosstalk), slower but still convergent as memories approach parallel — and works for single- and
# multi-memory teachers alike. Unlike the additive sum, it needs no teacher to roam.
