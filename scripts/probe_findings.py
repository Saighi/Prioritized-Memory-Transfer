import numpy as np
import torch
from src import ModelConfig, build_system
from src import experiments as ex

base = ModelConfig(d=32, P=4, seed=2)
_, info = build_system(base)
print(f"sigma2_min={info['sigma2_min']:.3f}  guard_scalar={info['guard_scalar']:.3f}  pi_TS=1")

# saddle exactness: circulation vs |pi_ST + pi_TS| sqrt(d) (signed pi_ST; saddle at pi_ST=-pi_TS)
ks = np.linspace(-2.0, 0.0, 9)
k, emp, theo = ex.sweep_circulation(base, ks)
print("\n[saddle] max|emp-theo| =", float(np.max(np.abs(emp - theo))), "(want ~0)")

# stability spectral: growth operator -pi_T S_T + |pi_ST| N_S
ks2 = np.linspace(0.5, 6.0, 12)
k2, gmax, _ = ex.offmanifold_growth(base, ks2)
onset = k2[np.argmax(gmax > 1e-6)]
print(f"[stability] spectral liftoff near |pi_ST|={onset:.2f}  (guard_scalar={info['guard_scalar']:.2f})")

# stability dynamical
ks3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
k3, occ, nov = ex.terminal_occupancy(base, ks3, n_steps=20000)
print("[stability] |pi_ST|:", k3)
print("    terminal occupancy:", np.round(occ, 3), " (want ~1 below guard, drop above)")

# timescales
ratios = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
r, rk, rn = ex.timescale_discrimination(base, ratios, known_idx=[0, 1], novel_idx=[2, 3])
print("[timescale] ratio:", ratios)
print("    known residual:", np.round(rk, 3), "(rises as r->1)")
print("    novel residual:", np.round(rn, 3), "(~constant)")

# subspace rank
Mo = build_system(ModelConfig(d=32, P=6, pattern_kind="orthonormal", seed=3))[1]["patterns"]
Mc = build_system(ModelConfig(d=32, P=6, pattern_kind="correlated", corr_rank=3, corr_noise=0.0, seed=3))[1]["patterns"]
print(f"[subspace] eff_rank orthonormal(P=6)={ex.effective_rank(Mo)}  correlated(rank3)={ex.effective_rank(Mc)}")
io = build_system(ModelConfig(d=32, P=6, pattern_kind="orthonormal", seed=3))[1]
ic = build_system(ModelConfig(d=32, P=6, pattern_kind="correlated", corr_rank=3, corr_noise=0.0, seed=3))[1]
print(f"    manifold_dim orthonormal={io['manifold_dim']}  correlated={ic['manifold_dim']}  "
      f"(resid {io['memory_residual']:.1e}/{ic['memory_residual']:.1e})")
print("PROBE DONE")
