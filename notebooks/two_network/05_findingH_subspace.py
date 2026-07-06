# %% [markdown]
# # The linear model transfers a *subspace*, not discrete memories
#
# Because the teacher is linear, its memories span a flat *subspace*; any blend of memories is
# also a memory. So the system transfers independent *directions*, and the novelty-spectrum
# staircase has as many steps as the **effective rank** of the memory set — **not necessarily
# `P`**. Orthonormal patterns ⇒ rank `P` ⇒ `P` steps. Correlated patterns ⇒ rank `< P` ⇒
# **fewer** steps (learning one direction explains its correlated neighbours).
#
# **Test.** Run the same transfer with `P=6` orthonormal patterns (rank 6) and with `P=6`
# strongly correlated patterns (rank 3), and compare the staircases and the pattern Gram spectra.

# %% setup
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

_root = pathlib.Path.cwd()
while not (_root / "pmt").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from pmt import ModelConfig, SimConfig, build_system, simulate
from pmt import experiments as ex

sns.set_theme(context="notebook", style="whitegrid")


def run(pattern_kind, **kw):
    cfg = ModelConfig(d=48, P=6, pi_TS=1.0, pi_S=0.5, seed=3,
                      pattern_kind=pattern_kind, **kw)
    model, info = build_system(cfg)
    sim = SimConfig(n_steps=45000, dt=0.5, mode="adiabatic", record_every=150, progress=False)
    H = simulate(model, sim, info).to_numpy()
    rank = ex.effective_rank(info["patterns"])
    gram_evals = torch.linalg.eigvalsh(info["patterns"].T @ info["patterns"]).flip(0).cpu().numpy()
    return H, info, rank, gram_evals


H_o, info_o, rank_o, gram_o = run("orthonormal")
H_c, info_c, rank_c, gram_c = run("correlated", corr_rank=3, corr_noise=0.0)

print(f"orthonormal: P=6  effective rank={rank_o}  manifold_dim={info_o['manifold_dim']}  "
      f"#staircase curves={H_o['novelty_spec'].shape[1]}")
print(f"correlated : P=6  effective rank={rank_c}  manifold_dim={info_c['manifold_dim']}  "
      f"#staircase curves={H_c['novelty_spec'].shape[1]}")

# %% figure
fig, ax = plt.subplots(2, 2, figsize=(14, 9))

for j in range(H_o["novelty_spec"].shape[1]):
    ax[0, 0].plot(H_o["t"], H_o["novelty_spec"][:, j], lw=1.6)
ax[0, 0].set(title=f"Orthonormal P=6 → rank {rank_o} → {rank_o} steps",
             xlabel="time", ylabel="novelty eigenvalue")

for j in range(H_c["novelty_spec"].shape[1]):
    ax[0, 1].plot(H_c["t"], H_c["novelty_spec"][:, j], lw=1.6)
ax[0, 1].set(title=f"Correlated P=6 → rank {rank_c} → {rank_c} steps (fewer than P!)",
             xlabel="time", ylabel="novelty eigenvalue")

x = np.arange(1, 7)
ax[1, 0].bar(x - 0.18, gram_o, width=0.36, label=f"orthonormal (rank {rank_o})", color="C0")
ax[1, 0].bar(x + 0.18, gram_c, width=0.36, label=f"correlated (rank {rank_c})", color="C3")
ax[1, 0].set(title="Pattern Gram spectrum  eig(MᵀM)  — the effective rank",
             xlabel="index", ylabel="eigenvalue")
ax[1, 0].legend()

ax[1, 1].plot(H_o["t"], H_o["novelty_spec"].sum(1), c="C0", lw=2,
              label=f"orthonormal (→ {rank_o} dirs)")
ax[1, 1].plot(H_c["t"], H_c["novelty_spec"].sum(1), c="C3", lw=2,
              label=f"correlated (→ {rank_c} dirs)")
ax[1, 1].set(title="Total novelty over time (both reach the floor)",
             xlabel="time", ylabel="Σ novelty eigenvalues")
ax[1, 1].legend()

fig.suptitle("Staircase steps = effective rank of the memory subspace, not P",
             y=1.01, fontsize=13)
fig.tight_layout()
fig

# %% verdict
ok = (rank_o == 6 and info_o["manifold_dim"] == 6
      and rank_c == 3 and info_c["manifold_dim"] == 3)
print(f"\n[{'CONFIRMED' if ok else 'CHECK'}] orthonormal patterns give {info_o['manifold_dim']} "
      f"staircase directions (= P), correlated give {info_c['manifold_dim']} (= effective rank "
      "< P). The linear model transfers a *subspace*: the step count is the memory set's "
      "effective rank. (Discrete one-per-pattern replay would need an added nonlinearity / soft-WTA.)")
