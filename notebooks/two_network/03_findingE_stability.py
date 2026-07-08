# %% [markdown]
# # The stability guard (don't chase noise)
#
# In sleep the teacher's reversed precision drives it toward whatever the student can't predict.
# Noise is also unpredictable, so a *strong enough* reversed drive could push the teacher into
# noise — unless the teacher's self-damping wins. This is a **magnitude** question (how strong is
# the reversed precision `|π_ST|`), distinct from the *sign* question of the saddle (notebook 02).
#
# Off the memory manifold the damping is `≥ π_T·σ²_min` (spectral gap) and the drive is
# `|π_ST|·n(1)` with `n(1)=π_S/(π_TS+π_S)`. Stability requires `|π_ST|·n(1) < π_T·σ²_min`, i.e.
# **`|π_ST| < π_T·σ²_min·(π_TS+π_S)/π_S`** (the scalar guard). `|π_ST| < π_T·σ²_min` is the
# always-safe default.
#
# **Two tests.**
# 1. *Spectral:* with the student fully consolidated (`W_S=W_T`), the max eigenvalue of the
#    reduced growth operator `G(|π_ST|) = −π_T S_T + |π_ST| N_S` lifts off 0 exactly at the guard.
# 2. *Dynamical:* run a full replay transfer at each `|π_ST|` and measure the terminal fraction of
#    `x_T` inside the manifold. Below the guard the teacher stays on-manifold (quiescent); above it,
#    once the manifold is learned, the teacher is driven off into noise → occupancy collapses.

# %% setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pmt import ModelConfig, build_system
from pmt import experiments as ex

sns.set_theme(context="notebook", style="whitegrid")

base = ModelConfig(d=32, P=4, pi_TS=1.0, pi_S=0.5, seed=2)
_, info = build_system(base)
gap = info["sigma2_min"]; guard = info["guard_scalar"]
print(f"σ²_min (safe default) = {gap:.3f}   scalar guard |π_ST|* = π_T·σ²_min(π_TS+π_S)/π_S = {guard:.3f}")

# %% test 1 — spectral liftoff (sweep the reversed-precision magnitude)
ks_spec = np.linspace(0.5, 6.0, 60)
k1, gmax, _ = ex.offmanifold_growth(base, ks_spec)
onset = k1[np.argmax(gmax > 1e-9)]
print(f"spectral liftoff at |π_ST| ≈ {onset:.2f}  (predicted |π_ST|* = {guard:.2f})")

# %% test 2 — dynamical terminal occupancy
ks_dyn = np.linspace(1.0, 6.0, 16)
k2, occ, nov = ex.terminal_occupancy(base, ks_dyn, n_steps=22000)

# %% figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

ax[0].plot(k1, gmax, c="C0", lw=2)
ax[0].axhline(0, c="k", lw=0.8)
ax[0].axvline(guard, ls="--", c="C3", label=f"scalar guard |π_ST|*={guard:.2f}")
ax[0].axvline(gap, ls=":", c="C2", label=f"safe default σ²_min={gap:.2f}")
ax[0].set(title="Spectral: max growth of G=−π_T S_T+|π_ST| N_S (off-manifold)",
          xlabel="|π_ST|  (reversed-precision magnitude)", ylabel="max eigenvalue of G")
ax[0].annotate("noise amplified\n(g > 0)", xy=(guard + 0.7, 0.1), fontsize=9, color="C3")
ax[0].legend(loc="upper left")

ax[1].plot(k2, occ, "o-", c="C0", lw=1.8, label="terminal manifold occupancy")
ax[1].axvline(guard, ls="--", c="C3", label=f"scalar guard |π_ST|*={guard:.2f}")
ax[1].axvline(gap, ls=":", c="C2", label=f"safe default σ²_min={gap:.2f}")
ax[1].set(title="Dynamical: terminal fraction of x_T on the manifold",
          xlabel="|π_ST|", ylabel="manifold occupancy", ylim=(-0.03, 1.05))
ax[1].annotate("T chases noise →\noccupancy collapses", xy=(guard + 0.5, 0.4), fontsize=9, color="C3")
ax[1].legend(loc="lower left")
fig.suptitle("Stability boundary at |π_ST| = π_T·σ²_min·(π_TS+π_S)/π_S", y=1.02, fontsize=13)
fig.tight_layout()
fig

# %% verdict
drop_k = k2[np.argmax(occ < 0.5)] if np.any(occ < 0.5) else float("nan")
ok = abs(onset - guard) < 0.15 and abs(drop_k - guard) < 0.8
print(f"\n[{'CONFIRMED' if ok else 'CHECK'}] spectral liftoff at |π_ST|≈{onset:.2f} and the dynamical "
      f"occupancy collapse near |π_ST|≈{drop_k:.2f} both sit at the predicted guard |π_ST|*={guard:.2f}. "
      f"Below it the teacher consolidates and stays quiescent on-manifold; above it it chases noise.")
