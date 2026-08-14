# %% [markdown]
# # Transfer with an arbitrary number of memories
#
# Same system as `01_single_run_non_linear.py`, but `d` and `P` are plain knobs and **every figure
# scales with `P`**. The 3-D trajectory plot only ever worked at exactly `P = 3`; here the replay
# path is shown as a *raster of pattern coordinates* instead, which reads the same way at `P = 3`
# or `P = 40`.
#
# The coordinates are the true expansion coefficients `c = argmin‖x_T − Mc‖`, not inner products:
# with non-orthogonal patterns `x·m_p` is positive for almost any state and barely discriminates.
# `c_p < 0` means the teacher has drifted off the memory manifold, so the raster is **signed** and
# uses a diverging colour scale centred on zero.
#
# Only the constraint `P <= d-1` is enforced by `ModelConfig`; in practice the covPCN fit also
# needs each neuron to be predictable from the others, which gets harder as `P` approaches `d`.

# %% imports & setup
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from prioritized_memory_transfer import ModelConfig, SimConfig, build_system, simulate
from prioritized_memory_transfer.viz_style import mpl_style

SHOW = os.environ.get("PMT_NO_SHOW") != "1"
mpl_style()
torch.manual_seed(0)

# %% knobs — d = neurons, P = stored memories. n_steps is a plain number: scale it yourself.
ACTIVATION = "relu"          # "relu" | None (linear)
PATTERNS   = "nonneg"        # "nonneg" | "target_corr_nonneg" | "orthonormal" | "target_corr"
D, P_MEM   = 200, 6          # <-- neurons, stored memories


model_cfg = ModelConfig(
    d=D, P=P_MEM,
    pi_TS=1.0, pi_S=0.5,
    pi_ST=-0.5,                       # fixed precisions throughout, no "auto"
    tau_T=10.0, tau_S=1.0, eta=0.002,
    sigma_xi=0.05, r0=1.0,
    pattern_kind=PATTERNS,
    corr_target=0.3,                  # target_corr*: the pairwise correlation to hit
    activation=ACTIVATION,
    seed=0, device="cpu",
)
sim_cfg = SimConfig(n_steps=100000, dt=0.5, mode="full", record_every=100,
                    n_weight_snapshots=0, progress=True)

model, info = build_system(model_cfg)
ACT = info.activation or "linear"
P = model_cfg.P
print(f"activation {ACT} | d={model_cfg.d} P={P} | patterns_fixed={info.patterns_fixed} "
      f"| manifold dim {info.manifold_dim} | memory residual {info.memory_residual:.1e}")
assert info.memory_residual < 1e-4, "the pattern set is not representable — lower P or raise d"

hist = simulate(model, sim_cfg, info)
H = hist.to_numpy()
t = H["t"]
energy = 0.5 * model_cfg.pi_S * H["residual"] ** 2          # (T, P) = F_S at each memory
coef = np.linalg.lstsq(model.patterns.numpy(), H["x_T"].T, rcond=None)[0].T   # (T, P)
print(f"worst memory F_S: {energy[0].max():.4f} -> {energy[-1].max():.4f} | "
      f"||W_S-W_T||: {H['WS_dist'][0]:.3f} -> {H['WS_dist'][-1]:.3f}")

# %% [markdown]
# ## Consolidation across all P memories
#
# One faint line per memory plus the worst/median envelope, so the picture stays readable however
# large `P` gets. The right panel is the final `F_S(m_p)` sorted, which shows directly whether the
# transfer finished everywhere or left a tail of unconsolidated memories.

# %% figure 1 — consolidation
fig1, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for p in range(P):
    ax[0].plot(t, energy[:, p], lw=0.9, alpha=0.45, c="C0")
ax[0].plot(t, energy.max(1), lw=2.2, c="C3", label="worst memory")
ax[0].plot(t, np.median(energy, 1), lw=2.0, c="C1", label="median")
ax[0].set(title=f"$F_S(m_p)$ for all {P} memories ({ACT})", xlabel="time",
          ylabel=r"$F_S(m_p)$", yscale="log")
ax[0].legend()

order = np.argsort(energy[-1])
ax[1].bar(range(P), energy[-1][order], color="C0")
ax[1].set(title="Final $F_S(m_p)$, sorted", xlabel="memory (sorted)", ylabel=r"$F_S(m_p)$")
ax[1].set_xticks(range(P))
ax[1].set_xticklabels([str(p) for p in order], fontsize=7)
fig1.tight_layout()
fig1

# %% [markdown]
# ## Replay path in pattern coordinates
#
# This replaces the 3-D trajectory. Left: the signed coefficient raster — one row per memory, time
# across. Red/blue is `c_p > 0` / `c_p < 0`; a state sitting on memory `p` is a single bright row.
# Right: how much of the run each memory dominates (`|c_p|` is the largest coefficient), which says
# whether replay tours the memories or parks on a few.

# %% figure 2 — replay raster (works for any P)
vmax = float(np.abs(coef).max())
dominant = np.argmax(np.abs(coef), axis=1)
share = np.array([(dominant == p).mean() for p in range(P)])

fig2, ax = plt.subplots(1, 2, figsize=(14, 4.4),
                        gridspec_kw={"width_ratios": [3, 1]})
im = ax[0].imshow(coef.T, aspect="auto", origin="lower", cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, extent=[t[0], t[-1], -0.5, P - 0.5])
ax[0].set(title=r"Replay in memory coordinates $c_p(t)$", xlabel="time", ylabel="memory $p$")
ax[0].set_yticks(range(P))
ax[0].grid(False)
cb = fig2.colorbar(im, ax=ax[0], fraction=0.03, pad=0.02)
cb.set_label("$c_p$   (blue = off-manifold)")

ax[1].barh(range(P), share, color="C0")
ax[1].axvline(1.0 / P, ls="--", c="k", alpha=0.5, label="uniform tour")
ax[1].set(title="Fraction of time dominant", ylabel="memory $p$", xlabel="share")
ax[1].set_yticks(range(P))
ax[1].legend(fontsize=8)
fig2.tight_layout()
print(f"replay dominance: max {share.max():.2%} on memory {share.argmax()}, "
      f"{int((share > 0.02).sum())}/{P} memories dominant >2% of the time")
fig2

# %% [markdown]
# ## Recall, scaled to P memories
#
# Same clamped protocol as notebook 01: clamp a fraction of units to `m_p`, relax the student
# alone, measure `‖x∞ − m_p‖ / ‖m_p‖`. The starting point is `m_p` itself — the earlier sweep
# showed the landing point is independent of where the free units start, so that axis is dropped.
#
# The scatter is the point of this panel: does consolidation (`F_S`) predict recall? One dot per
# memory, so it stays readable at any `P`.

# %% recall
def landing(mdl, fracs=(0.25, 0.5, 0.75), n_probe=5, seed=0):
    """Clamped partial-cue recall -> relative landing distance, shape (len(fracs), P)."""
    gen = torch.Generator().manual_seed(seed)
    d, P_ = mdl.patterns.shape
    out = np.zeros((len(fracs), P_))
    for i, frac in enumerate(fracs):
        for p in range(P_):
            m = mdl.patterns[:, p]
            errs = []
            for _ in range(n_probe):
                known = torch.rand(d, generator=gen, dtype=m.dtype) < frac
                x = mdl.recall(m.unsqueeze(0), known=known, cue=m)
                errs.append(float(((x - m).norm(dim=1) / m.norm()).mean()))
            out[i, p] = np.mean(errs)
    return out

fracs = (0.25, 0.5, 0.75)
land = landing(model, fracs)

# Reference scales, measured from the patterns themselves. Without them the landing values are
# just small numbers with no meaning. Everything is relative to ||m_p|| (patterns are unit-norm):
#   - "cue only"      = give up and leave the hidden units at zero  -> sqrt(1 - frac)
#   - "a wrong memory" = ||m_p - m_q|| for p != q; landing well below the NEAREST one means the
#                        query resolved to the correct memory, not merely to somewhere plausible.
Mu = model.patterns.numpy()
Mu = Mu / np.linalg.norm(Mu, axis=0, keepdims=True)
pair_d = np.sqrt(np.clip(2 - 2 * (Mu.T @ Mu)[~np.eye(P, dtype=bool)], 0, None))
wrong_mean, wrong_min = float(pair_d.mean()), float(pair_d.min())
cue_only = np.sqrt(1.0 - np.asarray(fracs, dtype=float))

for i, frac in enumerate(fracs):
    print(f"{frac:.0%} revealed : mean {land[i].mean():.4f}  worst {land[i].max():.4f}   "
          f"[cue only {cue_only[i]:.2f}]")
print(f"a wrong memory sits at {wrong_mean:.2f} on this scale (nearest pair {wrong_min:.2f}); "
      f"best recall is {wrong_min / land.min():.0f}x closer than the nearest wrong memory")

fig3, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].axhline(wrong_mean, ls="--", c="C3", lw=1.6,
              label=f"a wrong memory (mean {wrong_mean:.2f})")
ax[0].axhline(wrong_min, ls=":", c="C3", lw=1.4,
              label=f"nearest wrong memory ({wrong_min:.2f})")
ax[0].plot(fracs, cue_only, "s:", c="0.55", lw=1.5, label="no completion (cue only)")
ax[0].plot(fracs, land.mean(1), "o-", lw=2.2, c="C0", label="recall (mean over memories)")
ax[0].fill_between(fracs, land.min(1), land.max(1), alpha=0.2, color="C0", label="min–max")
ax[0].set(title=f"Landing distance ({ACT})", xlabel="fraction of units revealed",
          ylabel=r"$\|x_\infty - m_p\| / \|m_p\|$", yscale="log")
ax[0].set_ylim(bottom=max(float(land.min()) * 0.5, 1e-5), top=2.0)
ax[0].legend(fontsize=8, loc="center right")

ax[1].scatter(energy[-1], land[-1], s=45, c=range(P), cmap="viridis")
for p in range(P):
    ax[1].annotate(str(p), (energy[-1][p], land[-1][p]), fontsize=7,
                   xytext=(3, 3), textcoords="offset points")
ax[1].set(title=f"Does consolidation predict recall?  ({fracs[-1]:.0%} revealed)",
          xlabel=r"final $F_S(m_p)$", ylabel=r"landing distance")
fig3.tight_layout()
fig3

# %% [markdown]
# ## Notes
#
# - **Raster, not 3-D.** The old `trajectory_3d` needs exactly three patterns; the raster is the
#   same information at any `P`, and it makes off-manifold excursions (`c_p < 0`, blue) visible,
#   which the 3-D view did not.
# - **Watch the dominance bars.** At `P = 3` the teacher already tends to park on one memory rather
#   than tour all of them. Whether that gets worse with `P` — a few memories hogging replay while a
#   tail never consolidates — is exactly what the sorted-`F_S` bar chart and the dominance bars
#   answer together.
# - **`P` vs `d`.** `build_memory` asserts representability, so an over-full pattern set fails loudly
#   rather than silently storing something else.
