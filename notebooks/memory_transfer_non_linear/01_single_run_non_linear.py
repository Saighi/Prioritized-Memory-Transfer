# %% [markdown]
# # Prioritized memory transfer with nonlinear units
#
# Run cell-by-cell in VS Code (select the **`pytorch`** conda interpreter as the kernel).
# Same two-population system as `notebooks/memory_transfer/01_single_run.py` — teacher **T**
# (frozen) above student **S** (plastic), sleep/replay regime — but every unit transmits `f(x)`
# with **no bias**. Same engine, same equations, same amplitude leash, all precisions fixed.
# Switch `activation` below between `"relu"` and `None` (linear) to compare.
#
# The self error becomes `eps = x - W f(x)`, so `M = I - W` generalizes to the state-dependent
# Jacobian `J = I - W diag(f'(x))`. The memories are exact zero-error fixed points for *any*
# activation (`build_memory` fits `W` through `f`), but only relu on NONNEGATIVE patterns also
# satisfies `f(m_p) = m_p`, which is what keeps `ker(I - W_T)` equal to the memory manifold.
# Where that fails the manifold-based panels are flagged; the rest stays exact.
#
# **Only exact observables here.** Every linear-model diagnostic that silently became a local
# linearization was dropped rather than replotted:
#
# | dropped | why |
# |---|---|
# | novelty staircase `eig(Uᵀ N_S U)` | `N_S` is linearized at `x_S` yet evaluated on far-away directions of `U_T`; and its defining identity (the fast-student steady state) no longer holds |
# | manifold occupancy | measures projection on the linear *hull*, so a state far off the manifold can still score 1.0. `F_T` answers the same question exactly |
# | `\|cos(x_T, m_p)\|` raster | the absolute value is now wrong (`-m_p` is not a memory), and with non-orthogonal patterns the cosine barely discriminates. Replaced by the true expansion coefficients |
# | `S_T` spectrum + guard | a stability threshold from `I - W_T` alone, which does not govern the dynamics |
#
# Replaced by the exact manifold spectrum, the memory coordinates of `x_T`, and a clamped recall
# test. Switching `activation` keeps all of these valid except the manifold spectrum, which is
# labelled automatically (it needs `f(m_p) = m_p`).

# %% imports & setup
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from prioritized_memory_transfer import ModelConfig, SimConfig, build_system, simulate
from prioritized_memory_transfer import diagnostics as dg
from prioritized_memory_transfer import viz_interactive as vi
from prioritized_memory_transfer.viz_style import mpl_style

SHOW = os.environ.get("PMT_NO_SHOW") != "1"   # set PMT_NO_SHOW=1 to run headless
mpl_style()
torch.manual_seed(0)

# %% build, check, run
# All precisions are FIXED numbers here — no "auto" resolution off the spectral gap. pi_TS > pi_S
# keeps the student pinned near the teacher's manifold; pi_ST < 0 is the sleep/replay drive. The
# same values are used for every activation, so the comparisons below are apples to apples.
ACTIVATION = "relu"          # "relu" | None (linear)
PATTERNS   = "nonneg"        # "nonneg" | "target_corr_nonneg" | "orthonormal" | "target_corr"
D, P_MEM   = 100, 3

# Both activations (relu, identity) are positively homogeneous, so the dynamics do not depend on
# this at all — it only sets the units the states are reported in.


model_cfg = ModelConfig(
    d=D, P=P_MEM,
    pi_TS=1.0, pi_S=0.5,              # student follows the teacher (pi_TS > pi_S)
    pi_ST=-0.5,                       # reversed (negative) ⇒ sleep/replay drive
    tau_T=10.0, tau_S=1.0, eta=0.002,
    sigma_xi=0.05, r0=1.0,          # the leash: with no bias, x=0 is itself a zero-error
                                      # memory, and sleep adds an unopposed +|pi_ST| x_T term
    pattern_kind=PATTERNS,            # nonneg kinds give f(m_p)=m_p under a rectifying f
    corr_target=0.3,                  # target_corr*: the pairwise correlation to hit
    activation=ACTIVATION,
    seed=0, device="cpu",
)
model, info = build_system(model_cfg)

# Checks that still hold exactly. `patterns_fixed` (f(m_p) = m_p) is the load-bearing one: it is
# what keeps ker(I - W_T) equal to the memory manifold. Circulation is untouched by the
# nonlinearity because only the (linear) interface contributes cross-derivatives.
e_xS, e_WS = dg.check_gradients(model)              # nonlinear rates vs autograd on F_S
c = dg.circulation(model)
ACT = info.activation or "linear"      # labels follow the activation, so switching it stays honest
print(f"activation             : {ACT}")
print(f"patterns fixed f(mₚ)=mₚ : {info.patterns_fixed} | manifold dim {info.manifold_dim}")
print(f"memory residual        : {info.memory_residual:.1e}  (exact fixed points of the network)")
print(f"gradient check         : err_xS={e_xS:.1e} err_WS={e_WS:.1e}")
print(f"circulation            : {c:.3f}  (= |π_ST+π_TS|·√d, as in the linear model)")
assert info.memory_residual < 1e-4 and e_xS < 1e-8 and e_WS < 1e-8

# `patterns_fixed` is a soft gate, not an assert, so relu on SIGNED patterns still runs: with
# f(m_p) != m_p the memories become isolated points, ker(I - W_T) collapses, and the manifold
# spectrum stops being meaningful. Everything else below stays exact.
EXACT_MANIFOLD = bool(info.patterns_fixed)
if not EXACT_MANIFOLD:
    print("  !! f(mₚ)≠mₚ — the manifold spectrum panel is NOT valid for this activation.")

# mode="full" is mandatory — adiabatic elimination of the student needs linearity.
sim_cfg = SimConfig(n_steps=100000, dt=0.5, mode="full", record_every=100,
                    n_weight_snapshots=0, progress=True)
hist = simulate(model, sim_cfg, info)
H = hist.to_numpy()
t = H["t"]
assert float(torch.diagonal(model.W_S).abs().max()) < 1e-10, "W_S diagonal drifted (autapse!)"
assert bool(torch.isfinite(model.W_S).all()), "W_S blew up"

# %% [markdown]
# ## Consolidation
#
# **`F_S(m_p)`** — the student's free energy *at* each stored memory. The interface term vanishes
# there (clamping both populations to `m_p` gives `eps_TS = 0`), so this is the exact total free
# energy of that configuration, no linearization anywhere. It starts at `½π_S‖m_p‖² = 0.25`.
#
# **The manifold spectrum** generalizes it to the whole manifold in closed form. On the cone the
# student's error map is linear, so with error vectors `E = [m_p − W_S f(m_p)]`, every manifold
# state `x = Mc` has `F_S(Mc) = ½π_S cᵀGc` with `G = EᵀE`, and `‖Mc‖² = cᵀKc` with `K = MᵀM`. The
# eigenvalues of the pencil `(G, K)` are exactly `F_S` per unit `‖x‖²` over the *entire* manifold.
# Same units as `F_S(m_p)`; the top eigenvalue is the worst-case memory state.

# %% figure 1 — consolidation
energy = 0.5 * model_cfg.pi_S * H["residual"] ** 2      # F_S at each stored memory
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

for p in range(energy.shape[1]):
    ax[0].plot(t, energy[:, p], lw=1.9, label=f"$m_{p}$")
ax[0].set(title="Free energy at each stored memory", xlabel="time", ylabel=r"$F_S(m_p)$")
ax[0].legend()

for j in range(H["manifold_spec"].shape[1]):
    ax[1].plot(t, H["manifold_spec"][:, j], lw=1.9)
ax[1].set(title="Manifold surprise spectrum"
                + (" (exact)" if EXACT_MANIFOLD else "  [INVALID: f(mₚ)≠mₚ]"),
          xlabel="time", ylabel=r"$F_S$ per unit $\|x\|^2$")

ax[2].plot(t, H["F_T"], lw=1.8, label="$F_T$  (T on its manifold?)")
ax[2].plot(t, H["F_S"], lw=1.8, label="$F_S$")
ax[2].plot(t, H["Phi"], lw=1.8, label=r"$\Phi = F_T - F_S$")
ax[2].set(title="Free energies along the run", xlabel="time")
ax[2].legend()
fig.tight_layout()
fig

# %% [markdown]
# ## Did the student copy the teacher's synapses?
#
# The three matrices side by side on a shared colour scale. Note the **sign** structure: with
# rectified units every rate is nonnegative, yet `W_S` must still grow inhibitory synapses to
# match `W_T`. It can, because the Hebbian rule `ΔW_ij ∝ ε_i f(x_j)` takes its sign from the
# postsynaptic *error*, not from the presynaptic rate. A rate-rate rule `ΔW_ij ∝ x_i x_j` could
# only ever grow positive weights here — which is why Hopfield-style storage needs ±1 patterns.

# %% figure — W_S vs W_T
WT = model.W_T.detach().numpy()
WS = model.W_S.detach().numpy()
v = float(np.abs(np.stack([WT, WS])).max())

figw, axw = plt.subplots(1, 3, figsize=(13, 4.0))
for a, Mx, ttl in zip(axw, (WT, WS, WS - WT),
                      ("$W_T$  (teacher, frozen)", "$W_S$  (student, learned)", "$W_S - W_T$")):
    imw = a.imshow(Mx, cmap="RdBu_r", vmin=-v, vmax=v)
    a.set(title=ttl, xticks=[], yticks=[])
    a.grid(False)
figw.colorbar(imw, ax=axw, fraction=0.02, pad=0.02)
print(f"negative entries : W_T {(WT < 0).mean():.1%}   W_S {(WS < 0).mean():.1%}")
print(f"correlation      : {np.corrcoef(WT.ravel(), WS.ravel())[0, 1]:.3f}")
print(f"relative error   : ||W_S-W_T||/||W_T|| = "
      f"{np.linalg.norm(WS - WT) / np.linalg.norm(WT):.3f}")
figw

# %% [markdown]
# ## Where replay goes
#
# The true expansion coefficients `c = argmin‖x_T − Mc‖`, *not* inner products: nonnegative
# patterns are strongly non-orthogonal, so `x·m_p` is positive for essentially any state and would
# barely discriminate. `c_p` is honestly "how much of memory `p`", and `c_p < 0` means the teacher
# has drifted off the memory manifold.

# %% figure 2 — memory coordinates of x_T
coef = np.linalg.lstsq(model.patterns.numpy(), H["x_T"].T, rcond=None)[0].T   # (T, P)

fig2, ax2 = plt.subplots(figsize=(11, 4.0))
for p in range(coef.shape[1]):
    ax2.plot(t, coef[:, p], lw=1.6, label=f"$c_{p}$")
ax2.axhline(0, c="k", lw=0.8, alpha=0.5)
ax2.set(title=r"Memory coordinates of $x_T$  ($x_T \approx \sum_p c_p m_p$)", xlabel="time",
        ylabel="$c_p$")
ax2.legend()
fig2.tight_layout()
fig2

# %% [markdown]
# ## Recall: where does a partial query land?
#
# `F_S(m_p) = 0` makes `m_p` a zero-energy state, but says nothing about the student's
# **dynamics** — whether a query actually converges there. That needs its own experiment.
#
# **Clamping is mandatory, for the linear model just as much as the ReLU one.** The memory
# manifold is degenerate (a subspace, or a cone), so *unclamped* relaxation has no preferred
# destination on it: every manifold point is an equally good minimum, and `x = 0` is one of them.
# A partial query breaks the degeneracy. Both models therefore get the identical protocol.
#
# One question: clamp a fraction of the units to `m_p`, start the free units a controlled distance
# `δ` away, relax the student alone (`model.recall`), and measure how far the landing point is from
# `m_p`. Below the diagonal means the query moved *toward* the memory; flat and low means it
# converged regardless of where it started.

# %% recall probe
def probe_recall(mdl, patterns, fracs=(0.25, 0.5, 0.75), deltas=(0.0, 0.25, 0.5, 1.0),
                 n_probe=6, seed=0):
    """Clamped partial-cue recall. Returns the landing distance ||x_inf - m_p|| / ||m_p||,
    shaped (len(fracs), len(deltas), P). Only free units are displaced, by exactly `delta`."""
    gen = torch.Generator().manual_seed(seed)
    d, P = patterns.shape
    out = np.zeros((len(fracs), len(deltas), P))
    for i, frac in enumerate(fracs):
        for p in range(P):
            m = patterns[:, p]
            known = torch.rand(d, generator=gen, dtype=patterns.dtype) < frac
            for j, delta in enumerate(deltas):
                xi = torch.randn(n_probe, d, generator=gen, dtype=patterns.dtype)
                xi[:, known] = 0.0                                  # the clamp holds these anyway
                xi = delta * xi / xi.norm(dim=1, keepdim=True).clamp_min(1e-12)
                x = mdl.recall(m.unsqueeze(0) + xi, known=known, cue=m)
                out[i, j, p] = float(((x - m).norm(dim=1) / m.norm()).mean())
    return out

fracs, deltas = (0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 1.0)
landing = probe_recall(model, model.patterns, fracs, deltas)

print("landing distance  ‖x∞ − mₚ‖ / ‖mₚ‖    (start distance δ across columns)")
print(f"{'revealed':>9}   " + "".join(f"δ={v:<7.2f}" for v in deltas))
for i, frac in enumerate(fracs):
    print(f"{frac:>8.0%}   " + "".join(f"{v:<9.3f}" for v in landing[i].mean(1)))
j = len(deltas) - 1
print(f"\nper memory at δ={deltas[j]} (F_S(mₚ) = {np.round(energy[-1], 4)}):")
for i, frac in enumerate(fracs):
    print(f"   {frac:.0%} revealed : {np.round(landing[i, j], 3)}")

# %% [markdown]
# ## Verdict: ReLU vs linear, identical patterns and seed

# %% linear baseline + comparison
lin_model, lin_info = build_system(ModelConfig(**{**model_cfg.__dict__, "activation": None}))
assert torch.allclose(lin_model.patterns, model.patterns), "baseline must use the same memories"
L = simulate(lin_model, sim_cfg, lin_info).to_numpy()
lin_energy = 0.5 * model_cfg.pi_S * L["residual"] ** 2
lin_landing = probe_recall(lin_model, lin_model.patterns, fracs, deltas)   # same clamped protocol

# Compare on maxₚ F_S(mₚ), which is exact for EVERY activation — unlike the manifold spectrum,
# which needs f(mₚ)=mₚ and is therefore not comparable when that fails.
print(f"final worst memory maxₚF_S  : {ACT}={energy[-1].max():.4f}   linear={lin_energy[-1].max():.4f}")
print(f"final ||W_S-W_T||_F         : {ACT}={H['WS_dist'][-1]:.3f}   linear={L['WS_dist'][-1]:.3f}")
print(f"final ||W_S||_F             : {ACT}={float(model.W_S.norm()):.3f}   "
      f"linear={float(lin_model.W_S.norm()):.3f}   (target {float(model.W_T.norm()):.3f})")

fig3, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for p in range(energy.shape[1]):
    ax[0].plot(t, energy[:, p], lw=1.8, c=f"C{p}", label=f"$m_{p}$")
    ax[0].plot(L["t"], lin_energy[:, p], lw=1.3, c=f"C{p}", ls="--", alpha=0.7)
ax[0].set(title=f"$F_S(m_p)$  (solid {ACT}, dashed linear)", xlabel="time")
ax[0].legend(fontsize=9)

ax[1].plot(t, energy.max(1), lw=2.0, label=ACT)
ax[1].plot(L["t"], lin_energy.max(1), lw=2.0, ls="--", label="linear")
ax[1].set(title=r"Worst memory  $\max_p F_S(m_p)$", xlabel="time", yscale="log")
ax[1].legend()

ax[2].plot(deltas, deltas, c="k", lw=0.9, ls=":", label="no movement")
for i, frac in enumerate(fracs):
    ax[2].plot(deltas, landing[i].mean(1), "o-", lw=1.8, c=f"C{i}", label=f"{frac:.0%} revealed")
    ax[2].plot(deltas, lin_landing[i].mean(1), "s--", lw=1.2, c=f"C{i}", alpha=0.6)
ax[2].set(title=f"Where the query lands ({ACT} solid, linear dashed)",
          xlabel=r"start distance $\delta$ from $m_p$", ylabel=r"$\|x_\infty - m_p\|$")
ax[2].legend(fontsize=8)
fig3.tight_layout()
fig3

# %% interactive: x_T trajectory in memory coordinates (press ▶ or drag the time slider)
# Axes are the expansion coefficients c — for a rectifying f the positive octant is the memory cone.
if SHOW:
    vi.trajectory_3d(hist, model, info, basis="patterns").show()

# %% [markdown]
# ## What to read off
#
# 1. **Does it consolidate?** Every `F_S(m_p)` should fall from `0.25` toward zero, and the worst
#    memory curve with it. Compare against the dashed linear baseline for the cost of the
#    nonlinearity.
# 2. **Where does a query land?** The landing-distance table is the direct answer: it depends on how
#    much you reveal, and — flatly, exactly — **not** on where you start. The clamped problem has a
#    single attractor per cue, so `δ` is irrelevant across the whole sweep including `δ = 1.0` (a
#    displacement as large as the pattern itself). The basin is global in the free coordinates.
# 3. **Is low energy enough?** Not automatically. Landing distance tracks `F_S(m_p)` per memory but
#    is a strictly stronger requirement: a memory can be ~95% consolidated by energy and still land
#    poorly. Read the two together, never `F_S` alone.
#
# ### Why rectification is the nonlinearity to study
# Two reasons, one biological and one structural. Cortical pyramidal cells are roughly
# threshold-linear over their operating range, so relu is the realistic idealization. And relu is
# **positively homogeneous** (`f(cx) = c f(x)` for `c > 0`), so it behaves identically at every
# amplitude — whereas any nonlinearity with an intrinsic scale is probed only near zero once the
# state is normalized (per-unit activity `~r0/√d`) and therefore degenerates to the identity as `d`
# grows. Rectification is the only choice that stays a genuine nonlinearity at any network size,
# which is why it is the only one this project keeps.
#
# ## Try next
# - **Robustness**: `seed=1,2` — the ordering of memories changes, the collapse should not.
# - **Accuracy vs speed**: `eta` 0.02 / 0.05 degrades the floor without diverging.
# - **More memories**: `P=6` — do later memories starve?
# - **Wake control**: `pi_ST > 0` ⇒ no transfer drive; nothing should consolidate.
