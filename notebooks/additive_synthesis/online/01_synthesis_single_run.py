# %% [markdown]
# # Additive memory synthesis — single run & dynamics
#
# Run cell-by-cell in VS Code (select the **`pytorch`** conda interpreter as the kernel).
# Two **frozen** predictive-coding teachers `T1`, `T2` are summed into one combined prediction
# `y = α₁x₁ + α₂x₂`; a single common error `ε_Σ = x_S − y` drives one **plastic** synthesis
# network `S`. In sleep (`ρ < 0`) the reversed teacher-side precision amplifies whatever part of
# the combined teacher activity `S` cannot yet predict, so `S` learns the **subspace sum**
# `𝒰_Σ = 𝒰₁ + 𝒰₂` — with automatic *source rebalancing* (a learned source loses its drive) and
# clean *self-termination*.
#
# This is the normal regime: **partially correlated memories** (the two teacher subspaces share a
# few directions) with small decorrelated exploration noise. Faithful to
# `three_network_additive_memory_synthesis.md`. The three networks are wired with the composable
# `pmt.macro` engine — the same engine the two-population model in `../two_network/` runs on.

# %% imports & path setup
import os
import sys
import pathlib

import torch

# make `import pmt` work whether cwd is the repo root or a notebooks/ subfolder
_root = pathlib.Path.cwd()
while not (_root / "pmt").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from pmt import AdditiveSynthesisConfig, SimConfig, build_additive_synthesis, simulate
from pmt import diagnostics as dg
from pmt import viz_additive as va

SHOW = os.environ.get("PMT_NO_SHOW") != "1"   # set PMT_NO_SHOW=1 to run headless
torch.manual_seed(0)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# %% configure & build the three-network system
cfg = AdditiveSynthesisConfig(
    d=40,
    rank1=3, rank2=3, geometry="shared", overlap=1,   # partially correlated: 1 shared direction
    pi_T1=1.0, pi_T2=1.0, pi_S=0.5, pi_I=1.0,          # precision guard pi_I > pi_S
    rho="auto", rho_safety=0.9,                        # reversed precision, inside the structure guard
    alpha1=1.0, alpha2=1.0,
    tau_T1=10.0, tau_T2=10.0, tau_S=1.0, eta=0.05,     # tau_S << tau_T << 1/eta
    sigma_xi1=0.12, sigma_xi2=0.12,                    # small *decorrelated* teacher noise
    r1=1.0, r2=1.0, seed=0, device="cpu",
)
macro, info = build_additive_synthesis(cfg)

print(f"teacher ranks         : r1={info['rank1']}  r2={info['rank2']}  overlap={info['overlap']}")
print(f"target subspace       : r_Σ = r1+r2-overlap = {info['r_Sigma']}   (capacity_ok={info['capacity_ok']})")
print(f"spectral gaps         : σ₁²_min={info['sigma1_min']:.3f}  σ₂²_min={info['sigma2_min']:.3f}")
print(f"ρ (signed)            : {info['rho']:.4f}   (reversed / negative ⇒ sleep-replay)")
print(f"structure guard       : |ρ|<guard({info['guard']:.3f})? {info['rho_ok']}   "
      f"precision pi_I>pi_S? {info['precision_ok']}")

# %% faithfulness self-checks (cheap; assert the invariants)
for name in ("T1", "T2"):
    W = macro.populations[name].W
    assert float(torch.diagonal(W).abs().max()) < 1e-12, f"{name} diagonal not zero"
# each teacher's patterns lie in ker(M) of its own frozen weights (memory is faithful):
for name, key in (("T1", "M1"), ("T2", "M2")):
    M = macro.populations[name].M
    resid = (M @ info[key]).norm(dim=0).max().item()
    print(f"{name} memory residual max_p‖M m_p‖ : {resid:.2e}")
    assert resid < 1e-4, f"{name} patterns not in ker M"

# %% run the transfer (adiabatic: S sits at its exact steady state x_S* = K_S y each step)
# ρ<0 (the reversed precision) ⇒ this is the sleep / replay regime.
sim = SimConfig(n_steps=150000, dt=0.5, mode="adiabatic", record_every=1000,
                n_weight_snapshots=6, progress=True)
hist = simulate(macro, sim, info)

H = hist.to_numpy()
print(f"E_Σ   : {H['E_Sigma'][0]:.3f} -> {H['E_Sigma'][-1]:.3f}   (combined-subspace deficit)")
print(f"E₁,E₂ : {H['E1'][0]:.3f}->{H['E1'][-1]:.3f} | {H['E2'][0]:.3f}->{H['E2'][-1]:.3f}")
print(f"novelty (sum eig U_Σᵀ N_S U_Σ): {H['novelty_spec'][0].sum():.3f} -> {H['novelty_spec'][-1].sum():.3f}")
print(f"termination: ‖ε_Σ‖ {H['eps_Sigma_norm'][0]:.3f}->{H['eps_Sigma_norm'][-1]:.3f}  "
      f"‖dW_S‖ {H['dWS_norm'][0]:.4f}->{H['dWS_norm'][-1]:.4f}")
assert float(torch.diagonal(macro.populations["S"].W).abs().max()) < 1e-10, "W_S diagonal drifted (autapse!)"

# %% static dashboard — the 6-panel overview of the dynamics
fig = va.dashboard(hist, info)
fig   # displays inline

# %% synthesis weights W_S over training, next to the two frozen teachers
fig_w = va.weight_snapshots(hist, info)
fig_w

# %% interactive: combined-subspace novelty staircase
if SHOW:
    va.staircase(hist).show()

# %% interactive: automatic source rebalancing (a learned source loses its drive)
if SHOW:
    va.source_balance(hist).show()

# %% [markdown]
# ## What to look for
# - **E_Σ, E₁, E₂ → a small floor**: `S` acquires the *complete* combined subspace; both sources
#   transfer (spec §19.1). The floor shrinks with more steps / slower learning / better timescale
#   separation.
# - **Novelty staircase → 0**: the eigenvalues of `U_Σᵀ N_S U_Σ` fall as directions are learned;
#   there are `r_Σ` independent decaying modes and **shared directions appear only once** (§19.2).
# - **Source rebalancing (J₁, J₂)**: whichever source is learned first loses its novelty drive, so
#   the remaining source takes over — *novelty homeostasis* (§12, §19.3).
# - **Self-termination**: after transfer `‖ε_Σ‖, ‖ε_S‖, ‖dW_S‖ → 0` (§15). Contrast this with the
#   **separate-error** control in `02_synthesis_edge_cases.py`, whose residual never vanishes.
#
# ## Try next
# - **Orthogonal / oblique teachers**: `geometry="orthogonal"` or `"oblique"` (see §20).
# - **Source imbalance**: `alpha2=0.2` — `T1` transfers first, then `T2` (§12, ablation H).
# - **Wake**: `rho=+0.3` (an ordinary, un-reversed precision) → no transfer drive.
# - All the edge cases and ablations live in `02_synthesis_edge_cases.py`.
