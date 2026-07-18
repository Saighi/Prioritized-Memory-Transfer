# ARCHIVED: this notebook exercises the additive-synthesis API (pmt.additive) removed in
# commit c4e894d (2026-07-18); it last ran against commit fbafa8c. Kept for reference only.
# See archive/three_network_additive_memory_synthesis.md for the retired model spec.

# %% [markdown]
# # Additive memory synthesis — edge cases & ablations
#
# Run cell-by-cell in VS Code (**`pytorch`** kernel). This exercises the additive three-network
# model across the recommended geometries (spec §20) and the required ablations A–I (§21), each
# with a short adiabatic run and a printed **prediction → observation** check. It is the companion
# to `01_synthesis_single_run.py` (the core-dynamics demo).
#
# The runs here are deliberately small/short so the whole notebook executes quickly; the point is
# the *qualitative* behavior, not a polished floor. Numbers will look cleaner with longer runs.

# %% imports, path setup, and a tiny run helper
import os

import numpy as np
import torch


from pmt import AdditiveSynthesisConfig, SimConfig, build_additive_synthesis, simulate

SHOW = os.environ.get("PMT_NO_SHOW") != "1"
torch.manual_seed(0)


def run(n_steps=40000, record_every=1000, mode="adiabatic", **cfg_kw):
    """Build + simulate an additive system; return (H, info, macro)."""
    cfg = AdditiveSynthesisConfig(**cfg_kw)
    macro, info = build_additive_synthesis(cfg)
    sim = SimConfig(n_steps=n_steps, dt=0.5, mode=mode, record_every=record_every,
                    n_weight_snapshots=4, progress=False)
    H = simulate(macro, sim, info).to_numpy()
    return H, info, macro


# common "normal-regime" knobs reused across cases (kept small for speed)
BASE = dict(d=32, pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9,
            eta=0.05, sigma_xi1=0.12, sigma_xi2=0.12, seed=0)

# %% [markdown]
# ## §20 Teacher geometries — orthogonal / partially-shared / oblique
# `r_Σ = r₁ + r₂ − dim(𝒰₁∩𝒰₂)`. Shared directions are counted once; oblique spaces have no exact
# intersection (small principal angles) so `r_Σ = r₁ + r₂`.

# %%
for geom, kw in [("orthogonal", dict(geometry="orthogonal")),
                 ("shared(ov=2)", dict(geometry="shared", overlap=2)),
                 ("oblique(θ=0.3)", dict(geometry="oblique", principal_angle=0.3))]:
    H, info, _ = run(n_steps=45000, rank1=4, rank2=4, **kw, **BASE)
    print(f"{geom:16s}  r_Σ={info['r_Sigma']}  overlap={info['overlap']:>1}  "
          f"E_Σ {H['E_Sigma'][0]:.2f}->{H['E_Sigma'][-1]:.3f}  "
          f"novelty {H['novelty_spec'][0].sum():.2f}->{H['novelty_spec'][-1].sum():.3f}")
# shared spaces need less capacity than the naive r1+r2:
_, info_sh, _ = run(n_steps=1, rank1=4, rank2=4, geometry="shared", overlap=2, **BASE)
assert info_sh["r_Sigma"] == 4 + 4 - 2, "overlap not counted once"
print("→ overlap learned once: r_Σ = r1+r2-overlap =", info_sh["r_Sigma"], "(§13, §19.4)")

# %% [markdown]
# ## §20.5–6 Capacity edge and over-capacity (ablation G)
# Transfer succeeds when `r_Σ ≤ d−1` and hits a **structural floor** when the target exceeds
# synthesis capacity (`r_Σ ≥ d`). We probe a tiny `d` so the edge is reachable quickly.

# %%
d_small = 8
edge, _, _ = run(n_steps=45000, d=8, rank1=3, rank2=4, geometry="orthogonal",  # r_Σ=7=d-1
                 pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9, eta=0.05,
                 sigma_xi1=0.12, sigma_xi2=0.12, seed=0)
over_cfg = AdditiveSynthesisConfig(d=8, rank1=4, rank2=4, geometry="orthogonal",  # r_Σ=8=d (over)
                                   pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9, eta=0.05,
                                   sigma_xi1=0.12, sigma_xi2=0.12, seed=0)
over_macro, over_info = build_additive_synthesis(over_cfg)
over = simulate(over_macro, SimConfig(n_steps=45000, dt=0.5, mode="adiabatic",
                                      record_every=1000, progress=False), over_info).to_numpy()
print(f"capacity edge  r_Σ={7} (=d-1): E_Σ -> {edge['E_Sigma'][-1]:.3f}   (should reach a low floor)")
print(f"over capacity  r_Σ={over_info['r_Sigma']} (=d): E_Σ -> {over['E_Sigma'][-1]:.3f}   "
      f"(structural floor, capacity_ok={over_info['capacity_ok']})")
assert over_info["capacity_ok"] is False, "over-capacity should be flagged"

# %% [markdown]
# ## §3 / ablation A — one common additive error vs two separate errors
# The additive architecture self-terminates (`‖ε_Σ‖ → 0`). With **separate** interface errors the
# residual persists whenever the teachers disagree (`x₁ ≠ x₂`) — no clean termination.

# %%
Ha, _, _ = run(n_steps=60000, rank1=3, rank2=3, geometry="shared", overlap=1, **BASE)
Hs, _, _ = run(n_steps=60000, rank1=3, rank2=3, geometry="shared", overlap=1,
               separate_errors=True, **BASE)
print(f"additive  terminal ‖ε_Σ‖ = {Ha['eps_Sigma_norm'][-1]:.4f}   ‖dW_S‖ = {Ha['dWS_norm'][-1]:.4f}")
print(f"separate  terminal ‖ε‖   = {Hs['eps_Sigma_norm'][-1]:.4f}   ‖dW_S‖ = {Hs['dWS_norm'][-1]:.4f}")
assert Hs["eps_Sigma_norm"][-1] > 3 * Ha["eps_Sigma_norm"][-1], "separate errors should not terminate"
print("→ separate errors keep a persistent teacher-disagreement residual (§3)")

# %% [markdown]
# ## Ablation B — no teacher noise
# Without noise, just-learned / exactly-cancelling directions can become absorbing; transfer
# stalls except for numerical roundoff. Small decorrelated noise is load-bearing (§11, §19.6).

# %%
Hn, _, _ = run(n_steps=60000, rank1=3, rank2=3, geometry="shared", overlap=1,
               d=32, pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9, eta=0.05,
               sigma_xi1=0.0, sigma_xi2=0.0, seed=0)            # NO noise
print(f"no noise   novelty {Hn['novelty_spec'][0].sum():.2f}->{Hn['novelty_spec'][-1].sum():.3f}  "
      f"E_Σ->{Hn['E_Sigma'][-1]:.3f}   mix λ_min end={Hn['mix_min_eig'][-1]:.2e}")
print(f"with noise novelty {Ha['novelty_spec'][0].sum():.2f}->{Ha['novelty_spec'][-1].sum():.3f}  "
      f"E_Σ->{Ha['E_Sigma'][-1]:.3f}   (same geometry, σ=0.12)")
print("→ noise breaks cancellation / absorbing saddles and exposes the unlearned directions")

# %% [markdown]
# ## Ablation C — perfectly correlated teacher noise
# Sharing one noise draw across teachers reduces mixture diversity, so the persistent-excitation
# margin `λ_min(U_Σᵀ Σ_y U_Σ)` shrinks and exploration of `𝒰_Σ` is poorer (§14, §19.5).

# %%
Hdec, _, _ = run(n_steps=45000, rank1=3, rank2=3, geometry="shared", overlap=1,
                 correlated_noise=False, **BASE)
Hcor, _, _ = run(n_steps=45000, rank1=3, rank2=3, geometry="shared", overlap=1,
                 correlated_noise=True, **BASE)
print(f"decorrelated  mix λ_min end = {Hdec['mix_min_eig'][-1]:.3e}   E_Σ->{Hdec['E_Sigma'][-1]:.3f}")
print(f"correlated    mix λ_min end = {Hcor['mix_min_eig'][-1]:.3e}   E_Σ->{Hcor['E_Sigma'][-1]:.3f}")
print("→ correlated noise → smaller minimum mixture eigenvalue (weaker excitation)")

# %% [markdown]
# ## Ablation D — reversed precision above the structure guard
# Below the guard the teachers stay near their own manifolds; pushed above it, novelty
# amplification beats self-damping and teacher activity leaks off-manifold (§17).

# %%
guard = build_additive_synthesis(AdditiveSynthesisConfig(
    d=32, rank1=3, rank2=3, geometry="shared", overlap=1))[1]["guard"]
Hsafe, _, _ = run(n_steps=30000, rank1=3, rank2=3, geometry="shared", overlap=1,
                  rho=-0.5 * guard, d=32, pi_S=0.5, pi_I=1.0, eta=0.05,
                  sigma_xi1=0.12, sigma_xi2=0.12, seed=0)
Hbig, _, _ = run(n_steps=30000, rank1=3, rank2=3, geometry="shared", overlap=1,
                 rho=-6.0 * guard, d=32, pi_S=0.5, pi_I=1.0, eta=0.05,
                 sigma_xi1=0.12, sigma_xi2=0.12, seed=0)
print(f"guard |ρ|* = {guard:.3f}")
print(f"|ρ|=0.5·guard  leakage L1,L2 end = {Hsafe['L1'][-1]:.3f}, {Hsafe['L2'][-1]:.3f}  (small)")
print(f"|ρ|=6·guard    leakage L1,L2 end = {Hbig['L1'][-1]:.3f}, {Hbig['L2'][-1]:.3f}  (blown up)")
assert Hbig["L1"][-1] > 5 * Hsafe["L1"][-1], "above-guard drive should increase leakage"

# %% [markdown]
# ## Ablation E — wake sign throughout (ρ > 0)
# With an ordinary (un-reversed) precision the teachers *chase* the prediction; there is no
# novelty amplification, so autonomous transfer is weak or absent (§7 wake).

# %%
Hwake, _, _ = run(n_steps=45000, rank1=3, rank2=3, geometry="shared", overlap=1,
                  rho=+0.3, d=32, pi_S=0.5, pi_I=1.0, eta=0.05,
                  sigma_xi1=0.12, sigma_xi2=0.12, seed=0)
print(f"wake (ρ=+0.3)  novelty {Hwake['novelty_spec'][0].sum():.2f}->{Hwake['novelty_spec'][-1].sum():.3f}  "
      f"E_Σ {Hwake['E_Sigma'][0]:.2f}->{Hwake['E_Sigma'][-1]:.3f}")
print(f"sleep (ρ<0)    novelty {Ha['novelty_spec'][0].sum():.2f}->{Ha['novelty_spec'][-1].sum():.3f}  "
      f"E_Σ {Ha['E_Sigma'][0]:.2f}->{Ha['E_Sigma'][-1]:.3f}")
print("→ wake does not drive transfer; sleep (reversed precision) does")

# %% [markdown]
# ## Ablation F — remove the norm constraints
# Without the amplitude leash, positive novelty growth makes teacher amplitudes diverge **along
# unlearned directions** (§17). To isolate the leash we freeze the synthesis network (`eta=0`), so
# every direction stays unlearned (`N_S` never decays) and the reversed-precision drive keeps
# pushing — the leashed teacher stays pinned at `r₁=1` while the unleashed one runs away.

# %%
common_F = dict(d=32, rank1=3, rank2=3, geometry="shared", overlap=1, pi_S=0.5, pi_I=1.0,
                rho="auto", rho_safety=0.9, eta=0.0, sigma_xi1=0.12, sigma_xi2=0.12, seed=0)
cfg_leash = AdditiveSynthesisConfig(**common_F, norm_constraint=True)
cfg_free = AdditiveSynthesisConfig(**common_F, norm_constraint=False)
short = SimConfig(n_steps=4000, dt=0.5, mode="adiabatic", record_every=4000, progress=False)
mL, iL = build_additive_synthesis(cfg_leash); simulate(mL, short, iL)
mF, iF = build_additive_synthesis(cfg_free); simulate(mF, short, iF)
nL = float(mL.populations["T1"].x.norm())
nF = float(mF.populations["T1"].x.norm())
print(f"leashed   ‖x₁‖ = {nL:.3f}  (held at r₁=1)")
print(f"unleashed ‖x₁‖ = {nF:.3e}  (grows without bound)")
assert nF > 10 * nL, "removing the leash should let teacher amplitude diverge"

# %% [markdown]
# ## Ablation H — strong source imbalance
# With `α₂ ≪ α₁`, teacher 1 transfers first; teacher 2 still transfers afterwards but slower. At
# `α₂ = 0`, teacher 2 is invisible and cannot transfer (§12, §19.3).

# %%
Himb, _, _ = run(n_steps=60000, rank1=3, rank2=3, geometry="orthogonal",
                 alpha1=1.0, alpha2=0.15, d=32, pi_S=0.5, pi_I=1.0, rho="auto",
                 rho_safety=0.9, eta=0.05, sigma_xi1=0.12, sigma_xi2=0.12, seed=0)
print(f"imbalance α=(1.0, 0.15):  E₁ {Himb['E1'][0]:.2f}->{Himb['E1'][-1]:.3f}  "
      f"E₂ {Himb['E2'][0]:.2f}->{Himb['E2'][-1]:.3f}")
print(f"  early J₁,J₂ = {Himb['J1'][2]:.3f}, {Himb['J2'][2]:.4f}   "
      f"late J₁,J₂ = {Himb['J1'][-1]:.4f}, {Himb['J2'][-1]:.4f}")
print("→ the strong source (T1) transfers first and loses its drive; the weak source lags")

# %% [markdown]
# ## Ablation I — slow synthesis inference (full mode)
# Increasing `τ_S/min(τ₁,τ₂)` delays novelty filtering. In `mode="full"` the synthesis state is
# integrated (not solved), so it lags the additive prediction and agrees less well with the
# fast-synthesis (`adiabatic`) reduction (§8, §21I).

# %%
Hadi, _, _ = run(n_steps=45000, rank1=3, rank2=3, geometry="shared", overlap=1,
                 mode="adiabatic", tau_S=1.0, **BASE)
Hful, _, mac = run(n_steps=45000, rank1=3, rank2=3, geometry="shared", overlap=1,
                   mode="full", tau_S=3.0, **BASE)
print(f"adiabatic (τ_S→0): E_Σ->{Hadi['E_Sigma'][-1]:.3f}  ‖ε_Σ‖ end={Hadi['eps_Sigma_norm'][-1]:.3f}")
print(f"full (τ_S=3):      E_Σ->{Hful['E_Sigma'][-1]:.3f}  ‖ε_Σ‖ end={Hful['eps_Sigma_norm'][-1]:.3f}")
print("→ a slow synthesis network filters novelty less cleanly (weaker agreement with theory)")

# %% [markdown]
# ## Summary
# Across all cases the additive three-network model behaves as `three_network_additive_memory_synthesis.md`
# predicts: overlap is learned once, capacity follows `r_Σ ≤ d−1`, the common additive error
# self-terminates (separate errors do not), decorrelated noise is load-bearing, the structure guard
# bounds teacher leakage, wake does not transfer, the norm leash prevents divergence, source drive
# rebalances toward the unlearned teacher, and fast synthesis inference matches the reduced theory.
