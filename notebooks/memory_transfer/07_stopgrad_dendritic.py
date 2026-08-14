# %% [markdown]
# # Stop-gradient (dendritic) variant — does our transfer dynamic survive?
#
# Tang et al. (2023, §"Dendritic covariance-learning PCNs", Eqs 16–18) make the recurrent
# PCN more biologically plausible by putting a **stop-gradient** on the prediction:
#
#     F = -1/2 || x - W·sg(x) - v ||²,   sg(x)=x forward,  d sg/dx = 0
#
# The only thing this changes is the **value-neuron inference**: it loses the backward
# excitatory term Wᵀε, so a neuron relaxes under  **-ε = -M x**  instead of the full
# PC gradient **-Mᵀε = -S x**  (S = MᵀM). The learning rule and the fixed points are
# *unchanged*. It also removes the implausible error→value backprojection.
#
# In **our** two-population model that backward term appears in exactly two places:
#   * T's self-relaxation     -M_Tᵀ ε_T   →   -ε_T   = -M_T x_T
#   * S's pattern-completion  -π_S M_Sᵀ ε_S →  -π_S ε_S = -π_S M_S x_S
# The teacher's reversed-precision drive (π_ST·ε_TS with π_ST<0) and the student's input
# pull (-π_TS ε_TS) are *interface* terms across the identity T↔S wiring — no recurrent Wᵀ —
# so the stop-grad leaves the load-bearing transfer drive untouched.
#
# **What this notebook tests (confined; the core `src` package is not modified):**
#   1. *Full memory transfer still happens* under stop-grad: the novelty staircase falls
#      to its floor, ‖M_S m_p‖→0 for every pattern, and W_S → W_T.
#   2. *The crux risk*: S = MᵀM is PSD **by construction**; M is not. Does the student's
#      learned M_S stay positive-definite / real-spectrumed under the destabilizing transfer
#      drive, or does the stop-grad relaxation go unstable?
#   3. *Side-effect of asymmetry*: M_S is not symmetric, so the restricted novelty
#      operator UᵀN_S^sg U need not be symmetric — we read it with general eigenvalues
#      and watch the imaginary part.

# %% imports & path setup
import os
import math
from collections import defaultdict
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]   # repo root (for the headless figure save)

import matplotlib
SHOW = os.environ.get("PMT_NO_SHOW") != "1"   # set PMT_NO_SHOW=1 to run headless
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from prioritized_memory_transfer import ModelConfig, SimConfig, build_system
from prioritized_memory_transfer.model import TwoPopModel

torch.manual_seed(0)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())


# %% [markdown]
# ## The stop-gradient model (confined subclass)
# Only the two value-neuron relaxation terms change: drop the `bwd(M, ·)` wrapper, i.e.
# replace `-Mᵀε` with `-ε`. We also re-derive the student's fast steady state and the novelty
# operator with `M_S` in place of `S_S`. Everything else (learning rule, errors, energies,
# renorm, zero-diag, signed-π_ST drive) is inherited unchanged from `TwoPopModel`.

# %% StopGradModel
class StopGradModel(TwoPopModel):
    """Tang et al. (2023) dendritic / stop-gradient variant of the two-population model.

    Full PC gradient relaxation  -Mᵀε = -S x   →   stop-grad relaxation  -ε = -M x.
    The fixed points (ε=0) are identical, so learning is unchanged; only the *path* of
    inference differs, and the always-PSD operator S=MᵀM is replaced by the (generally
    non-symmetric, not-necessarily-PD) operator M.
    """

    # T's environment dynamics: self-relaxation drops the bwd term  (-M_Tᵀε_T -> -ε_T)
    def rate_x_T_det(self) -> torch.Tensor:
        drift = -self.pi_T * self.eps_T() + self.pi_ST * self.eps_TS()
        return drift / self.tau_T

    # S's perception: pattern-completion drops the bwd term  (-π_S M_Sᵀε_S -> -π_S ε_S)
    def rate_x_S(self) -> torch.Tensor:
        return (-self.pi_TS * self.eps_TS() - self.pi_S * self.eps_S()) / self.tau_S

    # Fast-S steady state with the stop-grad completion:
    #   π_TS(x_T - x_S) - π_S M_S x_S = 0  =>  x_S* = π_TS (π_TS I + π_S M_S)^-1 x_T
    def solve_xS_steady(self, x_T: torch.Tensor) -> torch.Tensor:
        Amat = self.pi_TS * self.I + self.pi_S * self.M_S
        return self.pi_TS * torch.linalg.solve(Amat, x_T)

    # Novelty operator with M_S (not S_S):  ε_TS* = -N_S^sg x_T,
    #   N_S^sg = π_S M_S (π_TS I + π_S M_S)^-1
    def novelty_operator(self) -> torch.Tensor:
        MS = self.M_S
        Amat = self.pi_TS * self.I + self.pi_S * MS
        return self.pi_S * MS @ torch.linalg.inv(Amat)


# %% [markdown]
# ## Spectral-health helpers (asymmetry-aware)
# `S`-based operators are symmetric, so the baseline reads cleanly with `eigvalsh`. The
# stop-grad `M`-based operators are not symmetric, so we use general `eigvals` and report
# both the real spectrum and the largest imaginary part. We also probe M_T / M_S directly:
# `-M x` is a contraction iff the **symmetric part** of M is positive-definite.

# %% helpers
def restricted_novelty(model, U_T):
    """Eigenvalues of UᵀN_S U restricted to the teacher's manifold, descending real parts + max|Im|."""
    R = U_T.transpose(-2, -1) @ model.novelty_operator() @ U_T
    ev = torch.linalg.eigvals(R)
    real = torch.sort(ev.real, descending=True).values
    return real, float(ev.imag.abs().max())


def matrix_health(Mmat):
    """Spectral health of a relaxation operator M (governs stability of -M x):
       min Re λ(M), max |Im λ(M)|, and min eig of the symmetric part ½(M+Mᵀ).
       sym_min > 0  <=>  -M x is a strict contraction (Lyapunov via ½‖x‖²)."""
    ev = torch.linalg.eigvals(Mmat)
    sym = 0.5 * (Mmat + Mmat.transpose(-2, -1))
    sym_min = float(torch.linalg.eigvalsh(sym).min())
    return float(ev.real.min()), float(ev.imag.abs().max()), sym_min


def offmanifold_health(M_T, U_T, tol=1e-6):
    """Health of M_T restricted to the off-manifold subspace (complement of ker M_T) —
    the directions that must actually be damped. Returns (min Re λ, min eig sym part)."""
    d = M_T.shape[0]
    # orthonormal complement of U_T
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=M_T.dtype, device=M_T.device))
    P_off = torch.eye(d, dtype=M_T.dtype, device=M_T.device) - U_T @ U_T.transpose(-2, -1)
    # project, then take the nonzero block via SVD of P_off to get an orthobasis V_off
    U, S, _ = torch.linalg.svd(P_off)
    V_off = U[:, S > tol]                          # (d, d-k) orthobasis of off-manifold
    Mr = V_off.transpose(-2, -1) @ M_T @ V_off     # restricted operator
    return matrix_health(Mr)[0], matrix_health(Mr)[2]


# %% [markdown]
# ## A confined tracking loop
# Mirrors `prioritized_memory_transfer.model.simulate` (same Euler-Maruyama integration, same invariants:
# renormalize ‖x_T‖=r₀, zero diagonal of W_S), but records the asymmetry-aware novelty
# spectrum and the M_S health trace. Works for both models via polymorphism.

# %% track
def track(model, sim, info, seed_offset=12345):
    cfg = model.cfg
    gen = torch.Generator(device=model.device).manual_seed(cfg.seed + seed_offset)
    model.W_S = torch.zeros_like(model.W_T)        # S starts empty (idempotent re-runs)
    model.reset_state(gen)                          # pi_ST<0 (reversed) ⇒ sleep/replay regime
    if sim.pretrain_subset is not None:
        model.pretrain(sim.pretrain_subset)

    U_T, M = info.U_T, model.patterns
    dt = sim.dt
    noise_scale = model.sigma_xi / model.tau_T * math.sqrt(dt)
    rec = defaultdict(list)

    for step in range(sim.n_steps):
        # --- S (perception) ---
        if sim.mode == "adiabatic":
            model.x_S = model.solve_xS_steady(model.x_T)
        else:
            for _ in range(sim.s_substeps):
                model.x_S = model.x_S + dt * model.rate_x_S()

        # --- T (environment): drift + noise, then renormalize ---
        xi = noise_scale * torch.randn(model.d, generator=gen, dtype=model.dtype, device=model.device)
        model.x_T = model.x_T + dt * model.rate_x_T_det() + xi
        model.renorm_x_T()

        # --- S's learning (slow), then re-zero diagonal ---
        model.W_S = model.W_S + dt * model.rate_W_S()
        model.zero_diag_W_S()

        if step % sim.record_every == 0:
            nov_real, nov_im = restricted_novelty(model, U_T)
            ms_re, ms_im, ms_sym = matrix_health(model.M_S)
            rec["t"].append(step * dt)
            rec["nov_real"].append(nov_real.detach().cpu())
            rec["nov_imag_max"].append(nov_im)
            rec["ws_dist"].append(float((model.W_S - model.W_T).norm()))
            rec["resid"].append((model.M_S @ M).norm(dim=0).detach().cpu())
            rec["epsTS"].append(float(model.eps_TS().norm()))
            rec["manifold_occ"].append(
                float((U_T.transpose(-2, -1) @ model.x_T).norm() / model.x_T.norm().clamp_min(1e-12))
            )
            rec["ms_min_re"].append(ms_re)
            rec["ms_max_im"].append(ms_im)
            rec["ms_sym_min"].append(ms_sym)

    out = {
        "t": np.asarray(rec["t"]),
        "nov_real": torch.stack(rec["nov_real"]).numpy(),
        "nov_imag_max": np.asarray(rec["nov_imag_max"]),
        "ws_dist": np.asarray(rec["ws_dist"]),
        "resid": torch.stack(rec["resid"]).numpy(),
        "epsTS": np.asarray(rec["epsTS"]),
        "manifold_occ": np.asarray(rec["manifold_occ"]),
        "ms_min_re": np.asarray(rec["ms_min_re"]),
        "ms_max_im": np.asarray(rec["ms_max_im"]),
        "ms_sym_min": np.asarray(rec["ms_sym_min"]),
    }
    return out


# %% [markdown]
# ## Build the shared system, then a baseline and a stop-grad model on the *same* W_T

# %% build
model_cfg = ModelConfig(
    d=48, P=3,
    pi_TS=1.0, pi_S=0.5,               # precision guard π_TS > π_S
    pi_ST="auto", pi_ST_safety=0.5,    # π_ST = 0.5 · σ²_min (well inside the guard)
    tau_T=10.0, tau_S=1.0, eta=0.02,   # τ_S << τ_T << 1/η
    sigma_xi=0.05, r0=1.0,             # enough exploration noise for robust subspace coverage
    pattern_kind="orthonormal",
    seed=0, device="cpu",
)
baseline, info = build_system(model_cfg)                       # full-PC reference
stopgrad = StopGradModel(baseline.W_T, model_cfg,              # shares W_T + patterns
                         pi_ST=info.pi_ST, patterns=baseline.patterns)

U_T = info.U_T
sig2 = info.sigma2_min
# stop-grad damping is M_T (eigs ~ σ) not S_T (eigs ~ σ²): the guard rescales σ²_min -> σ_min.
sym_MT = 0.5 * (baseline.M_T + baseline.M_T.T)
ev_symMT = torch.linalg.eigvalsh(sym_MT)
sigma_min = float(ev_symMT[ev_symMT > 1e-6].min())            # off-manifold gap of sym(M_T)

print(f"manifold dim (eff rank) : {info.manifold_dim}  (= P for orthonormal patterns)")
print(f"π_ST                    : {info.pi_ST:.4f}  (reversed / <0)")
print(f"full-PC guard σ²_min     : {sig2:.4f}    (|π_ST| < σ²_min? {abs(info.pi_ST) < sig2})")
print(f"stop-grad guard σ_min    : {sigma_min:.4f}    (|π_ST| < σ_min?  {abs(info.pi_ST) < sigma_min})")
print(f"precision guard π_TS>π_S  : {info.precision_ok}")

# %% sanity: at W_S = 0 the two novelty operators must coincide (the 'extremes agree' claim)
n_base, _ = restricted_novelty(baseline, U_T)
n_sg, n_sg_im = restricted_novelty(stopgrad, U_T)
floor = model_cfg.pi_S / (model_cfg.pi_TS + model_cfg.pi_S)
print(f"\nAt W_S=0 (S empty): baseline novelty max={float(n_base.max()):.4f}, "
      f"stop-grad max={float(n_sg.max()):.4f}, theory n(1)={floor:.4f}")
print(f"  |N_S^sg - N_S| over manifold: {float((n_base - n_sg).abs().max()):.2e}  (should be ~0)")
print(f"  stop-grad novelty max |Im|  : {n_sg_im:.2e}  (0 here: M_S=I is symmetric)")

# off-manifold health of M_T: are the damped directions actually damped under -M_T x?
mt_re, mt_sym = offmanifold_health(baseline.M_T, U_T)
print(f"\nM_T off-manifold: min Re λ={mt_re:.4f}, min eig sym part={mt_sym:.4f}  "
      f"(>0 => -M_T x damps every off-manifold direction)")

# %% [markdown]
# ## Run both transfers

# %% run
sim_cfg = SimConfig(n_steps=100000, dt=0.5, mode="full", s_substeps=1,
                    record_every=200, progress=False)
print("running baseline (full PC) ...")
Hb = track(baseline, sim_cfg, info)
print("running stop-grad (dendritic) ...")
Hs = track(stopgrad, sim_cfg, info)


# S's perception matrix is -(π_TS I + π_S M_S): stable  <=>  Re λ(M_S) > -π_TS/π_S.
# (This is the *operative* bar. The stricter "-M_S x alone is a contraction" needs the
#  symmetric part > 0, but the +π_TS I input cushion makes that unnecessary.)
s_stab_thresh = -model_cfg.pi_TS / model_cfg.pi_S


def report(name, H):
    print(f"\n[{name}]")
    print(f"  novelty max : {H['nov_real'][0].max():.4f} -> {H['nov_real'][-1].max():.4f}")
    print(f"  ‖M_S m_p‖   : max {H['resid'][0].max():.3f} -> {H['resid'][-1].max():.4f}  (per-pattern residual)")
    print(f"  ‖W_S-W_T‖_F : {H['ws_dist'][0]:.3f} -> {H['ws_dist'][-1]:.4f}  (may floor >0: off-manifold underdetermined)")
    print(f"  manifold occ: {H['manifold_occ'].mean():.3f} (mean)  — high => not chasing noise")
    print(f"  M_S health  : min Re λ {H['ms_min_re'].min():.3f} | min eig sym {H['ms_sym_min'].min():.3f} "
          f"| max |Im λ| {H['ms_max_im'].max():.3f}")
    print(f"  S-perception margin: min Re λ(M_S)={H['ms_min_re'].min():.3f}  vs  instability at "
          f"{s_stab_thresh:.2f}  => margin {H['ms_min_re'].min() - s_stab_thresh:.3f}")


report("baseline / full PC", Hb)
report("stop-grad / dendritic", Hs)

# Honest, multi-part verdict (a single binary would hide the real result).
all_transferred = bool((Hs["resid"][-1] < 0.30).all())      # every direction learned (1.0 -> <0.3)
res_ratio = Hs["resid"][-1].max() / max(Hb["resid"][-1].max(), 1e-9)  # stop-grad floor / baseline floor
tight_as_baseline = res_ratio < 1.3
stable = Hs["ms_min_re"].min() > s_stab_thresh             # operative bar (S perception)
strict_pd = Hs["ms_sym_min"].min() > 0                      # stricter bar (-M_S x contraction)
print(f"\n>>> ALL DIRECTIONS TRANSFERRED (stop-grad): {all_transferred}  "
      f"(final per-pattern residuals all < 0.30: {np.array2string(Hs['resid'][-1], precision=3)})")
print(f">>> HELD AS TIGHTLY AS FULL-PC: {tight_as_baseline}  "
      f"(stop-grad floor {Hs['resid'][-1].max():.3f} = {res_ratio:.2f}x baseline {Hb['resid'][-1].max():.3f})")
print(f">>> S-PERCEPTION STABLE throughout: {stable}  "
      f"(min Re λ(M_S)={Hs['ms_min_re'].min():.3f} > {s_stab_thresh:.2f}; "
      f"strict -M_S x contractive? {strict_pd})")
print(f">>> novelty operator UᵀN_S^sg U non-symmetric: smallest real eig reaches "
      f"{Hs['nov_real'].min():.3f} (can go <0), max |Im λ| = {Hs['nov_imag_max'].max():.3f} "
      f"(baseline max |Im λ| = {Hb['nov_imag_max'].max():.0e})")

# Memory-MAINTENANCE diagnostic: does the per-pattern residual drift back UP after a
# direction is learned? (full-PC -S_S x is a clean PSD attractor; stop-grad -M_S x is weaker.)
for name, H in [("full PC", Hb), ("stop-grad", Hs)]:
    maxres = H["resid"].max(axis=1)
    best, final = float(maxres.min()), float(maxres[-1])
    tail = float(H["resid"].max(axis=1)[-len(maxres) // 4:].mean())
    print(f"  [{name}] max-residual best-ever={best:.3f}, final={final:.3f}, last-quarter mean={tail:.3f}  "
          f"-> drift-up factor {final / max(best, 1e-9):.2f}")
    print(f"           final per-pattern residuals: {np.array2string(H['resid'][-1], precision=3)}")

# %% [markdown]
# ## Comparison dashboard

# %% figure
sns.set_theme(context="notebook", style="whitegrid")
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Stop-gradient (dendritic) vs full-PC — does prioritized transfer survive?",
             fontsize=15, y=0.995)

# (0,0) baseline staircase
a = ax[0, 0]
for j in range(Hb["nov_real"].shape[1]):
    a.plot(Hb["t"], Hb["nov_real"][:, j], lw=1.6)
a.axhline(floor, ls="--", c="k", alpha=0.4, label=f"n(1)={floor:.3f}")
a.set(title="Full-PC baseline: novelty staircase  eig(UᵀN_S U)",
      xlabel="time", ylabel="novelty eigenvalue")
a.legend(fontsize=8)

# (0,1) stop-grad staircase
a = ax[0, 1]
for j in range(Hs["nov_real"].shape[1]):
    a.plot(Hs["t"], Hs["nov_real"][:, j], lw=1.6)
a.axhline(floor, ls="--", c="k", alpha=0.4, label=f"n(1)={floor:.3f}")
a.set(title="Stop-grad: novelty staircase  eig(UᵀN_S^sg U)",
      xlabel="time", ylabel="novelty eigenvalue")
a.legend(fontsize=8)

# (0,2) max-novelty (transfer completion), both overlaid
a = ax[0, 2]
a.plot(Hb["t"], Hb["nov_real"].max(1), c="C0", lw=2.0, label="full PC")
a.plot(Hs["t"], Hs["nov_real"].max(1), c="C3", lw=2.0, label="stop-grad")
a.axhline(0.0, ls=":", c="k", alpha=0.4)
a.set(title="Transfer completion: max novelty eigenvalue", xlabel="time", ylabel="max eigenvalue")
a.legend(fontsize=8)

# (1,0) W_S -> W_T, both
a = ax[1, 0]
a.plot(Hb["t"], Hb["ws_dist"], c="C0", lw=2.0, label="full PC")
a.plot(Hs["t"], Hs["ws_dist"], c="C3", lw=2.0, label="stop-grad")
a.set(title="Learning: ‖W_S − W_T‖_F", xlabel="time", ylabel="‖W_S − W_T‖_F")
a.legend(fontsize=8)

# (1,1) stop-grad per-pattern residual (full transfer, pattern by pattern)
a = ax[1, 1]
for p in range(Hs["resid"].shape[1]):
    a.plot(Hs["t"], Hs["resid"][:, p], lw=1.5, label=f"m{p}")
a.set(title="Stop-grad: S's residual per memory  ‖M_S m_p‖", xlabel="time", ylabel="residual")
a.legend(fontsize=8, ncol=2)

# (1,2) THE CRUX: stop-grad M_S spectral health vs the operative S-perception bar
a = ax[1, 2]
a.plot(Hs["t"], Hs["ms_min_re"], c="C2", lw=1.8, label="min Re λ(M_S)")
a.plot(Hs["t"], Hs["ms_sym_min"], c="C0", lw=1.8, label="min eig ½(M_S+M_Sᵀ)")
a.axhline(0.0, ls="--", c="k", alpha=0.4, label="strict-contraction bar")
a.axhline(s_stab_thresh, ls="-.", c="C3", alpha=0.8, label=f"S-perception unstable < {s_stab_thresh:.1f}")
a.set(title="CRUX — M_S stability under the transfer drive", xlabel="time", ylabel="eigenvalue")
a.legend(loc="center right", fontsize=7.5)
a2 = a.twinx()
a2.plot(Hs["t"], Hs["ms_max_im"], c="C1", lw=1.0, alpha=0.6)
a2.set_ylabel("max |Im λ(M_S)|", color="C1")
a2.grid(False)

fig.tight_layout(rect=[0, 0, 1, 0.98])

if SHOW:
    plt.show()
else:
    outpath = _root / "docs" / "stopgrad_dendritic.png"
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"\nsaved figure -> {outpath}")

# %% [markdown]
# ## Robustness across seeds (adiabatic — S exactly at steady state, isolates attractor strength)
# Is the looser stop-grad residual floor systematic or a one-seed fluke? Run a few noise/init
# seeds in adiabatic mode (fast, no S-lag confound). This is the last cell — it mutates the
# model objects, so re-run the dashboard cell above if you want to inspect them afterwards.

# %% seed sweep
def summarize(H):
    return {
        "nov": float(H["nov_real"][-1].max()),
        "res_floor": float(H["resid"].max(axis=1)[-len(H["t"]) // 4:].mean()),  # last-quarter mean of max-residual
        "occ": float(H["manifold_occ"].mean()),
        "margin": float(H["ms_min_re"].min() - s_stab_thresh),
    }


sweep_cfg = SimConfig(n_steps=30000, dt=0.5, mode="adiabatic", record_every=200, progress=False)
rows = []
for s in (1, 2, 3):
    rb = summarize(track(baseline, sweep_cfg, info, seed_offset=1000 * s))
    rs = summarize(track(stopgrad, sweep_cfg, info, seed_offset=1000 * s))
    rows.append((rb, rs))
    print(f"seed {s}: full-PC res_floor={rb['res_floor']:.3f} nov={rb['nov']:.3f} occ={rb['occ']:.3f}  |  "
          f"stop-grad res_floor={rs['res_floor']:.3f} nov={rs['nov']:.3f} occ={rs['occ']:.3f} margin={rs['margin']:.2f}")

fb = float(np.mean([r[0]["res_floor"] for r in rows]))
fs = float(np.mean([r[1]["res_floor"] for r in rows]))
print(f"\nmean residual floor: full-PC {fb:.3f} vs stop-grad {fs:.3f}  ->  stop-grad {fs / fb:.2f}x looser")
print(f"all stop-grad seeds stable (margin>0): {all(r[1]['margin'] > 0 for r in rows)}")

# %% [markdown]
# ## Verdict
# * **The transfer mechanism survives.** All P directions get transferred (every ‖M_S m_p‖
#   drops from 1.0 toward its floor), the novelty staircase falls, manifold occupancy stays
#   ~1 (no noise-chasing), and the student's inference never blows up. The prioritized-transfer
#   drive does not depend on the backward Wᵀε term — that term lives only in the *self-relaxation*,
#   while the teacher's reversed-precision drive (π_ST·ε_TS, π_ST<0) and the student's input pull
#   (-π_TS ε_TS, the identity T↔S interface) are untouched. So the answer to "does our dynamic
#   work under stop-grad" is **yes**.
# * **The crux (M_S stability) — refined.** S = MᵀM is PSD by construction; M is not, and the
#   symmetric part of M_S does dip slightly negative under the transfer drive (so -M_S x alone is
#   briefly not a strict contraction). But that is the wrong bar: the student's perception relaxes
#   under -(π_TS I + π_S M_S), stable iff Re λ(M_S) > -π_TS/π_S. The +π_TS I input cushion absorbs
#   the dip with a large margin, so the risk is real in principle but does not bite at π_TS > π_S.
# * **The price (this is the real finding).** Dropping the symmetric MᵀM structure makes
#   -M_S a *weaker, non-PSD* attractor than -S_S, so the transferred memories are held less
#   tightly: the stop-grad residual floor sits ~2x above the full-PC baseline's (seed sweep
#   above), and the restricted novelty operator UᵀN_S^sg U turns non-symmetric — its real
#   eigenvalues can dip below zero and pick up small imaginary parts. We also lose the exact
#   variational story: -M x is not ∇F (it is a frozen-target descent to the same fixed point),
#   and the stability guard rescales σ²_min → σ_min.
# * **Watch item (1 seed).** At low noise the stop-grad model covered the subspace less
#   reliably than full-PC (one direction lagged) — consistent with a weaker attractor leaning
#   harder on exploration noise. Worth a proper seed sweep before claiming it as systematic.
