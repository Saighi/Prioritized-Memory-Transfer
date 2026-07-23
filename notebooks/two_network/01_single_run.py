# %% [markdown]
# # Prioritized memory transfer — single run & dynamics
#
# Run cell-by-cell in VS Code (select the **`pytorch`** conda interpreter as the kernel).
# This builds the two-population predictive-coding system — teacher **T** (hippocampus-like,
# frozen, higher in the hierarchy) above student **S** (cortex-like, plastic) — runs one full
# transfer in the **sleep/replay** regime, and visualizes the in-time dynamics (static
# dashboard + interactive plotly).
#
# Faithful to `two_population_memory_transfer_model.md`; the math is checked in
# `analysis_stress_test.md`. Remember: the *linear* model does **prioritized subspace
# consolidation** — the staircase has ~`effective-rank` steps (= `P` for orthonormal patterns).

# %% imports & path setup
import os

import torch

# make `import prioritized_memory_transfer` work whether cwd is the repo root or notebooks/

from prioritized_memory_transfer import ModelConfig, SimConfig, build_system, simulate
from prioritized_memory_transfer import diagnostics as dg
from prioritized_memory_transfer import viz_static as vs
from prioritized_memory_transfer import viz_interactive as vi

SHOW = os.environ.get("PMT_NO_SHOW") != "1"   # set PMT_NO_SHOW=1 to run headless
torch.manual_seed(0)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# %% configure & build the system
model_cfg = ModelConfig(
    d=48, P=3,
    pi_TS=1.0, pi_S=0.5,        # precision guard pi_TS > pi_S
    pi_ST="auto", pi_ST_safety=0.5,   # pi_ST = 0.5 * spectral gap (well inside the guard)
    tau_T=10.0, tau_S=1.0, eta=0.005,  # tau_S << tau_T << 1/eta
    sigma_xi=0.05, r0=1.0,
    pattern_kind="orthonormal",
    seed=0, device="cpu",
)
model, info = build_system(model_cfg)

print(f"spectral gap σ²_min   : {info.sigma2_min:.4f}")
print(f"manifold dim (eff rank): {info.manifold_dim}  (= P for orthonormal patterns)")
print(f"memory residual maxₚ‖M_T mₚ‖: {info.memory_residual:.2e}")
print(f"pi_ST                 : {info.pi_ST:.4f}  (reversed / negative ⇒ sleep-replay)")
print(f"stability guards      : |π_ST|<σ²_min? {abs(info.pi_ST) < info.guard_safe} | "
      f"|π_ST|<scalar({info.guard_scalar:.3f})? {info.pi_ST_ok_aligned}")
print(f"precision guard pi_TS>pi_S: {info.precision_ok}")

# %% faithfulness self-checks (cheap; assert the invariants)
assert float(torch.diagonal(model.W_T).abs().max()) < 1e-12, "W_T diagonal not zero"
assert info.memory_residual < 1e-4, "patterns not in ker M_T"

e_xS, e_WS = dg.check_gradients(model)
print(f"gradient check: err_xS={e_xS:.1e}  err_WS={e_WS:.1e}  (perception=-∇F_S, learning=-∇_W F_S)")
assert e_xS < 1e-8 and e_WS < 1e-8

c = dg.circulation(model)
print(f"circulation (saddle): {c:.3f}   (≈0 only when π_ST=−π_TS; here π_ST≠−π_TS so it's |π_ST+π_TS|·√d)")

# %% run the transfer (full mode: S is integrated, so its relaxation is real)
# pi_ST<0 (the reversed precision) ⇒ this is the sleep / replay regime.
sim_cfg = SimConfig(n_steps=5000, dt=0.5, mode="full", record_every=100,
                    n_weight_snapshots=6, progress=True)
hist = simulate(model, sim_cfg, info)

H = hist.to_numpy()
print(f"total novelty: {H['novelty_spec'][0].sum():.3f} -> {H['novelty_spec'][-1].sum():.3f}")
print(f"||W_S-W_T||_F: {H['WS_dist'][0]:.3f} -> {H['WS_dist'][-1]:.3f}")
assert float(torch.diagonal(model.W_S).abs().max()) < 1e-10, "W_S diagonal drifted (autapse!)"

# %% static dashboard — the 6-panel overview of the dynamics
fig = vs.dashboard(hist, model, info)
fig   # displays inline

# %% W_S converging to W_T (zero diagonal throughout)
fig_w = vs.weight_snapshots(hist, model)
fig_w

# %% interactive: novelty-spectrum staircase
if SHOW:
    vi.staircase(hist).show()

# %% interactive: replay raster
if SHOW:
    vi.raster(hist).show()

# %% interactive: x_T trajectory in memory space (press ▶ or drag the time slider)
if SHOW:
    vi.trajectory_3d(hist, model, info, basis="patterns").show()

# %% [markdown]
# ## Try next
# - **Adiabatic mode** (faster, cleaner staircase): set `mode="adiabatic"` in `SimConfig`.
# - **Binary novelty**: `SimConfig(..., pretrain_subset=[0, 1, 2])` — S already knows those, so
#   the held-out memories transfer first (the novel-before-known result).
# - **Exact saddle**: `ModelConfig(..., exact_saddle=True)` → `π_ST = −π_TS` → `circulation`≈0.
# - **Wake / recall**: build with `pi_ST > 0` (an ordinary, un-reversed precision) → both
#   populations descend; there is no transfer drive, so the staircase stays flat.
# - **Subspace test**: `pattern_kind="correlated"` → effective rank < P, so the staircase
#   should show **fewer than P** steps.
