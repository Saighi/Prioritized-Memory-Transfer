# %% [markdown]
# # Interleaved subspace addition — building the combined subspace one teacher at a time
#
# Run cell-by-cell (**`pytorch`** kernel). Two frozen teachers hold one memory each; a plastic
# synthesis network learns **both** by rehearsing **one teacher per replay bout**, alternating.
# Each bout is a plain two-network reversed-precision transfer, so the covariance the synthesis
# learns is `Σ = p₁Σ₁ + p₂Σ₂` — no teacher cross-term — and it is full-rank on
# `𝒰_Σ = 𝒰₁+𝒰₂` even for single-memory, correlated teachers.
#
# The **back-and-forth** is the co-excitation that cancels crosstalk: rehearse T1 → nulls `m₁`,
# bumps `m₂`; rehearse T2 → nulls `m₂`, bumps `m₁`; the bumps shrink to zero as both are jointly
# nulled (interleaved rehearsal — the standard cure for catastrophic forgetting).

# %% imports & path setup
import os

import torch


from src import InterleavedConfig, SimConfig, build_interleaved_synthesis, simulate_interleaved
from src import viz_interleaved as vi

SHOW = os.environ.get("PMT_NO_SHOW") != "1"
torch.manual_seed(0)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# %% configure: two CORRELATED SINGLE memories (the hard case for any summed merge)
cfg = InterleavedConfig(
    d=32,
    rank1=1, rank2=1, geometry="oblique", principal_angle=0.5,   # two memories at ~60°, correlated
    pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9, eta=0.05,
    sigma_xi1=0.1, sigma_xi2=0.1, seed=0,
)
macro, info = build_interleaved_synthesis(cfg)
cos = float((info.U1[:, 0] * info.U2[:, 0]).sum())
print(f"two single memories, |cos(m1,m2)| = {abs(cos):.2f}  |  r_Σ = {info.r_Sigma} (want 2)")

# %% run the interleaved merge (alternating T1 / T2 replay bouts)
sim = SimConfig(n_steps=90000, dt=0.5, mode="adiabatic", bout_steps=3000, progress=False)
hist = simulate_interleaved(macro, sim, info)
H = hist.to_numpy()
print(f"bouts = {len(H['bout'])}")
print(f"‖M_S U1‖ : {H['resid1'][0]:.3f} -> {H['resid1'][-1]:.3f}")
print(f"‖M_S U2‖ : {H['resid2'][0]:.3f} -> {H['resid2'][-1]:.3f}")
print(f"union deficit ‖M_S U_Σ‖² : {H['union_deficit'][0]:.3f} -> {H['union_deficit'][-1]:.4f}")
assert H["union_deficit"][-1] < 0.1, "interleaving should null both memories"

# %% the crosstalk-cancellation dashboard (sawtooth + union deficit -> 0)
fig = vi.dashboard(hist, info)
fig   # displays inline

# %% interactive crosstalk plot
if SHOW:
    vi.crosstalk_plotly(hist).show()

# %% [markdown]
# ## What to look for
# - **Sawtooth (left panel)**: each shaded bout rehearses one teacher; its residual drops while the
#   other's **bumps up** — that bump is the crosstalk from learning a correlated memory. The bumps
#   **decay** bout after bout: the back-and-forth cancels the crosstalk (the covariance inversion
#   gets the co-excitation it needs).
# - **Union deficit → 0 (right panel)**: the combined subspace `𝒰_Σ` is actually constructed.
#
# ## Try next (`02_interleaved_edge_cases.py`)
# - Vary the correlation (orthogonal → oblique → near-parallel): flat convergence vs pronounced
#   sawtooth vs slow, ill-conditioned merging.
# - Multi-memory teachers — interleaving handles any mix of ranks.
