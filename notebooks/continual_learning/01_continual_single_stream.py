# %% [markdown]
# # Continual learning — buffer → synthesis → storage (interleaved consolidation)
#
# Run cell-by-cell (**`pytorch`** kernel). A stream of **correlated** memories arrives one at a time.
# Three predictive-coding networks consolidate them so the long-term **Storage** ends up holding the
# *whole* stream — even though the fast **Buffer** can only ever hold the latest memory.
#
# **Each cycle:**
# 1. **write** the new memory into the Buffer with one-shot covPCN (this overwrites the previous one).
#    The Buffer (hippocampus) is the *only* network that ever learns from an actual memory.
# 2. **consolidate** `Buffer + Storage → Synthesis` by **interleaved rehearsal**: the Synthesis is
#    coupled to ONE network per replay bout, alternating Buffer (the new memory) and Storage (the old
#    ones). Never summing them removes the cross-term that pins the single-memory Buffer, and the
#    back-and-forth cancels the crosstalk between correlated memories.
# 3. **download** `Synthesis → Storage` so Storage absorbs the union.

# %% imports & path setup
import os

import torch


from src import ContinualConfig, ContinualLearner
from src import viz_continual as vc
from src import viz_interleaved as vi

SHOW = os.environ.get("PMT_NO_SHOW") != "1"
torch.manual_seed(0)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# %% configure & build the continual learner (correlated memories by default)
cfg = ContinualConfig(
    d=32, n_memories=5, memory_kind="correlated",
    pi_teacher=1.0, pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9,
    tau_teacher=10.0, tau_S=1.0, eta=0.05, sigma_xi=0.12, r=1.0,
    consolidate_bouts=12, download_bouts=5, bout_steps=4000, mode="adiabatic",
    synthesis_warm_start=True, storage_support=True, seed=0, device="cpu",
)
learner = ContinualLearner(cfg)
G = learner.memories.T @ learner.memories
off = G[~torch.eye(cfg.n_memories, dtype=torch.bool)]
print(f"streaming {learner.n_memories} memories in d={cfg.d}  "
      f"(avg |pairwise cosine| = {float(off.abs().mean()):.2f}, correlated)")

# %% run the stream (write → interleaved consolidate → download) — the slow cell
hist = learner.run(progress=True)
H = hist.to_numpy()
print(f"memories retained in Storage per cycle : {H['n_retained'].tolist()}   (grows 1..{learner.n_memories})")
print(f"final Storage residuals ‖M_Z mᵢ‖: {[round(x, 3) for x in H['final_storage_residual'].tolist()]}")
print(f"final baseline (buffer-only)    : {[round(x, 3) for x in H['final_baseline_residual'].tolist()]}")
print(f"fraction retained  — continual  : {H['retained_frac'][-1]:.2f}   baseline: {H['baseline_retained_frac'][-1]:.2f}")
assert H["n_retained"][-1] == learner.n_memories, "storage should retain every memory"

# %% static dashboard — retention heatmaps + subspace growth + retained fraction
fig = vc.dashboard(hist, cfg)
fig   # displays inline

# %% the crosstalk of the LAST consolidation (interleaved rehearsal in action)
fig_ct = vi.dashboard(learner.last_consolidation_trace)
fig_ct

# %% interactive: Storage retention heatmap over the stream
if SHOW:
    vc.retention(hist).show()

# %% [markdown]
# ## What to look for
# - **Storage retention heatmap**: every memory stays dark (retained) once added — Storage keeps
#   **all** correlated memories; the **buffer-only baseline** keeps only the diagonal (the latest) =
#   catastrophic forgetting.
# - **Consolidation crosstalk (sawtooth)**: within a single consolidation, rehearsing the Buffer nulls
#   the new memory but bumps the old ones (correlated crosstalk); rehearsing Storage corrects them; the
#   bumps shrink — interleaved rehearsal cancels the crosstalk while assembling the union.
#
# This is why the loop handles *correlated* streams that the additive-sum consolidation could not: the
# single-memory Buffer never has to roam, and the back-and-forth supplies the co-excitation the
# covariance inversion needs.
#
# ## Try next (`02_continual_edge_cases.py`)
# - `storage_support=False` → no Storage rehearsal → correlated old memories degrade.
# - `n_memories > d−1` → Storage saturates at capacity.
# - `memory_kind="random"` → near-orthogonal memories: little crosstalk, flatter consolidation.
