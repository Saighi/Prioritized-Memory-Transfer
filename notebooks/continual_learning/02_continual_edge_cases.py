# %% [markdown]
# # Continual learning (interleaved) — edge cases & ablations
#
# Run cell-by-cell (**`pytorch`** kernel). Companion to `01_continual_single_stream.py`. Runs are kept
# small/fast; the point is the qualitative behavior of the interleaved consolidation loop.

# %% imports, path setup, run helper
import os

import numpy as np
import torch


from pmt import ContinualConfig, ContinualLearner

SHOW = os.environ.get("PMT_NO_SHOW") != "1"
torch.manual_seed(0)

FAST = dict(pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9, eta=0.06, sigma_xi=0.12,
            consolidate_bouts=10, download_bouts=4, bout_steps=3000, mode="adiabatic", seed=0)


def run(**kw):
    cfg = ContinualConfig(**{**FAST, **kw})
    L = ContinualLearner(cfg)
    H = L.run(progress=False).to_numpy()
    return L, H


def worst_old_residual(H):
    """Worst retention residual among all-but-the-last memory (old-memory retention)."""
    return float(H["final_storage_residual"][:-1].max())


# %% [markdown]
# ## Storage rehearsal prevents interference — and for CORRELATED memories it is load-bearing
# During consolidation the Storage is interleaved in (rehearsed) so its correlated old memories keep
# being re-nulled as the new one is learned. Turn that off (`storage_support=False`, i.e. rehearse the
# Buffer only) and the correlated old memories degrade — the crosstalk is never cancelled.

# %%
_, H_on = run(d=20, n_memories=3, memory_kind="correlated", storage_support=True)
_, H_off = run(d=20, n_memories=3, memory_kind="correlated", storage_support=False)
print(f"correlated stream, worst OLD-memory Storage residual:")
print(f"  with Storage rehearsal    : {worst_old_residual(H_on):.3f}   n_retained={H_on['n_retained'].tolist()}")
print(f"  without (buffer-only)      : {worst_old_residual(H_off):.3f}   n_retained={H_off['n_retained'].tolist()}")
assert worst_old_residual(H_off) > 1.4 * worst_old_residual(H_on), \
    "for correlated memories, dropping Storage rehearsal should degrade the old memories"
print("→ (the effect grows as dimensions get tighter / memories more correlated)")

# %% [markdown]
# ## Capacity: a stream longer than `d−1`
# Storage is a zero-diagonal covPCN, so it can null at most `d−1` directions. Stream more and it
# saturates; further memories cannot all be retained.

# %%
_, Hcap = run(d=8, n_memories=11, memory_kind="random",
              consolidate_bouts=8, download_bouts=3, bout_steps=2500)
print(f"d=8 (capacity d-1=7): memories retained per cycle = {Hcap['n_retained'].tolist()}")
print(f"  retained fraction end: {Hcap['retained_frac'][-1]:.2f}")
assert Hcap["n_retained"].max() <= 7, "cannot retain more than capacity d-1"

# %% [markdown]
# ## Correlated vs near-orthogonal streams
# Correlated memories generate crosstalk that the interleaving must cancel (Storage rehearsal
# matters); near-orthogonal memories barely interfere (warm-start alone nearly suffices).

# %%
_, Hc = run(d=24, n_memories=4, memory_kind="correlated")
_, Hr = run(d=24, n_memories=4, memory_kind="random")
print(f"correlated : n_retained={Hc['n_retained'].tolist()}  worst-old={worst_old_residual(Hc):.3f}")
print(f"random     : n_retained={Hr['n_retained'].tolist()}  worst-old={worst_old_residual(Hr):.3f}")

# %% [markdown]
# ## Warm-start vs reset synthesis
# Warm-starting Synthesis from Storage lets consolidation focus on the new memory; resetting it each
# cycle re-learns the union from the interleaved rehearsal of both. Both retain the stream.

# %%
_, Hwarm = run(d=20, n_memories=3, memory_kind="correlated", synthesis_warm_start=True)
_, Hreset = run(d=20, n_memories=3, memory_kind="correlated", synthesis_warm_start=False)
print(f"warm-start : final storage residuals {[round(x,3) for x in Hwarm['final_storage_residual'].tolist()]}")
print(f"reset      : final storage residuals {[round(x,3) for x in Hreset['final_storage_residual'].tolist()]}")

# %% [markdown]
# ## Summary
# Interleaved rehearsal makes the loop a genuine *continual* learner for **correlated** streams: the
# single-memory Buffer never needs to roam, and rehearsing Storage supplies the co-excitation that
# cancels crosstalk (load-bearing for correlated memories, unlike the orthogonal case). Capacity still
# follows `d−1`.
