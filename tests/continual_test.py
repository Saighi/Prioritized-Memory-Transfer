"""Self-check of the continual-learning loop (buffer -> synthesis -> storage), interleaved
consolidation. Run with:
    conda run -n pytorch --no-capture-output python tests/continual_test.py

Covers: (1) Storage accumulates a CORRELATED stream (every memory retained, count grows);
(2) the buffer-only baseline catastrophically forgets (keeps only the latest); (3) for correlated
memories, rehearsing Storage during consolidation is load-bearing — dropping it degrades the old
memories (the crosstalk is not cancelled).
"""
import torch

from pmt import ContinualConfig, ContinualLearner

torch.manual_seed(0)
TOL = 0.3

# ------------------------------------------------------- (1)+(2) accumulation vs forgetting (correlated)
cfg = ContinualConfig(d=20, n_memories=3, memory_kind="correlated", eta=0.06, rho_safety=0.9,
                      sigma_xi=0.12, consolidate_bouts=10, download_bouts=4, bout_steps=3000,
                      mode="adiabatic", seed=0)
L = ContinualLearner(cfg)
H = L.run(progress=False).to_numpy()
G = L.memories.T @ L.memories
avg_cos = float(G[~torch.eye(cfg.n_memories, dtype=torch.bool)].abs().mean())

store_res = H["final_storage_residual"]
base_res = H["final_baseline_residual"]
print(f"(1) correlated stream (avg |cos|={avg_cos:.2f})  memories retained/cycle: {H['n_retained'].tolist()}")
print(f"    final storage residual: {[round(x,3) for x in store_res.tolist()]}  (want all < {TOL})")
print(f"(2) final baseline resid  : {[round(x,3) for x in base_res.tolist()]}  (want only the last < {TOL})")
assert avg_cos > 0.3, "the default stream should be genuinely correlated"
assert H["n_retained"][-1] == cfg.n_memories, "storage did not retain every memory"
assert float(store_res.max()) < TOL, "storage did not retain every memory"
assert float(base_res[:-1].min()) > TOL and float(base_res[-1]) < TOL, \
    "buffer-only baseline should keep ONLY the latest memory"

# ------------------------------------------ (3) Storage rehearsal is load-bearing for correlated memories
def worst_old_residual(storage_support: bool) -> float:
    c = ContinualConfig(d=20, n_memories=3, memory_kind="correlated", eta=0.06, rho_safety=0.9,
                        sigma_xi=0.12, consolidate_bouts=10, download_bouts=4, bout_steps=3000,
                        mode="adiabatic", seed=0, storage_support=storage_support)
    res = ContinualLearner(c).run(progress=False).to_numpy()["final_storage_residual"]
    return float(res[:-1].max())          # worst residual among the OLD memories

with_support = worst_old_residual(True)
without_support = worst_old_residual(False)
print(f"(3) worst OLD-memory residual:  with Storage rehearsal={with_support:.3f}  "
      f"without={without_support:.3f}")
assert without_support > 1.5 * with_support, \
    "for correlated memories, dropping Storage rehearsal should degrade the old memories"

print("\nCONTINUAL TEST PASSED")
