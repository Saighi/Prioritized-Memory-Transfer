"""Fast self-check of the pmt core: faithfulness invariants, gradient/circulation checks,
and a short adiabatic run that should show the novelty spectrum falling. Run with:
    conda run -n pytorch python tests/smoke_test.py
"""
import torch

from pmt import ModelConfig, SimConfig, build_system, simulate
from pmt import diagnostics as dg

cfg = ModelConfig(d=40, P=5, seed=1)
model, info = build_system(cfg)
print(f"sigma2_min      = {info['sigma2_min']:.4f}")
print(f"manifold_dim    = {info['manifold_dim']} (expected P={cfg.P})")
print(f"memory_residual = {info['memory_residual']:.2e}  (want ~0)")
print(f"pi_ST           = {info['pi_ST']:.4f}  (reversed: <0)  "
      f"|pi_ST|<guard_safe={info['guard_safe']:.4f}? {info['pi_ST_ok']}  "
      f"(aligned bound {info['guard_scalar']:.4f}? {info['pi_ST_ok_aligned']})")

# --- zero-diagonal invariant ---
assert float(torch.diagonal(model.W_T).abs().max()) < 1e-12, "W_T has nonzero diagonal"
assert info["manifold_dim"] == cfg.P, "manifold dim != P for orthonormal patterns"
assert info["memory_residual"] < 1e-4, "M_T m_p not ~0"

# --- gradient check: perception & learning are gradient descent on F_S ---
e_xS, e_WS = dg.check_gradients(model)
print(f"grad err: xS={e_xS:.2e}  WS={e_WS:.2e}  (want ~1e-10)")
assert e_xS < 1e-8 and e_WS < 1e-8, "gradient mismatch"

# --- Corollary 1 (surprise identity): settled F_S == (pi_TS/2) x_T^T N_S x_T ---
e_id = dg.surprise_identity_error(model)
print(f"surprise identity err = {e_id:.2e}  (want ~1e-15)")
assert e_id < 1e-10, "surprise identity (Corollary 1) violated"

# --- circulation: the saddle is exact (circulation ~0) only when pi_ST == -pi_TS ---
c_auto = dg.circulation(model)
model_eq, _ = build_system(ModelConfig(d=40, P=5, seed=1, exact_saddle=True))
c_eq = dg.circulation(model_eq)
print(f"circulation: auto(pi_ST!=-pi_TS)={c_auto:.4f}   pi_ST=-pi_TS={c_eq:.2e}")
assert c_eq < 1e-6 < c_auto, "circulation should vanish only at pi_ST=-pi_TS"

# --- adiabatic run: TOTAL novelty should fall toward a floor, W_S should approach W_T.
# (The spectrum *max* is the envelope and only drops when the LAST direction is learned;
#  total novelty = sum of the spectrum is the right transfer signal.) ---
sim = SimConfig(n_steps=40000, dt=0.5, mode="adiabatic", record_every=200, progress=False)
hist = simulate(model, sim, info)
H = hist.to_numpy()
tot0 = float(H["novelty_spec"][0].sum())
tot1 = float(H["novelty_spec"][-1].sum())
print(f"total novelty: start={tot0:.3f}  end={tot1:.3f}")
print(f"||W_S-W_T||:   start={H['WS_dist'][0]:.3f}  end={H['WS_dist'][-1]:.3f}")
assert float(torch.diagonal(model.W_S).abs().max()) < 1e-10, "W_S diagonal drifted"
assert tot1 < 0.25 * tot0, "total novelty did not fall toward floor"
assert H["WS_dist"][-1] < 0.5 * H["WS_dist"][0], "W_S did not approach W_T"
assert dg.surprise_identity_error(model) < 1e-10, "surprise identity violated after learning"

print("\nSMOKE TEST PASSED")
