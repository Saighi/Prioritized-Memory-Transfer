"""Self-check of the composable engine and the additive three-network model. Run with:
    conda run -n pytorch python tests/macro_test.py

Covers: (A) the engine reproduces the two-population equations exactly (the one-teacher limit,
spec section 6); (B) additive combined-subspace transfer; (C) clean self-termination and the
separate-error control's persistent residual (spec section 3); (D) overlap counted once (section 13).
"""
import torch

from src import (AdditiveSynthesisConfig, ModelConfig, SimConfig, build_additive_synthesis,
                 build_system, fwd, bwd, outer, simulate)

torch.manual_seed(0)

# ----------------------------------------------------------------------------- (A) engine == 2-pop
cfg = ModelConfig(d=32, P=4, seed=1)
model, info = build_system(cfg)
d = model.d
model.x_T = torch.randn(d, dtype=model.dtype)
model.x_S = torch.randn(d, dtype=model.dtype)
model.W_S = model._S.W  # keep zero-diagonal init; give it some structure:
model.W_S = torch.diag_embed(torch.zeros(d, dtype=model.dtype))  # start at 0 (M_S = I)

# hand-computed two-population rates (the documented equations), compared to the engine's assembly
M_T, M_S = model.M_T, model.M_S
eps_T = fwd(M_T, model.x_T)
eps_S = fwd(M_S, model.x_S)
eps_TS = model.x_S - model.x_T
rate_T_hand = (-model.pi_T * bwd(M_T, eps_T) + model.pi_ST * eps_TS) / model.tau_T
rate_S_hand = (-model.pi_TS * eps_TS - model.pi_S * bwd(M_S, eps_S)) / model.tau_S
rate_W_hand = model.eta * model.pi_S * outer(eps_S, model.x_S)

e_T = float((model.rate_x_T_det() - rate_T_hand).norm())
e_S = float((model.rate_x_S() - rate_S_hand).norm())
e_W = float((model.rate_W_S() - rate_W_hand).norm())
e_steady = float((model.solve_xS_steady(model.x_T)
                  - model.macro.solve_steady("S")).norm())
print(f"(A) engine vs 2-pop equations:  rate_T={e_T:.1e}  rate_S={e_S:.1e}  "
      f"rate_W={e_W:.1e}  steady={e_steady:.1e}")
assert max(e_T, e_S, e_W, e_steady) < 1e-10, "engine does not reproduce the two-population equations"

# ------------------------------------------------------------------------- (D) overlap counted once
acfg = AdditiveSynthesisConfig(d=32, rank1=3, rank2=3, geometry="shared", overlap=1,
                               eta=0.05, rho_safety=0.9, sigma_xi1=0.1, sigma_xi2=0.1, seed=0)
macro, ainfo = build_additive_synthesis(acfg)
print(f"(D) rank1={ainfo['rank1']} rank2={ainfo['rank2']} overlap={ainfo['overlap']} "
      f"r_Sigma={ainfo['r_Sigma']}  (expect r_Sigma = 3+3-1 = 5)")
assert ainfo["overlap"] == 1 and ainfo["r_Sigma"] == 5, "subspace-sum rank/overlap wrong"

# --------------------------------------------------------------------- (B) additive transfer works
sim = SimConfig(n_steps=80000, dt=0.5, mode="adiabatic", record_every=1000,
                n_weight_snapshots=4, progress=False)
H = simulate(macro, sim, ainfo).to_numpy()
E0, E1 = float(H["E_Sigma"][0]), float(H["E_Sigma"][-1])
nov0, nov1 = float(H["novelty_spec"][0].sum()), float(H["novelty_spec"][-1].sum())
print(f"(B) E_Sigma {E0:.2f}->{E1:.3f} | novelty {nov0:.2f}->{nov1:.3f} | "
      f"E1 {float(H['E1'][-1]):.3f} E2 {float(H['E2'][-1]):.3f}")
assert E1 < 0.4 * E0, "combined-subspace deficit did not fall"
assert nov1 < 0.5 * nov0, "restricted novelty did not fall"
assert float(H["E1"][-1]) < 0.5 * float(H["E1"][0]), "teacher-1 deficit did not fall"
assert float(H["E2"][-1]) < 0.5 * float(H["E2"][0]), "teacher-2 deficit did not fall"
eps_additive = float(H["eps_Sigma_norm"][-1])

# ----------------------------------------------------- (C) separate-error control keeps a residual
scfg = AdditiveSynthesisConfig(d=32, rank1=3, rank2=3, geometry="shared", overlap=1,
                               eta=0.05, rho_safety=0.9, sigma_xi1=0.1, sigma_xi2=0.1, seed=0,
                               separate_errors=True)
smacro, sinfo = build_additive_synthesis(scfg)
Hs = simulate(smacro, sim, sinfo).to_numpy()
eps_separate = float(Hs["eps_Sigma_norm"][-1])
print(f"(C) terminal interface residual:  additive={eps_additive:.4f}  separate={eps_separate:.4f}")
assert eps_separate > 3 * eps_additive, "separate-error control should keep a large interface residual"

print("\nMACRO TEST PASSED")
