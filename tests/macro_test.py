"""Self-check of the composable engine (`src.macro`). Run with:
    conda run -n pytorch python tests/macro_test.py

The engine must reproduce the documented two-population equations exactly (the one-teacher
limit): assembled rates, Hebbian update, and the adiabatic steady state.
"""
import torch

from src import ModelConfig, build_system, fwd, bwd, outer

torch.manual_seed(0)

cfg = ModelConfig(d=32, P=4, seed=1)
model, info = build_system(cfg)
d = model.d
model.x_T = torch.randn(d, dtype=model.dtype)
model.x_S = torch.randn(d, dtype=model.dtype)
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
print(f"engine vs 2-pop equations:  rate_T={e_T:.1e}  rate_S={e_S:.1e}  "
      f"rate_W={e_W:.1e}  steady={e_steady:.1e}")
assert max(e_T, e_S, e_W, e_steady) < 1e-10, "engine does not reproduce the two-population equations"

print("\nMACRO TEST PASSED")
