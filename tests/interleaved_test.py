"""Self-check of the interleaved merge (`src.interleaved`). Run with:
    conda run -n pytorch --no-capture-output python tests/interleaved_test.py

Core claim: for two CORRELATED SINGLE-memory teachers, interleaving builds the combined subspace
(both memories nulled) while the additive SUM collapses onto the blend and fails. The crosstalk
sawtooth is present for correlated memories and ~absent for orthogonal ones.
"""
import numpy as np
import torch

from src import (AdditiveSynthesisConfig, SimConfig, build_interleaved_synthesis,
                 simulate_interleaved, build_additive_synthesis, simulate_additive)

torch.manual_seed(0)


def crosstalk_gap(H):
    """Second-half mean of (inactive residual - active residual). The active teacher was just
    rehearsed (at the learned floor); the inactive one is disturbed by it. For orthogonal memories
    there is no disturbance so the gap ~ 0; for correlated memories the gap is the crosstalk."""
    a = H["active"]
    active_res = np.where(a == 0, H["resid1"], H["resid2"])       # the one just rehearsed
    inactive_res = np.where(a == 0, H["resid2"], H["resid1"])     # the one disturbed
    gap = (inactive_res - active_res)
    return float(gap[len(gap) // 2:].mean())


# --- two correlated single memories ---
cfg = AdditiveSynthesisConfig(d=32, rank1=1, rank2=1, geometry="oblique", principal_angle=0.5,
                              pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9, eta=0.05,
                              sigma_xi1=0.1, sigma_xi2=0.1, seed=0)
sim = SimConfig(n_steps=72000, dt=0.5, mode="adiabatic", bout_steps=3000, progress=False)

macro, info = build_interleaved_synthesis(cfg)
cos = abs(float((info["U1"][:, 0] * info["U2"][:, 0]).sum()))
Hi = simulate_interleaved(macro, sim, info).to_numpy()
print(f"correlated single memories (|cos|={cos:.2f}), r_Sigma={info['r_Sigma']}")
print(f"  interleaved: union deficit {Hi['union_deficit'][0]:.3f} -> {Hi['union_deficit'][-1]:.4f}")
assert info["r_Sigma"] == 2, "two independent memories should give r_Sigma=2"
assert Hi["union_deficit"][-1] < 0.1, "interleaving should null both correlated memories"

# --- the additive SUM fails on the identical case ---
amacro, ainfo = build_additive_synthesis(cfg)
Ha = simulate_additive(amacro, SimConfig(n_steps=72000, dt=0.5, mode="adiabatic",
                                         record_every=3000, progress=False), ainfo).to_numpy()
print(f"  additive sum: E_Sigma {Ha['E_Sigma'][0]:.3f} -> {Ha['E_Sigma'][-1]:.3f} (collapses to blend)")
assert Ha["E_Sigma"][-1] > 5 * Hi["union_deficit"][-1], "the sum should fail where interleaving succeeds"

# --- sawtooth present for correlated, ~absent for orthogonal ---
gap_corr = crosstalk_gap(Hi)
macro_o, info_o = build_interleaved_synthesis(
    AdditiveSynthesisConfig(d=32, rank1=1, rank2=1, geometry="orthogonal",
                            pi_S=0.5, pi_I=1.0, rho="auto", rho_safety=0.9, eta=0.05,
                            sigma_xi1=0.1, sigma_xi2=0.1, seed=0))
Ho = simulate_interleaved(macro_o, sim, info_o).to_numpy()
gap_orth = crosstalk_gap(Ho)
print(f"  crosstalk gap: correlated={gap_corr:.3f}  orthogonal={gap_orth:.3f}")
assert gap_corr > gap_orth + 0.02, "correlated memories should show a clear crosstalk sawtooth"

print("\nINTERLEAVED TEST PASSED")
