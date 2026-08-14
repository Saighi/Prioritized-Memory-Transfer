"""Self-check of the optional unit nonlinearity (`ModelConfig.activation`). Run with:
    conda run -n pytorch python tests/nonlinear_test.py

Four things must hold:
  1. `activation=None` is byte-identical to `activation="identity"` (no silent drift in the
     linear model the paper analyses).
  2. The ReLU rates match the hand-written equations `eps = x - W f(x)`,
     `dx/dt ∋ -pi J^T eps` with `J = I - W diag(f'(x))`, `dW/dt = eta pi eps f(x)^T`.
  3. Nonnegative patterns are exact zero-error states of the ReLU network, and adiabatic
     elimination of the nonlinear student is REFUSED rather than silently approximated.
  4. The transfer mechanism still works: total novelty collapses and W_S approaches W_T.
"""
import warnings

import torch

from prioritized_memory_transfer import ModelConfig, SimConfig, build_system, simulate, bwd, fwd, outer
from prioritized_memory_transfer import diagnostics as dg

torch.manual_seed(0)

BASE = dict(d=32, P=3, pi_TS=1.0, pi_S=0.5, pi_ST="auto", pi_ST_safety=0.5,
            tau_T=10.0, tau_S=1.0, eta=0.005, sigma_xi=0.05, r0=1.0, seed=1)

# --- 1. None vs "identity": the linear path must be untouched ---------------------------------
m_none, i_none = build_system(ModelConfig(**BASE, pattern_kind="orthonormal", activation=None))
m_id, i_id = build_system(ModelConfig(**BASE, pattern_kind="orthonormal", activation="identity"))
assert torch.equal(m_none.W_T, m_id.W_T), "identity changed the fitted teacher"
sim = SimConfig(n_steps=2000, dt=0.5, mode="full", record_every=100, progress=False)
H_none = simulate(m_none, sim, i_none).to_numpy()
H_id = simulate(m_id, sim, i_id).to_numpy()
for key in ("novelty_spec", "residual", "WS_dist", "x_T"):
    assert (H_none[key] == H_id[key]).all(), f"identity diverged from None on {key}"
print("1. activation=None == 'identity': bit-identical over a 2000-step run")

# --- 2. ReLU rates vs the hand-written equations ----------------------------------------------
cfg = ModelConfig(**BASE, pattern_kind="nonneg", activation="relu")
model, info = build_system(cfg)
d = model.d
gen = torch.Generator().manual_seed(7)
model.x_T = torch.randn(d, generator=gen, dtype=model.dtype)
model.x_S = torch.randn(d, generator=gen, dtype=model.dtype)
model.W_S = torch.zeros(d, d, dtype=model.dtype)
model.W_S = model.W_S + 0.1 * torch.randn(d, d, generator=gen, dtype=model.dtype)
model.zero_diag_W_S()

W_T, W_S = model.W_T, model.W_S


def relu(x):
    return x.clamp_min(0.0)


def relu_p(x):
    return (x > 0).to(x.dtype)


eps_T = model.x_T - fwd(W_T, relu(model.x_T))
eps_S = model.x_S - fwd(W_S, relu(model.x_S))
eps_TS = model.x_S - model.x_T
# J^T eps = eps - f'(x) * (W^T eps)
selfT = eps_T - relu_p(model.x_T) * bwd(W_T, eps_T)
selfS = eps_S - relu_p(model.x_S) * bwd(W_S, eps_S)
rate_T_hand = (-model.pi_T * selfT + model.pi_ST * eps_TS) / model.tau_T
rate_S_hand = (-model.pi_TS * eps_TS - model.pi_S * selfS) / model.tau_S
rate_W_hand = model.eta * model.pi_S * outer(eps_S, relu(model.x_S))

e_epsT = float((model.eps_T() - eps_T).norm())
e_epsS = float((model.eps_S() - eps_S).norm())
e_T = float((model.rate_x_T_det() - rate_T_hand).norm())
e_S = float((model.rate_x_S() - rate_S_hand).norm())
e_W = float((model.rate_W_S() - rate_W_hand).norm())
print(f"2. relu engine vs hand equations: eps_T={e_epsT:.1e} eps_S={e_epsS:.1e} "
      f"rate_T={e_T:.1e} rate_S={e_S:.1e} rate_W={e_W:.1e}")
assert max(e_epsT, e_epsS, e_T, e_S, e_W) < 1e-12, "relu rates do not match the equations"

# the same algebra, independently checked against autograd on F_S
e_xS, e_WS = dg.check_gradients(model)
print(f"   gradient check vs autograd: err_xS={e_xS:.1e} err_WS={e_WS:.1e}")
assert e_xS < 1e-8 and e_WS < 1e-8, "relu perception/learning != -grad F_S"

# --- 3. memories are exact fixed points; adiabatic is refused ---------------------------------
assert info.patterns_fixed, "nonneg patterns should satisfy f(m_p) == m_p"
assert info.memory_residual < 1e-6, f"memories not nulled: {info.memory_residual:.2e}"
assert info.manifold_dim == cfg.P, f"manifold dim {info.manifold_dim} != P={cfg.P}"
for call in (lambda: model.solve_xS_steady(model.x_T), lambda: model.macro.solve_steady("S")):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError("adiabatic elimination of a nonlinear student must be refused")
print(f"3. memory residual={info.memory_residual:.1e}, manifold dim={info.manifold_dim}, "
      "adiabatic refused")

# relu on SIGNED patterns does NOT fix them (relu(m_p) != m_p) -> must warn, manifold collapses.
# This is the guard that keeps nonnegative codes load-bearing rather than decorative.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _, info_signed = build_system(ModelConfig(**BASE, pattern_kind="orthonormal", activation="relu"))
assert not info_signed.patterns_fixed, "relu on signed patterns should not fix them"
assert any("does not fix the stored patterns" in str(w.message) for w in caught), \
    "build_system must warn when f(m_p) != m_p"
assert info_signed.manifold_dim == 0, \
    f"the memories should be isolated (manifold dim 0, got {info_signed.manifold_dim})"
print("   relu + signed patterns: not fixed -> warned, ker(I-W_T) collapsed to {0} as documented")

# --- 4. the mechanism still works under ReLU ---------------------------------------------------
model, info = build_system(ModelConfig(**BASE, pattern_kind="nonneg", activation="relu"))
H = simulate(model, SimConfig(n_steps=20000, dt=0.5, mode="full", record_every=200,
                              progress=False), info).to_numpy()
tot0, tot1 = float(H["novelty_spec"][0].sum()), float(H["novelty_spec"][-1].sum())
print(f"4. total novelty {tot0:.3f} -> {tot1:.3f} | "
      f"||W_S-W_T|| {H['WS_dist'][0]:.3f} -> {H['WS_dist'][-1]:.3f} | "
      f"residual {H['residual'][0].round(2)} -> {H['residual'][-1].round(2)}")
assert tot1 < 0.4 * tot0, "relu: total novelty did not collapse"
assert H["WS_dist"][-1] < 0.5 * H["WS_dist"][0], "relu: W_S did not approach W_T"
assert (H["residual"][-1] < 0.5).all(), "relu: some memory was never consolidated"
assert float(torch.diagonal(model.W_S).abs().max()) < 1e-10, "W_S diagonal drifted (autapse!)"
assert bool(torch.isfinite(model.W_S).all()), "W_S blew up"

# --- 5. the exact manifold spectrum agrees with a brute-force sample of the cone ---------------
# F_S(Mc) = 0.5 pi_S c^T G c must be the ACTUAL energy at manifold points, and the pencil
# eigenvalues must bracket the surprise density there.
spec = dg.manifold_surprise_spectrum(model, model.patterns)
gen = torch.Generator().manual_seed(3)
c = torch.rand(200, cfg.P, generator=gen, dtype=model.dtype)        # c >= 0: inside the cone
X = (model.patterns @ c.T).T                                        # (200, d) manifold states
F_direct = 0.5 * model.pi_S * (X - X @ model.W_S.T).pow(2).sum(-1)  # f(X)=X on the cone
density = F_direct / X.pow(2).sum(-1)
e_bracket = float((density.max() - spec.max()).clamp_min(0)) + float((spec.min() - density.min()).clamp_min(0))
print(f"5. manifold spectrum {spec.numpy().round(5)} brackets sampled density "
      f"[{density.min():.5f}, {density.max():.5f}]  slack={e_bracket:.1e}")
assert e_bracket < 1e-9, "pencil eigenvalues do not bracket the true surprise density on the cone"
assert torch.allclose(spec[0] * 0 + 0.5 * model.pi_S * model.pattern_residual()[0] ** 2,
                      model.pattern_energy()[0]), "pattern_energy inconsistent with the residual"

# --- 6. recall probe: clamped completion beats a blank start, and free relaxation lands ---------
m0 = model.patterns[:, 0]
free = model.recall(m0.unsqueeze(0).repeat(4, 1) + 0.25 * torch.randn(4, d, generator=gen,
                                                                     dtype=model.dtype))
assert float(model.pattern_energy(free.T).max()) < 1e-4, "free relaxation did not reach the manifold"
assert float(free.norm(dim=1).min()) > 0.1, "free relaxation collapsed to the silent state"
known = torch.rand(d, generator=gen, dtype=model.dtype) < 0.75
done = model.recall(torch.zeros(d, dtype=model.dtype), known=known, cue=m0)
err = float((done[~known] - m0[~known]).norm() / m0[~known].norm())
assert torch.allclose(done[known], m0[known]), "clamp was not held"
print(f"6. recall: free relaxation lands on the manifold (F_S<1e-4, no collapse); "
      f"75% cue completes with relative error {err:.3f}")
assert err < 0.4, f"partial-cue completion failed (err={err:.3f})"

print("\nNONLINEAR TEST PASSED")
