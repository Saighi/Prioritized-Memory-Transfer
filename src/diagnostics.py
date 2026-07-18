"""Diagnostics: spectral quantities, the primary novelty-spectrum observable, alignments,
and three correctness checks tying the code to the paper's results (VFE gradients — Prop 1,
the surprise identity — Corollary 1, saddle circulation — Prop 3).
All pure functions over a model / tensors.
"""
from __future__ import annotations

import torch

from .model import TwoPopModel, bwd, fwd, outer
from .memory import zero_diag


# ----------------------------------------------------------------------------- spectral
def spectral_gap(S_T: torch.Tensor, tol: float = 1e-6) -> float:
    """Smallest NONZERO eigenvalue of S_T = M_T^T M_T (the spectral gap)."""
    evals = torch.linalg.eigvalsh(S_T)
    nonzero = evals[evals > tol]
    if nonzero.numel() == 0:
        return float(evals.max())
    return float(nonzero.min())


def manifold_basis(M_T: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis U_T of ker(M_T) = the teacher's memory manifold (columns with ~0 eigenvalue)."""
    S_T = M_T.transpose(-2, -1) @ M_T
    evals, evecs = torch.linalg.eigh(S_T)
    return evecs[:, evals < tol]


# --------------------------------------------------------------- primary novelty observable
def restricted_novelty_spectrum(model: TwoPopModel, U_T: torch.Tensor) -> torch.Tensor:
    """Eigenvalues (descending) of U_T^T N_S U_T — which directions of the teacher's manifold
    the student still cannot explain. The largest eigenvalue -> 0 marks transfer complete."""
    N = model.novelty_operator()
    R = U_T.transpose(-2, -1) @ N @ U_T
    return torch.linalg.eigvalsh(R).flip(-1)


def per_direction_residual(model: TwoPopModel, M: torch.Tensor) -> torch.Tensor:
    """||M_S m_p|| for each stored pattern: S's self-error residual along each named memory."""
    return (model.M_S @ M).norm(dim=0)


def alignment(x_T: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Signed cosine of x_T with each unit-norm pattern column of M -> (P,)."""
    return (M.transpose(-2, -1) @ x_T) / x_T.norm().clamp_min(1e-12)


def manifold_occupancy(x_T: torch.Tensor, U_T: torch.Tensor) -> float:
    """Fraction of x_T's norm lying inside the teacher's memory manifold."""
    return float((U_T.transpose(-2, -1) @ x_T).norm() / x_T.norm().clamp_min(1e-12))


# -------------------------------------------------------------------- correctness checks
def check_gradients(model: TwoPopModel, seed: int = 0):
    """Confirm that the student's perception = -grad_{x_S} F_S and the Hebbian rule =
    -grad_{W_S} F_S (full, unconstrained gradient). Returns (err_xS, err_WS), both ~1e-10."""
    d, dtype, device = model.d, model.dtype, model.device
    gen = torch.Generator(device=device).manual_seed(seed)
    xT = torch.randn(d, generator=gen, dtype=dtype, device=device)
    xS = torch.randn(d, generator=gen, dtype=dtype, device=device, requires_grad=True)
    WS = zero_diag(torch.randn(d, d, generator=gen, dtype=dtype, device=device)).requires_grad_(True)
    MS = torch.eye(d, dtype=dtype, device=device) - WS

    eps_TS = xS - xT
    eps_S = fwd(MS, xS)
    F_S = 0.5 * model.pi_TS * (eps_TS ** 2).sum() + 0.5 * model.pi_S * (eps_S ** 2).sum()
    g_xS, g_WS = torch.autograd.grad(F_S, [xS, WS])

    perception_rhs = -model.pi_TS * eps_TS.detach() - model.pi_S * bwd(MS.detach(), eps_S.detach())
    learn_rhs = model.pi_S * outer(eps_S.detach(), xS.detach())
    err_xS = float((g_xS + perception_rhs).norm())     # grad should equal -perception_rhs
    err_WS = float((g_WS + learn_rhs).norm())          # grad should equal -learn_rhs
    return err_xS, err_WS


def surprise_identity_error(model: TwoPopModel, n_trials: int = 5, seed: int = 0) -> float:
    """Corollary 1 (the surprise identity): at the fast-student equilibrium the student's free
    energy equals the novelty score of the teacher's state,

        F_S(x_S*, x_T) = (pi_TS/2) x_T^T N_S x_T .

    Settles the student (`solve_xS_steady`), evaluates F_S directly, and compares with the
    quadratic form. Returns the max absolute mismatch over `n_trials` random teacher states
    (~1e-15 in float64, for any W_S)."""
    gen = torch.Generator(device=model.device).manual_seed(seed)
    N = model.novelty_operator()
    err = 0.0
    for _ in range(n_trials):
        xT = torch.randn(model.d, generator=gen, dtype=model.dtype, device=model.device)
        xS = model.solve_xS_steady(xT)
        eps_TS = xS - xT
        eps_S = fwd(model.M_S, xS)
        F = 0.5 * model.pi_TS * (eps_TS ** 2).sum() + 0.5 * model.pi_S * (eps_S ** 2).sum()
        quad = 0.5 * model.pi_TS * (xT @ (N @ xT))
        err = max(err, float((F - quad).abs()))
    return err


# --------------------------------------------------------------- subspace diagnostics
def subspace_sum_basis(U1: torch.Tensor, U2: torch.Tensor, tol: float = 1e-6):
    """Orthonormal basis `U_Sigma` of the subspace sum `U1 + U2 = span(U1 ∪ U2)` plus the
    intersection dimension and `r_Sigma = dim(U1 + U2)`. `U1`, `U2` are assumed to have
    orthonormal columns (e.g. from `manifold_basis`)."""
    C = torch.cat([U1, U2], dim=1)
    Uc, S, _ = torch.linalg.svd(C, full_matrices=False)
    keep = S > tol * S.max() if S.numel() else S
    r_sigma = int(keep.sum())
    U_sigma = Uc[:, :r_sigma]
    overlap = U1.shape[1] + U2.shape[1] - r_sigma
    return U_sigma, overlap, r_sigma


def transfer_deficit(M_S: torch.Tensor, U: torch.Tensor) -> float:
    """`||M_S U||_F^2` — how far a plastic network is from nulling every direction of `U`.
    Zero iff it has learned the whole subspace `span(U)`."""
    return float((M_S @ U).pow(2).sum())


def circulation(model: TwoPopModel, seed: int = 0) -> float:
    """Frobenius norm of the saddle mismatch  d f_T/d x_S + (d f_S/d x_T)^T.
    For a true single-potential saddle it is 0; with the signed precision it equals
    |pi_ST + pi_TS| * sqrt(d), i.e. ~0 only when pi_ST == -pi_TS (the exact-saddle / zero-sum
    regime: the teacher's reversed precision exactly mirrors the student's)."""
    d, dtype, device = model.d, model.dtype, model.device
    gen = torch.Generator(device=device).manual_seed(seed)
    xT = torch.randn(d, generator=gen, dtype=dtype, device=device)
    xS = torch.randn(d, generator=gen, dtype=dtype, device=device)
    M_T, M_S = model.M_T, model.M_S

    def f_T(xt, xs):
        return -model.pi_T * bwd(M_T, fwd(M_T, xt)) + model.pi_ST * (xs - xt)

    def f_S(xt, xs):
        return -model.pi_TS * (xs - xt) - model.pi_S * bwd(M_S, fwd(M_S, xs))

    J_fT_xS = torch.autograd.functional.jacobian(lambda xs: f_T(xT, xs), xS)
    J_fS_xT = torch.autograd.functional.jacobian(lambda xt: f_S(xt, xS), xT)
    return float((J_fT_xS + J_fS_xT.transpose(-2, -1)).norm())
