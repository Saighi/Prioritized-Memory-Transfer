"""Pattern generation and construction of the teacher's frozen recurrent weights `W_T`.

Faithfulness notes:
  - `W_T` and `W_S` carry a ZERO DIAGONAL at all times (no autapses). `build_W_T` produces
    a zero-diagonal matrix by construction.
  - The default `covpcn` build learns, per node, the zero-diagonal linear predictor of that
    node from the *others* across all patterns. For P <= d-1 this drives the residual
    `M_T m_p = m_p - W_T m_p` to ~0, so the patterns span `ker M_T` (T is "flat") while the
    diagonal stays exactly zero. This is the linear covariance-PCN memory.
  - The `projector` build uses the ideal orthogonal projector onto span(patterns) with the
    diagonal zeroed afterwards — handy for clean-theory comparisons.
  - The same construction also pre-seeds a student that already "knows" a subset of patterns
    (see `TwoPopModel.pretrain`).
"""
from __future__ import annotations

import math

import torch

from .config import ModelConfig


def zero_diag(W: torch.Tensor) -> torch.Tensor:
    """Return a copy of W with its diagonal set to zero (no autapses)."""
    return W - torch.diag_embed(torch.diagonal(W, dim1=-2, dim2=-1))


def make_patterns(cfg: ModelConfig) -> torch.Tensor:
    """Return stored memories as a (d, P) matrix with unit-norm columns."""
    gen = torch.Generator().manual_seed(cfg.seed)
    d, P = cfg.d, cfg.P
    if cfg.pattern_kind == "orthonormal":
        A = torch.randn(d, P, generator=gen, dtype=cfg.dtype)
        Q, _ = torch.linalg.qr(A)          # reduced QR -> (d, P), orthonormal columns
        M = Q[:, :P]
    elif cfg.pattern_kind == "correlated":
        r = cfg.corr_rank or max(1, P // 2)
        B = torch.randn(d, r, generator=gen, dtype=cfg.dtype)
        C = torch.randn(r, P, generator=gen, dtype=cfg.dtype)
        M = B @ C + cfg.corr_noise * torch.randn(d, P, generator=gen, dtype=cfg.dtype)
    else:
        raise ValueError(f"unknown pattern_kind={cfg.pattern_kind!r}")
    M = M / M.norm(dim=0, keepdim=True)
    return M.to(cfg.device)


def make_teacher_subspaces(cfg) -> tuple:
    """Two teacher pattern matrices `(M1, M2)` with unit-norm columns and *controlled geometry*.

    Used by the additive three-network model (`pmt.additive`) to place the two frozen memory
    subspaces `U1 = span(M1)`, `U2 = span(M2)` in a prescribed relationship (spec section 20):

      - "orthogonal": `U1 ⟂ U2` (disjoint blocks of a shared random orthonormal frame);
      - "shared":     a common block of `cfg.overlap` directions belongs to both, so
                      `dim(U1 ∩ U2) = overlap` and `r_Σ = rank1 + rank2 − overlap`;
      - "oblique":    no exact intersection but small principal angle `cfg.principal_angle`
                      between paired directions.

    `cfg` must carry `d, rank1, rank2, geometry, overlap, principal_angle, seed, dtype, device`.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    d, r1, r2 = cfg.d, cfg.rank1, cfg.rank2
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=cfg.dtype))  # random ON frame

    if cfg.geometry == "orthogonal":
        if r1 + r2 > d:
            raise ValueError(f"orthogonal needs rank1+rank2 <= d (got {r1}+{r2} > {d}).")
        U1 = Q[:, :r1]
        U2 = Q[:, r1:r1 + r2]
    elif cfg.geometry == "shared":
        ov = cfg.overlap
        if ov > min(r1, r2):
            raise ValueError(f"overlap <= min(rank1,rank2) required (got {ov} > {min(r1, r2)}).")
        if r1 + (r2 - ov) > d:
            raise ValueError(f"shared geometry needs rank1+rank2-overlap <= d.")
        shared = Q[:, :ov]
        U1 = Q[:, :r1]                                   # cols 0..r1-1 (includes the shared block)
        own2 = Q[:, r1:r1 + (r2 - ov)]                   # T2's private directions
        U2 = torch.cat([shared, own2], dim=1)
    elif cfg.geometry == "oblique":
        theta = float(cfg.principal_angle)
        m = min(r1, r2)
        cols = []
        for i in range(r2):
            if i < m:                                    # tilt U1's i-th direction by angle theta
                cols.append(math.cos(theta) * Q[:, i] + math.sin(theta) * Q[:, r1 + i])
            else:                                        # any extra U2 directions are fresh
                cols.append(Q[:, r1 + i])
        U1 = Q[:, :r1]
        U2 = torch.stack(cols, dim=1)
    else:
        raise ValueError(f"unknown geometry={cfg.geometry!r}")

    M1 = U1 / U1.norm(dim=0, keepdim=True)
    M2 = U2 / U2.norm(dim=0, keepdim=True)
    return M1.to(cfg.device), M2.to(cfg.device)


def build_W_T(M: torch.Tensor, cfg, ridge: float = 1e-8) -> torch.Tensor:
    """Build frozen, zero-diagonal recurrent weights from patterns M (d, P).

    Used for the teacher's `W_T`, and (via `pretrain`) to pre-seed a student that already
    holds a subset of the patterns.
    """
    d = M.shape[0]
    dtype, device = M.dtype, M.device
    if cfg.W_T_kind == "projector":
        G = M.T @ M
        Pj = M @ torch.linalg.solve(G + ridge * torch.eye(G.shape[0], dtype=dtype, device=device), M.T)
        return zero_diag(Pj)
    if cfg.W_T_kind == "covpcn":
        # Per-node zero-diagonal least squares: row i predicts component i of each pattern
        # from the other components. X has patterns as rows (P, d).
        X = M.T
        W = torch.zeros(d, d, dtype=dtype, device=device)
        for i in range(d):
            others = [j for j in range(d) if j != i]
            Xi = X[:, others]                  # (P, d-1)
            t = X[:, i]                        # (P,)
            A_ = Xi.T @ Xi + ridge * torch.eye(d - 1, dtype=dtype, device=device)
            w = torch.linalg.solve(A_, Xi.T @ t)
            W[i, others] = w
        return W                                # zero diagonal by construction
    raise ValueError(f"unknown W_T_kind={cfg.W_T_kind!r}")


def memory_residual(W_T: torch.Tensor, M: torch.Tensor) -> float:
    """max_p ||M_T m_p|| with M_T = I - W_T. Should be ~0 for a faithful covPCN/projector."""
    M_T = torch.eye(W_T.shape[0], dtype=W_T.dtype, device=W_T.device) - W_T
    return (M_T @ M).norm(dim=0).max().item()
