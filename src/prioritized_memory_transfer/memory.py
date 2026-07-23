"""Pattern generation and construction of zero-diagonal covariance-PCN memories.

Faithfulness notes:
  - `W_T` and `W_S` carry a ZERO DIAGONAL at all times (no autapses). `build_memory` produces
    a zero-diagonal matrix by construction.
  - The default `covpcn` build learns, per node, the zero-diagonal linear predictor of that
    node from the *others* across all patterns. For generic dense patterns with P <= d-1 this
    drives `M_T m_p = m_p - W_T m_p` to ~0. P <= d-1 is necessary but not sufficient for
    arbitrary patterns: every neuron's values must be predictable from the other neurons.
    `build_memory` therefore measures representability and rejects an invalid fit.
  - The same construction also pre-seeds a student that already "knows" a subset of patterns
    (see `TwoPopModel.pretrain`).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from .config import ModelConfig


def zero_diag(W: torch.Tensor) -> torch.Tensor:
    """Return a copy of W with its diagonal set to zero (no autapses)."""
    return W - torch.diag_embed(torch.diagonal(W, dim1=-2, dim2=-1))


def make_patterns(cfg: ModelConfig) -> torch.Tensor:
    """Return stored memories as a `(d, P)` matrix with unit-norm columns.

    ``pattern_kind="correlated"`` is exactly rank ``corr_rank`` only when ``corr_noise=0``.
    Positive noise keeps the columns correlated but makes the set generically full-rank.
    """
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

    Used by the interleaved model (`prioritized_memory_transfer.interleaved`) to place the two frozen memory
    subspaces `U1 = span(M1)`, `U2 = span(M2)` in a prescribed relationship:

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


def _validate_pattern_matrix(M: torch.Tensor) -> None:
    if not isinstance(M, torch.Tensor):
        raise TypeError(f"patterns must be a torch.Tensor (got {type(M).__name__}).")
    if M.ndim != 2:
        raise ValueError(f"patterns must have shape (d, P) (got shape {tuple(M.shape)}).")
    if M.shape[0] < 2:
        raise ValueError(f"patterns need d >= 2 rows (got d={M.shape[0]}).")
    if not M.is_floating_point():
        raise TypeError(f"patterns must use a floating dtype (got {M.dtype}).")
    if not bool(torch.isfinite(M).all()):
        raise ValueError("patterns contain NaN or infinite values.")
    if M.shape[1] and bool((M.norm(dim=0) <= 0).any()):
        raise ValueError("every pattern column must have non-zero norm.")


def build_memory(
    M: torch.Tensor,
    *,
    ridge: float = 1e-8,
    tolerance: float = 1e-3,
) -> SimpleNamespace:
    """Fit a zero-diagonal covariance-PCN memory to pattern columns ``M``.

    Reject pattern sets that cannot be stored within ``tolerance`` under the no-autapse
    constraint instead of silently returning weights that do not encode the requested memory.
    """
    _validate_pattern_matrix(M)
    if ridge < 0 or tolerance <= 0:
        raise ValueError("ridge must be non-negative and tolerance must be positive.")

    d = M.shape[0]
    dtype, device = M.dtype, M.device
    # Per-node zero-diagonal least squares: row i predicts component i of each pattern
    # from the other components. X has patterns as rows (P, d).
    X = M.T
    W = torch.zeros(d, d, dtype=dtype, device=device)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        Xi = X[:, others]                  # (P, d-1)
        target = X[:, i]                   # (P,)
        A = Xi.T @ Xi + ridge * torch.eye(d - 1, dtype=dtype, device=device)
        W[i, others] = torch.linalg.solve(A, Xi.T @ target)

    M_op = torch.eye(d, dtype=dtype, device=device) - W
    residuals = (M_op @ M).norm(dim=0)
    max_residual = float(residuals.max()) if residuals.numel() else 0.0
    if max_residual > tolerance:
        raise ValueError(
            "patterns are not representable by a zero-diagonal covariance-PCN: "
            f"max ||(I-W)m_p||={max_residual:.3e}. "
            "P <= d-1 is necessary but not sufficient; "
            "each neuron's pattern values must be predictable from the remaining neurons."
        )
    return SimpleNamespace(W=W, M_op=M_op, max_residual=max_residual)


def memory_residual(W_T: torch.Tensor, M: torch.Tensor) -> float:
    """Return ``max_p ||(I-W_T)m_p||`` for an already-built memory."""
    _validate_pattern_matrix(M)
    if W_T.ndim != 2 or W_T.shape != (M.shape[0], M.shape[0]):
        raise ValueError(
            f"W_T must have shape ({M.shape[0]}, {M.shape[0]}) (got {tuple(W_T.shape)})."
        )
    M_T = torch.eye(W_T.shape[0], dtype=W_T.dtype, device=W_T.device) - W_T
    residuals = (M_T @ M).norm(dim=0)
    return residuals.max().item() if residuals.numel() else 0.0
