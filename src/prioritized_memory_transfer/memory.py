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
  - `act` handles populations with a unit nonlinearity: the fit predicts the *state* of node i
    from the *transmitted outputs* `f(m_p)` of the others. That is still an ordinary linear
    least-squares problem, because `f(M)` is fixed data. For nonnegative patterns under a
    rectifying `f` we have `f(m_p) = m_p`, so it reduces exactly to the linear fit and the
    memories are exact zero-error fixed points of the nonlinear network too.
"""
from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import torch

from .activations import ActivationSpec, resolve_activation
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
    elif cfg.pattern_kind in ("target_corr", "target_corr_nonneg"):
        M = make_correlated(d, P, cfg.corr_target, dtype=cfg.dtype, seed=cfg.seed,
                            nonneg=cfg.pattern_kind.endswith("_nonneg"))
    elif cfg.pattern_kind == "nonneg":
        # Rectified Gaussian: sparse NONNEGATIVE patterns, the natural code for rate units.
        # Under a rectifying activation f(m_p) = m_p, so these are exact zero-error fixed points
        # of the nonlinear network and `build_memory` needs no special casing. Nonnegative
        # combinations of them are memories too (the manifold is the patterns' convex cone);
        # negative combinations leave the orthant, get rectified, and acquire error -- which is
        # what turns the linear memory *subspace* into something more memory-like.
        M = torch.randn(d, P, generator=gen, dtype=cfg.dtype).clamp_min(0.0)
        if bool((M.norm(dim=0) <= 0).any()):
            raise ValueError("degenerate nonneg pattern draw (an all-zero column); change seed.")
    else:
        raise ValueError(f"unknown pattern_kind={cfg.pattern_kind!r}")
    M = M / M.norm(dim=0, keepdim=True)
    return M.to(cfg.device)


def make_correlated(
    d: int,
    P: int,
    target_corr,
    *,
    nonneg: bool = False,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    n_steps: int = 3000,
    lr: float = 0.05,
) -> torch.Tensor:
    """Unit-norm patterns with a PRESCRIBED pairwise correlation, by projected gradient descent.

    `M^T M` is the table of every pairwise correlation of the columns (the Gram matrix): entry
    `(p, q)` is `m_p . m_q`, which is the cosine once the columns are unit-norm. So asking for a
    correlation structure is just fitting that table to a target `G*`, by minimizing

        L(M) = 0.5 ||M^T M - G*||_F^2 ,      grad_M L = 2 M (M^T M - G*)

    which reads as a system of springs between patterns: each pair is pushed apart when it is more
    correlated than asked and pulled together when it is less, in proportion to the error.

    One tool for both pattern types -- signed and nonnegative differ ONLY by the projection: after
    each step the columns are clamped to `>= 0` (when `nonneg`) and renormalized, so the diagonal
    stays 1 and the off-diagonals really are cosines.

    `target_corr` is either a scalar `rho` (uniform: `G* = (1-rho) I + rho 11^T`) or an explicit
    `(P, P)` target Gram, which is how to ask for a rank-deficient or non-uniform structure.

    Feasibility, since unreachable targets simply leave the loss on a plateau: `G*` must be PSD
    (uniform case: `rho >= -1/(P-1)`), and with `nonneg=True` also entrywise nonnegative
    (`rho >= 0`) and completely positive -- for `P >= 5` the latter is strictly stronger. The
    achieved mean correlation is checked against the target and warned about, not silently
    returned.
    """
    gen = torch.Generator().manual_seed(int(seed))
    if isinstance(target_corr, torch.Tensor):
        G = target_corr.to(dtype)
        if G.shape != (P, P):
            raise ValueError(f"an explicit target Gram must have shape ({P}, {P}) (got {tuple(G.shape)}).")
    else:
        rho = float(target_corr)
        G = (1.0 - rho) * torch.eye(P, dtype=dtype) + rho * torch.ones(P, P, dtype=dtype)

    M = torch.randn(d, P, generator=gen, dtype=dtype)
    if nonneg:
        M = M.abs()
    M = M / M.norm(dim=0, keepdim=True).clamp_min(1e-12)
    for _ in range(int(n_steps)):
        # grad = 2 M (M^T M - G); the factor 2 is folded into lr
        M = M - lr * (M @ (M.transpose(-2, -1) @ M - G))
        if nonneg:
            M = M.clamp_min(0.0)
        M = M / M.norm(dim=0, keepdim=True).clamp_min(1e-12)

    if P > 1:
        off = ~torch.eye(P, dtype=torch.bool)
        got = float((M.transpose(-2, -1) @ M)[off].mean())
        want = float(G[off].mean())
        if abs(got - want) > 0.02:
            warnings.warn(
                f"target correlation not reached: asked {want:.3f}, got {got:.3f}. The target is "
                "probably infeasible (it must be PSD, and nonnegative + completely positive when "
                "nonneg=True); more steps will not help."
            )
    return M


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
        U1 = Q[:, :r1]
        U2 = Q[:, r1:r1 + r2]
    elif cfg.geometry == "shared":
        ov = cfg.overlap
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
    act: ActivationSpec = None,
) -> SimpleNamespace:
    """Fit a zero-diagonal covariance-PCN memory to pattern columns ``M``.

    Reject pattern sets that cannot be stored within ``tolerance`` under the no-autapse
    constraint instead of silently returning weights that do not encode the requested memory.

    With `act` set, node i is fit to predict its own *state* from the *transmitted outputs*
    `f(m_p)` of the other nodes, and the residual measured is `||m_p - W f(m_p)||`. Both reduce
    to the linear case when `f` is the identity — and also when the patterns are nonnegative and
    `f` rectifies, since then `f(M) = M`.
    """
    _validate_pattern_matrix(M)
    if ridge < 0 or tolerance <= 0:
        raise ValueError("ridge must be non-negative and tolerance must be positive.")
    f, _ = resolve_activation(act)

    d = M.shape[0]
    dtype, device = M.dtype, M.device
    # Per-node zero-diagonal least squares: row i predicts component i of each pattern from the
    # other components' outputs. X has patterns as rows (P, d); Z = f(X) is what the others send.
    X = M.T
    Z = f(X)
    W = torch.zeros(d, d, dtype=dtype, device=device)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        Zi = Z[:, others]                  # (P, d-1) transmitted outputs of the other nodes
        target = X[:, i]                   # (P,) state of node i
        A = Zi.T @ Zi + ridge * torch.eye(d - 1, dtype=dtype, device=device)
        W[i, others] = torch.linalg.solve(A, Zi.T @ target)

    M_op = torch.eye(d, dtype=dtype, device=device) - W
    residuals = (M - W @ f(M)).norm(dim=0)     # == (M_op @ M) in the linear case
    max_residual = float(residuals.max()) if residuals.numel() else 0.0
    if max_residual > tolerance:
        raise ValueError(
            "patterns are not representable by a zero-diagonal covariance-PCN: "
            f"max ||m_p - W f(m_p)||={max_residual:.3e}. "
            "P <= d-1 is necessary but not sufficient; "
            "each neuron's pattern values must be predictable from the remaining neurons."
        )
    return SimpleNamespace(W=W, M_op=M_op, max_residual=max_residual)
